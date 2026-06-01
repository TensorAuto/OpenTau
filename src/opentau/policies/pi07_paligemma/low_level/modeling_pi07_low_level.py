#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""π05: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi05.pdf)
"""

import builtins
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import NormalizationMode
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.normalize import resolve_num_datasets as _num_datasets
from opentau.policies.outlier_utils import detect_state_action_outliers
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import _global_or_branch_decisions
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.policies.utils import flow_matching_masked_mse
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.utils import get_safe_dtype

ItemType = Literal["text", "image", "video", "state", "discrete_action", "action"]
AttentionMode = Literal["continue", "bidirectional", "causal"]


@dataclass
class ContextItem:
    """One token-block contributed to the prefix or suffix sequence.

    The ``PI07PaligemmaLowLevelPolicy`` builds a list of ``ContextItem``s and
    hands it to the ``PI07PaligemmaLowLevelFlowMatching`` model. The model
    embeds each item by ``item_type`` (text, video, state, discrete_action,
    action), concatenates the embeddings, and derives the 1-D attention
    mask from each item's ``attention`` setting. To add or rearrange
    blocks in the prefix/suffix, edit the policy's list construction —
    the model is layout-agnostic.

    Fields:
        data: Per-type input. ``text``/``discrete_action``: token IDs
            ``(B, L)``. ``video``: pixel video ``(B, T, C, H, W)`` in
            ``[0, 1]``. ``image``: single pixel image ``(B, C, H, W)`` in
            ``[-1, 1]`` (subgoal images, embedded via the VLM's image tower).
            ``state``: continuous state ``(B, T, max_state_dim)``.
            ``action``: noisy actions ``(B, chunk_size, max_action_dim)``.
        item_type: Dispatch key for which embedding path to take.
        pad_mask: Padding mask. ``(B,)`` per-sample is allowed for
            ``video``/``image`` (the model expands it to
            ``(B, num_image_tokens)``). All other types must pass ``(B, L)``
            matching the embedded sequence length. ``True`` = real,
            ``False`` = padded.
        attention: 1-D attention pattern for this block:
            - ``"continue"``: ``[0]*L`` — token continues the previous
              block's attention scope (bidirectional with everything before).
            - ``"bidirectional"``: ``[1] + [0]*(L-1)`` — opens a new block
              boundary, then bidirectional within the block.
            - ``"causal"``: ``[1]*L`` — every token starts its own scope
              (fully causal within this block).
        exclude_from_cross_attention: When ``True``, this item's tokens are
            not cross-attended to by the suffix (action expert). Used for
            the trailing ``"Action: "`` indicator + discrete-action block,
            which the action expert must not condition on.
        obs_history_is_pad: Optional ``(B, T)`` mask threaded into the
            video encoder's temporal attention (only used by ``video``
            items). ``True`` marks padded history frames.
    """

    data: Tensor
    item_type: ItemType
    pad_mask: Tensor
    attention: AttentionMode = "continue"
    exclude_from_cross_attention: bool = False
    obs_history_is_pad: Tensor | None = None


def _preferred_dtype():
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: torch.device | str = "cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: A 2-D tensor of shape (batch_size, action_chunk_length).
        dimension: The dimension of the embedding vectors. Must be divisible by 2.
        min_period: The minimum period of the sinusoidal functions.
        max_period: The maximum period of the sinusoidal functions.
        device: The device to create the tensors on. Defaults to "cpu".

    Returns:
        A tensor of shape (batch_size, action_chunk_length, dimension) containing the positional embeddings.

    Raises:
        ValueError: If dimension is not divisible by 2 or if time tensor is not 2-D with shape (batch_size, action_chunk_length).
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 2:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, action_chunk_length)`.")

    dtype = (
        get_safe_dtype(torch.float64, device.type)
        if isinstance(device, torch.device)
        else get_safe_dtype(torch.float64, device)
    )
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = rearrange(scaling_factor, "d -> 1 1 d") * rearrange(time, "b c -> b c 1")
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=2)
    return pos_emb


def make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
    n_cross_att_tokens: int | None = None,
    cross_att_pad_masks: Tensor | None = None,
) -> Tensor:
    """Creates a 2-D attention mask given padding and 1-D attention masks.

    Tokens can attend to valid inputs tokens which have a cumulative `att_masks`
    smaller or equal to theirs. This way `att_masks` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
        pad_masks: bool[B, N] true if its part of the input, false if padding.
        att_masks: int32[B, N] mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.
        n_cross_att_tokens: Add attention mask for cross-attention tokens if
            `n_cross_att_tokens` is provided.
        cross_att_pad_masks: Padding masks for cross attention tokens. Required if
            `n_cross_att_tokens` is provided.

    Returns:
        A 2D attention mask tensor of shape (B, N + n_cross_att_tokens, N + n_cross_att_tokens)
        if n_cross_att_tokens is provided, else (B, N, N).

    Raises:
        ValueError: If att_masks or pad_masks are not 2D (including batch dimension).
        AssertionError: If cross_att_pad_masks is missing when n_cross_att_tokens is set,
            or if its shape is incorrect.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    # If `n_cross_att_tokens` is provided, we add a mask for cross-attention tokens at the end of the sequence.
    if n_cross_att_tokens is not None:
        assert cross_att_pad_masks is not None, (
            "cross_att_pad_masks must be provided if n_cross_att_tokens is provided"
        )
        assert cross_att_pad_masks.shape == (att_masks.size(0), n_cross_att_tokens), (
            "cross_att_pad_masks must have shape (batch_size, n_cross_att_tokens)"
        )

        cross_att_mask = torch.full(
            (att_masks.size(0), att_masks.size(1), n_cross_att_tokens),
            True,
            dtype=torch.bool,
            device=att_masks.device,
        )

        # Apply padding masks: pad_masks for rows, cross_att_pad_masks for columns
        cross_att_mask = cross_att_mask & pad_masks[:, :, None] & cross_att_pad_masks[:, None, :]
        # The cross_att_masks are concatenated before the att_2d_masks
        att_2d_masks = torch.cat((cross_att_mask, att_2d_masks), dim=2)

    return att_2d_masks


def resize_with_pad(img: Tensor, width: int, height: int, pad_value: int = -1) -> Tensor:
    """Resizes an image to fit within the specified dimensions while maintaining aspect ratio,
    and pads the remaining area with the specified value.

    Args:
        img: Input image tensor of shape (batch_size, channels, current_height, current_width).
        width: Target width.
        height: Target height.
        pad_value: Value to use for padding. Defaults to -1.

    Returns:
        The resized and padded image tensor of shape (batch_size, channels, height, width).

    Raises:
        ValueError: If the input image tensor does not have 4 dimensions.
    """
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_discrete_tokens(tokens: list[list[int]], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pads or truncates a list of discrete action token sequences to a fixed length.

    Args:
        tokens: A list of discrete action token sequences (lists of integers).
        max_length: The target length for the discrete action token sequences.

    Returns:
        A tuple containing:
            - discrete_action_tokens: A numpy array of shape (len(tokens), max_length) containing the padded discrete action tokens.
            - discrete_action_masks: A boolean numpy array of shape (len(tokens), max_length) indicating valid discrete action tokens (True) and padding (False).
    """
    discrete_action_tokens = []
    discrete_action_masks = []
    for token in tokens:
        if len(token) > max_length:
            logging.warning(
                f"Discrete action token length {len(token)} is greater than max_length {max_length}, truncating"
            )
            discrete_action_tokens.append(np.array(token[:max_length]))
            discrete_action_masks.append(np.ones(max_length, dtype=bool))
        else:
            discrete_action_masks.append(
                np.concatenate(
                    [np.ones(len(token), dtype=bool), np.zeros(max_length - len(token), dtype=bool)]
                )
            )
            discrete_action_tokens.append(np.pad(token, (0, max_length - len(token)), constant_values=0))
    return np.array(discrete_action_tokens), np.array(discrete_action_masks)


class PI07PaligemmaLowLevelPolicy(PreTrainedPolicy):
    """Wrapper class around PI07PaligemmaLowLevelFlowMatching model to train and run inference within OpenTau."""

    config_class = PI07PaligemmaLowLevelConfig
    name = "pi07_paligemma_low_level"

    def __init__(
        self,
        config: PI07PaligemmaLowLevelConfig,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
    ):
        """Initializes the PI07PaligemmaLowLevelPolicy.

        Args:
            config: Policy configuration class instance.
            per_dataset_stats: Ordered list of per-dataset stat dicts used to
                fill the stacked Normalize/Unnormalize buffers.
            dataset_names: Ordered list parallel to ``per_dataset_stats``.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        num_datasets = _num_datasets(per_dataset_stats, dataset_names, config)
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
        )
        self.normalize_discrete_actions = Normalize(
            config.output_features,
            {"ACTION": NormalizationMode.MIN_MAX},
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            config.discrete_action_tokenizer_path, trust_remote_code=True
        )
        # Get vocab size from processor
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI07PaligemmaLowLevelFlowMatching(
            config, discrete_action_vocab_size=discrete_action_vocab_size
        )

        self.reset()

    def reset(self) -> None:
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._state_buffer: deque | None = None
        self._obs_buffers: dict[str, deque] = {}

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping.

        Args:
            pretrained_name_or_path: Path to the pretrained model or its name on the Hub.
            config: Configuration object.
            force_download: Whether to force download the model weights.
            resume_download: Whether to resume download.
            proxies: Proxy configuration.
            token: Authentication token.
            cache_dir: Directory to cache downloaded files.
            local_files_only: Whether to only look for files locally.
            revision: Specific model revision.
            strict: Whether to strictly enforce state dict matching.
            **kwargs: Additional keyword arguments.

        Returns:
            The loaded model instance.

        Raises:
            ValueError: If pretrained_name_or_path is None.
        """
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        acc = get_proc_accelerator()
        is_main_process = acc.is_main_process if acc else True
        # Populated inside the try block when skip_normalization_weights fires;
        # used outside the try/except to gate the inf-buffer guard so the
        # ValueError is not swallowed by the broad `except Exception` that
        # otherwise just warns and returns the (broken) model.
        stripped_keys: frozenset[str] = frozenset()
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            if is_main_process:
                logging.info("Loading model from: %s", pretrained_name_or_path)
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                if is_main_process:
                    logging.info("Loaded state dict from model.safetensors")
            except Exception as e:
                if is_main_process:
                    logging.warning("Could not load state dict from remote files: %s", e)
                    logging.info("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model.") and "normalize" not in key:
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10 and is_main_process:
                        logging.debug("Remapped: %s -> %s", key, new_key)
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0 and is_main_process:
                logging.info("Remapped %d state dict keys", remap_count)

            # Strip saved normalize/unnormalize buffers when the user opted in
            # via config.skip_normalization_weights — see PreTrainedConfig and
            # PreTrainedPolicy._strip_normalization_buffers_from_state_dict.
            remapped_state_dict, stripped_keys = cls._strip_normalization_buffers_from_state_dict(
                remapped_state_dict, model.config, is_main_process=is_main_process
            )

            # Load the remapped state dict into the model
            # Promote legacy single-dataset Normalize/Unnormalize buffers from
            # `(*feat_shape,)` to the new `(1, *feat_shape)` stacked layout so pre-PR
            # checkpoints load via `model.load_state_dict(...)`.
            model._promote_legacy_norm_buffers_in_state_dict(remapped_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            # Hide deliberately-stripped buffer keys from the missing-keys
            # warning so the noisy WARNING does not directly contradict the
            # INFO logged just above. ``stripped_keys`` is empty when the
            # flag is off, so this is a no-op for default loads.
            unintended_missing = [key for key in missing_keys if key not in stripped_keys]

            if unintended_missing and is_main_process:
                logging.warning("Missing keys when loading state dict: %d keys", len(unintended_missing))
                for key in unintended_missing[:20]:
                    logging.warning("  - %s", key)
                if len(unintended_missing) > 20:
                    logging.warning("  ... and %d more", len(unintended_missing) - 20)

            if unexpected_keys and is_main_process:
                logging.warning("Unexpected keys when loading state dict: %d keys", len(unexpected_keys))
                for key in unexpected_keys[:20]:
                    logging.warning("  - %s", key)
                if len(unexpected_keys) > 20:
                    logging.warning("  ... and %d more", len(unexpected_keys) - 20)

            if not unintended_missing and not unexpected_keys and is_main_process:
                logging.info("All keys loaded successfully!")

        except Exception as e:
            if is_main_process:
                logging.warning("Could not remap state dict keys: %s", e)

        # Outside the try/except so the ValueError is not swallowed by the
        # broad ``except Exception`` that otherwise just warns and returns the
        # (broken) model. The helper itself no-ops when ``stripped_keys`` is
        # empty (flag was off or the try block bailed before the strip ran).
        cls._assert_normalize_buffers_initialized(model, stripped_keys=stripped_keys)

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, Tensor], model_config: PreTrainedConfig
    ) -> dict[str, Tensor]:  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture.

        Args:
            state_dict: The state dictionary to fix.
            model_config: The model configuration.

        Returns:
            The fixed state dictionary.
        """
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        """Returns the parameters to be optimized.

        Returns:
            A generator over the model parameters.
        """
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Batch of data containing environment observations.

        Returns:
            The predicted action chunk.

        Raises:
            NotImplementedError: Always, as this method is not implemented for PI05.
        """
        raise NotImplementedError("Currently not implemented for PI05")

    def _build_history_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Buffer the current observations and construct a temporal batch.

        Appends single-step state ``(B, D)`` and single-frame images
        ``(B, C, H, W)`` from ``batch`` into internal deques, then
        assembles a batch with ``n_obs_steps`` evenly-spaced frames
        (interval = ``history_interval``).  Early in an episode, missing
        history slots are zero-padded and marked in ``obs_history_is_pad``.

        Returns a new dict with:
        - ``"state"`` expanded to ``(B, T, D)``
        - each image key expanded to ``(B, T, C, H, W)``
        - ``"obs_history_is_pad"`` as ``(B, T)`` bool
        """
        n_hist: int = self.config.n_obs_steps
        interval = self.config.history_interval
        buf_maxlen = self.config.obs_buffer_size

        # --- buffer state ---
        if self._state_buffer is None:
            self._state_buffer = deque(maxlen=buf_maxlen)
        self._state_buffer.append(batch["state"])  # (B, D)

        # --- buffer images ---
        img_keys = [k for k in self.config.image_features if k in batch]
        for key in img_keys:
            if key not in self._obs_buffers:
                self._obs_buffers[key] = deque(maxlen=buf_maxlen)
            self._obs_buffers[key].append(batch[key])  # (B, C, H, W)

        buf_len = len(self._state_buffer)
        missing = buf_maxlen - buf_len

        # Keys that are neither state nor image pass through unchanged.
        skip_keys = {"state"} | set(img_keys)
        temporal_batch = {k: v for k, v in batch.items() if k not in skip_keys}

        # --- assemble state history ---
        state_frames = []
        is_pad = []
        for i in range(n_hist):
            idx = i * interval - missing
            if idx < 0:
                state_frames.append(torch.zeros_like(self._state_buffer[0]))
                is_pad.append(True)
            else:
                state_frames.append(self._state_buffer[idx])
                is_pad.append(False)
        temporal_batch["state"] = torch.stack(state_frames, dim=1)  # (B, T, D)

        # --- assemble image history (per camera) ---
        for key in img_keys:
            cam_buf = self._obs_buffers[key]
            frames = []
            for i in range(n_hist):
                idx = i * interval - missing
                if idx < 0:
                    frames.append(torch.zeros_like(cam_buf[0]))
                else:
                    frames.append(cam_buf[idx])
            temporal_batch[key] = torch.stack(frames, dim=1)  # (B, T, C, H, W)

        bsize = temporal_batch["state"].shape[0]
        device = temporal_batch["state"].device
        temporal_batch["obs_history_is_pad"] = (
            torch.tensor(is_pad, dtype=torch.bool, device=device).unsqueeze(0).expand(bsize, -1)
        )  # (B, T)

        return temporal_batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action from the queue, regenerating if needed.

        Builds temporal state history when configured, runs flow-matching
        inference to fill the action queue, and pops the next action.

        Args:
            batch: Environment observation dict.
            noise: Optional pre-sampled noise for deterministic evaluation.

        Returns:
            A single action tensor of shape ``(B, action_dim)``.
        """
        self.eval()

        if self.config.n_obs_steps > 1:
            batch = self._build_history_batch(batch)

        if len(self._action_queue) == 0 or len(self._action_queue) <= self.config.max_delay:
            # Use current queue as action prefix to replenish
            action_prefix = None
            delay = 0
            if len(self._action_queue) > 0:
                prefix_actions = list(self._action_queue)
                delay = min(len(prefix_actions), self.config.max_delay)
                assert delay == self.config.max_delay, f"Delay must be equal to {self.config.max_delay}"
                prefix_actions = prefix_actions[-delay:]
                action_prefix = torch.stack(prefix_actions, dim=1)
            delay = torch.tensor(delay, dtype=torch.long, device=batch["state"].device)
            actions = self.sample_actions(batch, noise=noise, action_prefix=action_prefix, delay=delay)
            actions = rearrange(actions, "b c d -> c b d")
            # Execute only the first n_action_steps of the predicted chunk, then
            # re-query with fresh observations (receding horizon). The config guard
            # (n_action_steps < chunk_size => max_delay == 0 => delay == 0) keeps this
            # slice in range and exactly n_action_steps long; when n_action_steps ==
            # chunk_size it clamps to actions[delay:] (unchanged behaviour).
            self._action_queue.extend(actions[delay : delay + self.config.n_action_steps])
            assert len(self._action_queue) == self.config.n_action_steps, (
                f"Action queue must have {self.config.n_action_steps} actions"
            )

        action = self._action_queue.popleft()
        return action

    def _embed_text(self, text: str, bsize: int, device: torch.device) -> tuple[Tensor, int]:
        """Tokenize a fixed string and replicate across the batch.

        Returns ``(token_ids, length)`` where ``token_ids`` has shape
        ``(bsize, length)``. Used by :meth:`_build_prefix_items` to insert
        layout strings (``"State: "``, ``"Subgoal: "``, separators, etc.)
        without each call site having to think about device/batching.
        """
        ids = self.language_tokenizer.encode(text, add_special_tokens=False)
        tokens = torch.tensor([ids] * bsize, device=device)
        return tokens, len(ids)

    def _build_prefix_items(
        self,
        batch: dict[str, Tensor],
        *,
        include_discrete_actions: bool,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
    ) -> list[ContextItem]:
        """Construct the ordered ``ContextItem`` list defining the prefix.

        This is the **single place** to edit when adding a new conditioning
        block, removing one, or rearranging the order. The downstream
        ``PI07PaligemmaLowLevelFlowMatching`` is layout-agnostic.

        Layout (top-to-bottom = sequence order):

            videos
            language ("Task: …")
            "State: "
            state (per-timestep stream)
            ", " — gated by sample_has_response
            response ("Subtask: …") — gated per-token by response_masks
            ", " — gated by sample_has_metadata
            metadata ("Metadata: Speed: …") — gated per-token by metadata_masks
            ", " — gated by sample_has_subgoal
            "Subgoal: "
            subgoal images (one per camera, per-sample masked)
            ":\n"
            (training only)
            "Action: " — excluded from cross-attention
            discrete_actions (FAST tokens) — excluded from cross-attention
        """
        obs_history_is_pad = batch.get("obs_history_is_pad")

        videos, vid_masks = self.prepare_videos(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        metadata_tokens, metadata_masks = self.prepare_metadata(batch)
        subgoal_images, subgoal_img_masks = self.prepare_subgoal_images(batch)
        state = self.prepare_state(batch)

        bsize = state.shape[0]
        device = state.device
        items: list[ContextItem] = []

        for vid, vid_mask in zip(videos, vid_masks, strict=True):
            items.append(
                ContextItem(
                    data=vid,
                    item_type="video",
                    pad_mask=vid_mask,
                    attention="continue",
                    obs_history_is_pad=obs_history_is_pad,
                )
            )

        items.append(
            ContextItem(data=lang_tokens, item_type="text", pad_mask=lang_masks, attention="continue")
        )

        state_indicator_tokens, _ = self._embed_text("State: ", bsize, device)
        state_indicator_mask = torch.ones(
            bsize, state_indicator_tokens.shape[1], dtype=torch.bool, device=device
        )
        items.append(
            ContextItem(
                data=state_indicator_tokens,
                item_type="text",
                pad_mask=state_indicator_mask,
                attention="continue",
            )
        )

        # State pad mask: real for every non-padded history step; the current
        # (last) step is always real even if obs_history_is_pad isn't given.
        # Both branches return fresh tensors, so no clone is needed before
        # the indexed write.
        t_steps = state.shape[1]
        if obs_history_is_pad is not None:
            state_mask = ~obs_history_is_pad
        else:
            state_mask = torch.zeros(bsize, t_steps, dtype=torch.bool, device=device)
        state_mask[:, -1] = True
        # Defense-in-depth: zero the (already-normalized) state at masked steps so
        # no historical proprioception leaks even if the attention mask later
        # regresses. The current step is preserved by `state_mask[:, -1] = True`.
        # This runs AFTER normalize_inputs, so a masked slot becomes a clean
        # post-norm zero — never the ill `-mean/std` that zeroing a *raw* state
        # before normalization would produce.
        state = state.masked_fill(rearrange(~state_mask, "b t -> b t 1"), 0.0)
        items.append(ContextItem(data=state, item_type="state", pad_mask=state_mask, attention="continue"))

        # Response block: gated by sample_has_response.
        sample_has_response = response_masks.any(dim=1)
        resp_comma_tokens, resp_comma_len = self._embed_text(", ", bsize, device)
        items.append(
            ContextItem(
                data=resp_comma_tokens,
                item_type="text",
                pad_mask=sample_has_response[:, None].expand(bsize, resp_comma_len),
                attention="continue",
            )
        )
        items.append(
            ContextItem(
                data=response_tokens,
                item_type="text",
                pad_mask=response_masks,
                attention="bidirectional",
            )
        )

        # Metadata block: gated by sample_has_metadata.
        sample_has_metadata = metadata_masks.any(dim=1)
        md_comma_tokens, md_comma_len = self._embed_text(", ", bsize, device)
        items.append(
            ContextItem(
                data=md_comma_tokens,
                item_type="text",
                pad_mask=sample_has_metadata[:, None].expand(bsize, md_comma_len),
                attention="continue",
            )
        )
        items.append(
            ContextItem(
                data=metadata_tokens,
                item_type="text",
                pad_mask=metadata_masks,
                attention="bidirectional",
            )
        )

        # Subgoal block at the prefix tail (after all text), per π0.7 Fig. 19:
        # image goals come after the text prompt. Gated by sample_has_subgoal
        # (OR across cameras).
        if subgoal_img_masks:
            sample_has_subgoal = torch.stack(subgoal_img_masks, dim=0).any(dim=0)
        else:
            sample_has_subgoal = torch.zeros(bsize, dtype=torch.bool, device=device)

        sg_comma_tokens, sg_comma_len = self._embed_text(", ", bsize, device)
        items.append(
            ContextItem(
                data=sg_comma_tokens,
                item_type="text",
                pad_mask=sample_has_subgoal[:, None].expand(bsize, sg_comma_len),
                attention="continue",
            )
        )
        sg_start_tokens, sg_start_len = self._embed_text("Subgoal: ", bsize, device)
        items.append(
            ContextItem(
                data=sg_start_tokens,
                item_type="text",
                pad_mask=sample_has_subgoal[:, None].expand(bsize, sg_start_len),
                attention="bidirectional",
            )
        )
        for sg_img, sg_img_mask in zip(subgoal_images, subgoal_img_masks, strict=True):
            items.append(
                ContextItem(
                    data=sg_img,
                    item_type="image",
                    pad_mask=sg_img_mask,
                    attention="continue",
                )
            )

        # Always-real state-end terminator.
        state_end_tokens, _ = self._embed_text(":\n", bsize, device)
        state_end_mask = torch.ones(bsize, state_end_tokens.shape[1], dtype=torch.bool, device=device)
        items.append(
            ContextItem(
                data=state_end_tokens,
                item_type="text",
                pad_mask=state_end_mask,
                attention="continue",
            )
        )

        if include_discrete_actions:
            assert discrete_actions is not None and discrete_action_masks is not None, (
                "discrete_actions / discrete_action_masks required when include_discrete_actions=True"
            )
            da_indicator_tokens, da_indicator_len = self._embed_text("Action: ", bsize, device)
            da_indicator_mask = torch.ones(bsize, da_indicator_len, dtype=torch.bool, device=device)
            items.append(
                ContextItem(
                    data=da_indicator_tokens,
                    item_type="text",
                    pad_mask=da_indicator_mask,
                    attention="causal",
                    exclude_from_cross_attention=True,
                )
            )
            items.append(
                ContextItem(
                    data=discrete_actions,
                    item_type="discrete_action",
                    pad_mask=discrete_action_masks,
                    attention="causal",
                    exclude_from_cross_attention=True,
                )
            )

        return items

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        action_prefix: Tensor | None = None,
        delay: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample actions from the policy given environment observations.

        Note: The provided action_prefix should NOT be normalized, as this method will handle normalization internally.
        The action_prefix should have shape (batch_size, action_chunk_length, action_dim) where action_chunk_length is less than or equal to config.chunk_size.

        Args:
            batch: Batch of data containing environment observations.
                ``response`` may be omitted (defaults to ``\"\"`` per sample).
                Episode metadata is normalized before tokenization (same rules as
                training — see :meth:`_hydrate_metadata_batch`):

                - **None of** ``speed`` / ``quality`` / ``mistake`` /
                  ``*_is_pad`` **present** (common at inference): behaves like an
                  all-padded metadata batch (no conditioning).
                - **Mixed keys**: any subset may be present; missing fields use
                  those same defaults so tokenization matches a fully-specified
                  batch.
                ``speed`` / ``quality`` are floats; ``mistake`` is ``torch.bool``.
                ``subgoal{k}`` / ``subgoal_is_pad`` may be omitted; omitted
                subgoals are treated like an all-padded dataloader batch (no
                subgoal conditioning), consistent with training when every
                sample has ``subgoal_is_pad=True``.
            action_prefix: Optional action prefix tensor of shape (batch_size, action_chunk_length, action_dim).
            delay: Optional number of frozen delay actions from action_prefix.
            noise: Optional noise tensor.
        Returns:
            The sampled actions tensor of shape (batch_size, action_chunk_length, action_dim).
        """
        if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
            assert delay is None or 0 <= delay.item() <= self.config.max_delay, (
                f"Delay must be None or between 0 and {self.config.max_delay}"
            )

        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)

        self._hydrate_optional_conditioning_batch(batch)

        prefix_items = self._build_prefix_items(batch, include_discrete_actions=False)
        bsize = batch["state"].shape[0]
        device = batch["state"].device

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=device)

        if action_prefix is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            action_prefix = torch.zeros(actions_shape, dtype=torch.float32, device=device)
        else:
            action_prefix = self.normalize_targets({"actions": action_prefix}, dataset_index)["actions"]
            action_prefix = F.pad(
                action_prefix,
                (0, 0, 0, self.config.chunk_size - action_prefix.shape[1]),
            )

        # `prepare_state` already ran inside `_build_prefix_items`; replicate
        # its T inference here without a second call (a 2-D state is
        # unsqueezed to T=1, otherwise T = state.shape[-2]).
        raw_state = batch["state"]
        t_dim = 1 if raw_state.ndim == 2 else raw_state.shape[-2]
        if self.config.n_obs_steps > 1 and t_dim == 1:
            logging.warning(
                "n_obs_steps=%d but state has T=%d (single timestep). "
                "History buffering may not have been called.",
                self.config.n_obs_steps,
                t_dim,
            )

        actions = self.model.sample_actions(
            prefix_items,
            action_prefix=action_prefix,
            delay=delay,
            noise=noise,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"actions": actions}, dataset_index)["actions"]

        return actions

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> dict[str, Tensor | list]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Batch of data containing environment observations, actions, and targets.
            noise: Optional noise tensor.
            time: Optional time tensor.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE"). When the
            ``warn_outlier_threshold`` check finds offending dims, a non-empty
            ``"outlier_records"`` list is also included for the training loop to log.
        """
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        batch["discrete_actions"] = self.normalize_discrete_actions(dict(batch), dataset_index)["actions"]
        batch = self.normalize_targets(batch, dataset_index)

        # Debug aid: detection is pure / collective-free (CLAUDE.md rule 5); the records are
        # gathered to rank 0 and logged by the training loop so the warning always reaches wandb
        # regardless of which rank held the offending sample.
        outlier_records = detect_state_action_outliers(batch, self.config.warn_outlier_threshold)

        self._hydrate_optional_conditioning_batch(batch)

        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(batch)
        actions = batch["actions"]
        actions_is_pad = batch.get("action_is_pad")

        prefix_items = self._build_prefix_items(
            batch,
            include_discrete_actions=True,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
        )
        losses = self.model.forward(
            prefix_items,
            actions=actions,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
            actions_is_pad=actions_is_pad,
            noise=noise,
            time=time,
            real_action_dim=batch.get("real_action_dim"),
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]

        out: dict[str, Tensor | list] = {"MSE": mse_loss, "CE": ce_loss}
        if outlier_records:
            out["outlier_records"] = outlier_records
        return out

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepares the continuous state tensor, padding or truncating to max_state_dim.

        Accepts either ``(B, D)`` (single timestep) or ``(B, T, D)``
        (multi-timestep history).  Single-timestep tensors are unsqueezed
        to ``(B, 1, D)`` so downstream code always receives 3-D state.

        Args:
            batch: Batch of data containing the ``"state"`` tensor.

        Returns:
            A tensor of shape ``(B, T, max_state_dim)``.

        Raises:
            ValueError: If the state dimension exceeds ``max_state_dim``
                or if ``n_obs_steps > 1`` but a 2-D state is provided.
        """
        state = batch["state"]

        if state.ndim == 2:
            if self.config.n_obs_steps > 1:
                raise ValueError(
                    f"n_obs_steps={self.config.n_obs_steps} requires a "
                    f"(B, T, D) state tensor, but got shape {tuple(state.shape)}. "
                    f"Ensure _build_state_history_batch was called before prepare_state."
                )
            state = state.unsqueeze(1)  # (B, D) -> (B, 1, D)

        state_dim = state.shape[-1]
        if state_dim > self.config.max_state_dim:
            raise ValueError(
                f"State dimension ({state_dim}) exceeds max_state_dim ({self.config.max_state_dim}). "
                f"Increase max_state_dim in the config to accommodate the state vector."
            )
        if state_dim < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state_dim))
        return state

    def prepare_discrete_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Prepares discrete actions for the model by tokenizing and padding them.

        Args:
            batch: Batch of data containing the key "discrete_actions".

        Returns:
            A tuple containing:
                - discrete_action_tokens: A tensor of shape (batch_size, max_length) containing the tokenized actions.
                - discrete_action_masks: A tensor of shape (batch_size, max_length) indicating valid tokens.
        """
        device = batch["discrete_actions"].device
        discrete_actions = batch["discrete_actions"].to(device="cpu", dtype=torch.float32)
        tokens = self.discrete_action_processor.__call__(discrete_actions)
        discrete_action_tokens, discrete_action_masks = pad_discrete_tokens(
            tokens, self.config.discrete_action_max_length
        )
        return torch.from_numpy(discrete_action_tokens).to(device=device, dtype=torch.long), torch.from_numpy(
            discrete_action_masks
        ).to(device=device, dtype=torch.bool)

    def prepare_videos(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess camera inputs into (B, T, C, H, W) video tensors.

        Pixel values stay in [0, 1]; the SpaceTime SigLIP encoder rescales
        to [-1, 1] internally.  Padded history frames are zeroed at the pixel
        level here (read from ``batch["obs_history_is_pad"]``) AND masked inside
        the video encoder via temporal attention, mirroring pi07.

        Args:
            batch: Batch of data containing image/video tensors. May contain
                ``obs_history_is_pad`` ``(B, T)`` marking padded history frames.

        Returns:
            A tuple (videos, vid_masks) where each element is a list
            with one entry per camera.
        """
        obs_history_is_pad = batch.get("obs_history_is_pad")

        videos: list[Tensor] = []
        vid_masks: list[Tensor] = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        last_vid: Tensor | None = None
        last_mask: Tensor | None = None

        for key in present_img_keys:
            vid = batch[key]  # (B, T, C, H, W) or (B, C, H, W)
            if vid.ndim == 4:
                if self.config.n_obs_steps > 1:
                    raise ValueError(
                        f"Expected 5D video tensor (B, T, C, H, W) when n_obs_steps > 1, "
                        f"got shape {vid.shape}. Ensure select_action() is being used."
                    )
                vid = vid.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

            if obs_history_is_pad is not None:
                # Zero padded history frames at the pixel level (defense in
                # depth alongside the encoder's temporal-attention mask) so the
                # encoder never processes clamped/repeated content.
                vid = vid * (~obs_history_is_pad)[:, :, None, None, None]

            if self.config.resize_imgs_with_padding is not None:
                b, t_frames = vid.shape[:2]
                flat = rearrange(vid, "B T C H W -> (B T) C H W")
                flat = resize_with_pad(flat, *self.config.resize_imgs_with_padding, pad_value=0)
                vid = rearrange(flat, "(B T) C H W -> B T C H W", B=b, T=t_frames)

            bsize = vid.shape[0]
            device = vid.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            videos.append(vid)
            vid_masks.append(mask)
            last_vid = vid
            last_mask = mask

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            assert last_vid is not None
            vid = torch.zeros_like(last_vid)
            mask = torch.zeros_like(last_mask)
            videos.append(vid)
            vid_masks.append(mask)

        return videos, vid_masks

    def prepare_subgoal_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess subgoal images for the VLM image tower.

        Derives subgoal keys from ``config.image_features``: for each
        ``camera{k}`` the corresponding batch key is ``subgoal{k}``.
        If no ``subgoal{k}`` keys are present in the batch, zero-filled
        ``(B, C, H, W)`` tensors with masks all ``False`` are returned
        so that the prefix sequence length stays fixed.

        Each image is resized with aspect-ratio padding and rescaled from
        ``[0, 1]`` to ``[-1, 1]`` (the range SigLIP expects), then embedded
        via :meth:`PaliGemmaWithExpertModel.embed_image` as a 4-D
        ``(B, C, H, W)`` tensor (byte-identical to ``embed_video`` at T=1 on
        the shared vision tower).

        When ``batch["subgoal_is_pad"]`` is ``True`` for a sample, the
        subgoal slots for that sample are zeroed out and masks cleared.

        If no ``subgoal{k}`` keys are present (common during ``sample_actions``
        when the environment does not provide subgoals), this returns the same
        tensor shapes as a padded training batch but with all masks ``False``,
        which matches an all-padded batch from the dataloader in ``embed_prefix``
        (no subgoal conditioning for any sample).

        ``subgoal_is_pad`` may be omitted (defaults to all-``True``), scalar, or
        shape ``(batch,)``.

        Args:
            batch: Batch dict containing subgoal image tensors keyed as
                ``subgoal{k}`` for each ``camera{k}`` in
                ``config.image_features``.

        Returns:
            A tuple ``(subgoal_images, subgoal_img_masks)`` of lists,
            where each image has shape ``(B, C, H, W)`` in ``[-1, 1]``.
        """
        subgoal_images: list[Tensor] = []
        subgoal_img_masks: list[Tensor] = []

        subgoal_keys = [key.replace("camera", "subgoal") for key in self.config.image_features]
        present_keys = [key for key in subgoal_keys if key in batch]
        missing_keys = [key for key in subgoal_keys if key not in batch]

        bsize = batch["state"].shape[0]
        device = batch["state"].device

        default_pad = torch.ones(bsize, dtype=torch.bool, device=device)
        subgoal_is_pad = batch.get("subgoal_is_pad", default_pad)
        subgoal_is_pad = torch.as_tensor(subgoal_is_pad, dtype=torch.bool, device=device).reshape(-1)
        if subgoal_is_pad.numel() == 1:
            subgoal_is_pad = subgoal_is_pad.expand(bsize)
        elif subgoal_is_pad.shape[0] != bsize:
            raise ValueError(
                f"subgoal_is_pad must have shape ({bsize},) or be scalar broadcastable; "
                f"got shape {tuple(subgoal_is_pad.shape)}."
            )

        if len(present_keys) == 0:
            # No subgoal tensors in batch (typical for Libero eval): fabricate one
            # zero slot per entry in ``image_features``, same cardinality as when
            # the dataloader emits ``subgoal{k}`` tensors. The cleared masks make
            # the fill value inert (these tokens are masked out of attention), and
            # yield the same prefix as training with ``subgoal_is_pad=True``
            # everywhere (comma + ``Subgoal:`` + image tokens fully masked out).
            h, w = self.config.resize_imgs_with_padding or (224, 224)
            for _ in subgoal_keys:
                subgoal_images.append(torch.zeros(bsize, 3, h, w, device=device))
                subgoal_img_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))
            return subgoal_images, subgoal_img_masks

        last_img: Tensor | None = None
        last_mask: Tensor | None = None

        for key in present_keys:
            subgoal_img = batch[key]  # (B, C, H, W)

            if self.config.resize_imgs_with_padding is not None:
                subgoal_img = resize_with_pad(subgoal_img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from [0, 1] to [-1, 1] as expected by SigLIP.
            subgoal_img = subgoal_img * 2.0 - 1.0  # (B, C, H, W)

            img_device = subgoal_img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=img_device)

            is_pad = subgoal_is_pad.to(device=img_device, dtype=torch.bool)
            mask = mask & ~is_pad
            subgoal_img = subgoal_img * (~is_pad)[:, None, None, None]

            subgoal_images.append(subgoal_img)
            subgoal_img_masks.append(mask)
            last_img = subgoal_img
            last_mask = mask

        # Pad to len(subgoal_keys) regardless of empty_cameras so the prefix
        # length matches the all-missing-keys path above. With mixed
        # present/missing keys and empty_cameras=0, the prior loop produced
        # fewer slots than subgoal_keys and made the prefix length depend on
        # which subgoal{k} keys happened to be in the batch.
        if last_img is not None and last_mask is not None:
            for _ in missing_keys:
                # Fill value is inert: the cleared mask excludes these slots from
                # attention, so the placeholder pixels never reach the model.
                subgoal_images.append(torch.zeros_like(last_img))
                subgoal_img_masks.append(torch.zeros_like(last_mask))

        return subgoal_images, subgoal_img_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the text input.

        Args:
            batch: Batch of data containing the key "prompt" and "state".

        Returns:
            A tuple containing:
                - lang_tokens: Tensor of language tokens.
                - lang_masks: Tensor of language attention masks.
        """
        device = batch["state"].device
        tasks = batch["prompt"]

        prompt = [f"Task: {task}, " for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.prompt_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _hydrate_metadata_batch(self, batch: dict) -> None:
        """Ensure episode metadata keys exist on ``batch`` (idempotent, in-place).

        Single source of truth for defaults shared by :meth:`prepare_metadata`,
        :meth:`forward`, and :meth:`sample_actions`:

        - **All metadata keys missing** — e.g. inference observation dicts with
          no ``speed`` / ``quality`` / ``mistake`` fields: ``speed`` / ``quality``
          default to float zeros; ``mistake`` defaults to ``False``; every
          ``*_is_pad`` defaults to ``True`` (no metadata conditioning),
          matching an all-padded training batch.
        - **Mixed keys** — only some keys present: fill each missing key with
          that same default so behavior matches a batch where every key was
          specified explicitly.
        - ``robot_type`` / ``control_mode`` are string lists (empty string is
          the pad signal, no separate ``_is_pad`` flag) and default to ``[""]``.
        - ``fps`` (the dataset's native frame rate, ``torch.long``) defaults
          to zeros and ``fps_is_pad`` defaults to ``True`` when the keys
          are absent. The dataloader emits ``fps`` unconditionally (no
          dropout) when ``DatasetMixtureConfig.emit_fps=True``; inference
          batches built by ``add_eval_metadata`` set it when
          ``EnvMetadataConfig.emit_fps=True``.

        Per-sample on/off is controlled only by ``*_is_pad`` (or the empty
        string for ``robot_type`` / ``control_mode``) after hydration; varying
        fields across rows uses the batch tensors, not missing dict keys.
        """
        bsz = batch["state"].shape[0]
        dev = batch["state"].device
        if "speed" not in batch:
            batch["speed"] = torch.zeros(bsz, dtype=torch.float32, device=dev)
        if "quality" not in batch:
            batch["quality"] = torch.zeros(bsz, dtype=torch.float32, device=dev)
        if "mistake" not in batch:
            batch["mistake"] = torch.zeros(bsz, dtype=torch.bool, device=dev)
        if "speed_is_pad" not in batch:
            batch["speed_is_pad"] = torch.ones(bsz, dtype=torch.bool, device=dev)
        if "quality_is_pad" not in batch:
            batch["quality_is_pad"] = torch.ones(bsz, dtype=torch.bool, device=dev)
        if "mistake_is_pad" not in batch:
            batch["mistake_is_pad"] = torch.ones(bsz, dtype=torch.bool, device=dev)
        if "robot_type" not in batch:
            batch["robot_type"] = [""] * bsz
        if "control_mode" not in batch:
            batch["control_mode"] = [""] * bsz
        if "fps" not in batch:
            batch["fps"] = torch.zeros(bsz, dtype=torch.long, device=dev)
        if "fps_is_pad" not in batch:
            batch["fps_is_pad"] = torch.ones(bsz, dtype=torch.bool, device=dev)

    def _hydrate_optional_conditioning_batch(self, batch: dict) -> None:
        """Ensure response / metadata keys exist (training + inference parity).

        Matches :meth:`prepare_response` when ``response`` is absent (empty
        strings per sample). Delegates metadata defaults to
        :meth:`_hydrate_metadata_batch`.
        """
        bsz = batch["state"].shape[0]
        if "response" not in batch:
            batch["response"] = [""] * bsz
        self._hydrate_metadata_batch(batch)

    def prepare_response(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the high-level planner subtask response.

        Wraps each response string as ``"Subtask: {response}, "`` and
        pads/truncates to ``response_max_length``.  Uses ``add_special_tokens=False``
        so no BOS token is inserted.

        Args:
            batch: Batch dict containing ``"response"`` (list of strings).

        Returns:
            ``(response_tokens, response_masks)`` each of shape
            ``(B, response_max_length)``.
        """
        device = batch["state"].device
        responses = batch["response"] if "response" in batch else [""] * batch["state"].shape[0]
        response_prompt = [f"Subtask: {response}" if response != "" else "" for response in responses]
        tokenized_response = self.language_tokenizer.__call__(
            response_prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.response_max_length,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        )
        response_tokens = tokenized_response["input_ids"].to(device=device)
        response_masks = tokenized_response["attention_mask"].to(device=device, dtype=torch.bool)
        return response_tokens, response_masks

    def prepare_metadata(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize episode metadata into PaliGemma token IDs.

        Builds strings ``Metadata: Speed: … Quality: … Mistake: … Robot: …
        FPS: … Control: …`` only from fields whose ``*_is_pad`` flag is
        ``False`` (``robot_type`` / ``control_mode`` are strings, omitted
        when empty). For each sample, a padded or empty field is omitted
        entirely (not concatenated to ``segments``). When every field is
        pad, the row is ``""`` before tokenization.

        ``fps`` is the effective per-sample frame rate (``torch.long``);
        the segment is omitted when ``fps_is_pad`` is True. Segment
        ordering is ``Speed → Quality → Mistake → Robot → FPS → Control``
        to match the other pi07 / pi07_paligemma ``prepare_metadata``
        implementations.

        Always runs :meth:`_hydrate_metadata_batch` first so callers see the same
        outcomes whether they hit :meth:`sample_actions` (often no metadata
        keys), :meth:`forward` with a partial-key batch, or a dataloader batch
        with every key set.

        Values are normalized to shape ``(B,)`` on ``state``'s device, with
        scalar tensors broadcast like ``subgoal_is_pad``.

        .. note::
            ``speed_is_pad`` / ``quality_is_pad`` / ``mistake_is_pad`` /
            ``fps_is_pad`` each default to ``torch.ones(B, dtype=bool)``
            (treat-as-pad) when the key is missing from ``batch``. This is
            a behavior change from the previous treat-as-real
            (``torch.zeros``) default. Hand-built inference batches that
            supply ``"speed"`` / ``"quality"`` / ``"mistake"`` / ``"fps"``
            but omit the corresponding ``_is_pad`` flag will now have
            those metadata fields silently dropped from the prefix string
            — pass ``..._is_pad=torch.zeros(...)`` explicitly when the
            metadata is real.

        Args:
            batch: Batch dict; metadata keys may be missing or partial before
                hydration (see :meth:`_hydrate_metadata_batch`).

        Returns:
            ``(metadata_tokens, metadata_masks)`` with shapes
            ``(B, metadata_max_length)``. Never ``None``.
        """
        self._hydrate_metadata_batch(batch)
        device = batch["state"].device
        b = batch["state"].shape[0]

        def _row_float(key: str) -> Tensor:
            t = batch[key]
            t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1)
            if t.numel() == 1:
                t = t.expand(b)
            elif t.shape[0] != b:
                raise ValueError(
                    f"{key} must have shape ({b},) or be scalar broadcastable; got {tuple(t.shape)}."
                )
            return t

        def _row_long(key: str) -> Tensor:
            """``(B,)`` long-int row (``fps``), scalar-broadcastable."""
            t = batch[key]
            t = torch.as_tensor(t, dtype=torch.long, device=device).reshape(-1)
            if t.numel() == 1:
                t = t.expand(b)
            elif t.shape[0] != b:
                raise ValueError(
                    f"{key} must have shape ({b},) or be scalar broadcastable; got {tuple(t.shape)}."
                )
            return t

        def _row_bool(key: str) -> Tensor:
            """``(B,)`` boolean row (``mistake`` or ``*_is_pad``), scalar-broadcastable."""
            t = batch[key]
            t = torch.as_tensor(t, dtype=torch.bool, device=device).reshape(-1)
            if t.numel() == 1:
                t = t.expand(b)
            elif t.shape[0] != b:
                raise ValueError(
                    f"{key} must have shape ({b},) or be scalar broadcastable; got {tuple(t.shape)}."
                )
            return t

        speed_t = _row_float("speed")
        quality_t = _row_float("quality")
        mistake_t = _row_bool("mistake")
        fps_t = _row_long("fps")
        pad_speed = _row_bool("speed_is_pad")
        pad_quality = _row_bool("quality_is_pad")
        pad_mistake = _row_bool("mistake_is_pad")
        pad_fps = _row_bool("fps_is_pad")
        robot_types = batch["robot_type"]
        control_modes = batch["control_mode"]

        metadata_rows: list[str] = []
        for (
            speed,
            quality,
            mistake,
            speed_is_pad,
            quality_is_pad,
            mistake_is_pad,
            robot_type,
            control_mode,
            fps,
            fps_is_pad,
        ) in zip(
            speed_t,
            quality_t,
            mistake_t,
            pad_speed,
            pad_quality,
            pad_mistake,
            robot_types,
            control_modes,
            fps_t,
            pad_fps,
            strict=True,
        ):
            segments: list[str] = []
            # *_is_pad True (or empty string) → omit that field from the metadata
            # string (same as the dataloader mask). Each segment carries its own
            # trailing ", " and segments are space-joined, matching pi07.
            if not speed_is_pad.item():
                segments.append(f"Speed: {str(speed.item())}, ")
            if not quality_is_pad.item():
                segments.append(f"Quality: {str(quality.item())}, ")
            if not mistake_is_pad.item():
                segments.append(f"Mistake: {str(mistake.item())}, ")
            if robot_type:
                segments.append(f"Robot: {robot_type}, ")
            if not fps_is_pad.item():
                segments.append(f"FPS: {str(fps.item())}, ")
            if control_mode:
                segments.append(f"Control: {control_mode}, ")
            metadata_rows.append(f"Metadata: {' '.join(segments)}" if segments else "")

        tokenized = self.language_tokenizer.__call__(
            metadata_rows,
            padding="max_length",
            padding_side="right",
            max_length=self.config.metadata_max_length,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        )
        metadata_tokens = tokenized["input_ids"].to(device=device)
        metadata_masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        return metadata_tokens, metadata_masks


class PI07PaligemmaLowLevelFlowMatching(nn.Module):
    """π07 Low-Level — A Vision-Language-Action Flow Matching model with
    SpaceTime SigLIP video, subtask/subgoal conditioning, and episode metadata.

    Architecturally inherits the π0.5 PaliGemma + Gemma-Expert flow-matching
    backbone ([π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf))
    and extends the prefix to ingest, on top of the original (image, language,
    state, discrete actions) tokens:

    * **Image history** — encoded by :class:`SpaceTimeSiglipVideoEncoder`.
      ``n_obs_steps`` frames per camera; T=1 is byte-identical to plain SigLIP,
      T>1 inserts space-time separable temporal attention every
      ``spacetime_layer_stride`` SigLIP layers (MEM-paper recipe).
    * **Per-timestep robot state stream** ``(B, T, max_state_dim)`` — one VLM
      token per history step, masked according to ``obs_history_is_pad``
      (current step is always real).
    * **Subtask response tokens** — the natural-language subtask emitted by
      :class:`PI07HighLevelPlannerModel`; fully masked out per-sample when the
      high-level planner is dropped, in which case only the leading attention
      "boundary" token is kept.
    * **Subgoal image(s)** — optional future-state visual goal frame(s), routed
      through the same SpaceTime SigLIP encoder; per-sample masked, and the
      subgoal SigLIP forward is skipped entirely when no sample in the batch has
      a subgoal (saves ~2× SigLIP compute).
    * **Episode metadata tokens** — optional ``"Speed: …, Quality: …, Mistake:
      …"`` string assembled from ``speed`` / ``quality`` / ``mistake`` features
      and tokenized to ``metadata_max_length``.

    The flow-matching action expert (Gemma expert with adaRMS time conditioning,
    ``num_steps``-step Euler integration, FAST-token discrete-action prefix that
    is excluded from continuous-action cross-attention via ``n_cross_att_tokens``)
    is unchanged from π0.5.

    ┌────────────────────────────────────────────────────────────┐
    │                         actions                            │
    │                            ▲                               │
    │                         ┌──┴───┐                           │
    │      kv cache           │Gemma │                           │
    │      ┌─────────────────►│Expert│ ◄── adaRMS(time)          │
    │      │                  │      │                           │
    │     ┌┴────────────┐     │ x 10 │                           │
    │     │             │     └──▲───┘                           │
    │     │  PaliGemma  │        │                               │
    │     │             │       noise                            │
    │     └▲─▲─▲─▲─▲─▲─▲                                         │
    │      │ │ │ │ │ │ └── discrete actions                      │
    │      │ │ │ │ │ └──── episode metadata                      │
    │      │ │ │ │ └────── subgoal image(s)                      │
    │      │ │ │ └──────── subtask response                      │
    │      │ │ └────────── robot state                           │
    │      │ └──────────── language tokens                       │
    │      └────────────── video (image history)                 │
    └────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: PI07PaligemmaLowLevelConfig, discrete_action_vocab_size: int | None = None):
        """Initializes the PI07PaligemmaLowLevelFlowMatching model.

        Args:
            config: Model configuration.
            discrete_action_vocab_size: Size of the discrete action vocabulary.
        """
        super().__init__()
        self.config = config

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            load_pretrained_paligemma=False,
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        self.video_encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=self.paligemma_with_expert.paligemma.vision_tower,
            multi_modal_projector=self.paligemma_with_expert.paligemma.multi_modal_projector,
            max_num_frames=self.config.n_obs_steps,
            spacetime_layer_stride=self.config.spacetime_layer_stride,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

        vlm_hidden_size = self.paligemma_with_expert.config.paligemma_config.text_config.hidden_size
        self.state_proj = nn.Linear(self.config.max_state_dim, vlm_hidden_size)

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        """Samples Gaussian noise.

        Args:
            shape: The shape of the noise tensor.
            device: The device to create the tensor on.

        Returns:
            A tensor containing the sampled noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        """Samples time steps from a Beta distribution.

        Args:
            bsize: Batch size.
            device: The device to create the tensor on.

        Returns:
            A tensor containing the sampled time steps.
        """
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_video(self, video: Tensor, obs_history_is_pad: Tensor | None = None) -> Tensor:
        """Encode a video through the SpaceTime SigLIP video encoder.

        At T=1 this is byte-identical to ``paligemma_with_expert.embed_image``.

        Args:
            video: (B, T, C, H, W) pixel values in [0, 1].
            obs_history_is_pad: Optional ``(B, T)`` bool mask — ``True`` for
                padded history frames.  Passed to the video encoder so that
                temporal attention blocks padded frames.

        Returns:
            (B, num_video_tokens, vlm_hidden_size)
        """
        return self.video_encoder(video, obs_history_is_pad=obs_history_is_pad)

    def _global_run_image_tower(self, items: list[ContextItem]) -> bool:
        """Decide once, across all ranks, whether the subgoal image tower runs.

        :meth:`embed_prefix` calls this exactly once per forward — before the
        per-item loop and on *every* rank, even one whose prefix happens to
        contain no ``image`` item — then threads the result into each
        :meth:`_embed_item` via ``run_image_tower``. Hoisting the collective
        out of the per-item branch (mirroring pi07's ``embed_prefix``) is what
        keeps DDP / FSDP collective counts aligned: a real
        ``subgoal_present_local`` flows through
        :func:`_global_or_branch_decisions`, so its presence-divergence check
        actually fires (raising loudly) if a future heterogeneous mixture ever
        gives ranks a different number of subgoal-image items — instead of the
        all-reduce count silently desyncing and hanging at NCCL (CLAUDE.md hard
        rule 5). Putting the all-reduce *inside* the ``image`` branch would
        reintroduce exactly that anti-pattern.

        The OR runs over all cameras (matching pi07's single ``has_subgoal``):
        if any subgoal on any rank is real, every camera's tower runs, and the
        per-camera pad masks still zero unused slots downstream. Bypassed
        (returns ``True``) under ``torch.compile`` / ONNX export, where the
        collective and Python branch can't be traced and the exported graph
        must always include the tower.

        Args:
            items: The ordered prefix ``ContextItem``s passed to
                :meth:`embed_prefix`.

        Returns:
            Whether this rank should run the image tower this step.
        """
        if torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export():
            return True
        image_items = [it for it in items if it.item_type == "image"]
        subgoal_present_local = bool(image_items)
        subgoal_any_local = any(bool(it.pad_mask.any()) for it in image_items)
        (run_image_tower,) = _global_or_branch_decisions(
            presence_locals=(subgoal_present_local,),
            any_locals=(subgoal_any_local,),
            field_names=("subgoal_image",),
            device=items[0].data.device,
        )
        return run_image_tower

    def _embed_item(self, item: ContextItem, *, run_image_tower: bool | None = None) -> tuple[Tensor, Tensor]:
        """Embed a single ``ContextItem`` and return ``(emb, expanded_pad_mask)``.

        Dispatches on ``item.item_type``:
        - ``text`` / ``discrete_action``: token-ID embedding via PaliGemma
          (text gets the ``sqrt(hidden)`` normalization).
        - ``state``: continuous-state projection.
        - ``video``: SpaceTime SigLIP. The unified video encoder accepts
          variable ``T`` (short-circuiting the temporal sublayer at
          ``T=1``), so single-frame subgoal videos are forwarded as-is —
          no zero-prepend to ``num_frames`` is required. When no sample
          needs this video (``pad_mask.any()`` is ``False``), the SigLIP
          backbone is skipped and zeros are emitted instead — a strict
          efficiency win since the per-token pad mask zeros the slot
          regardless. **DDP invariant:** the skip is data-dependent, so
          all ranks must agree on whether SigLIP runs. ``pad_mask`` for
          a ``video`` item must therefore be derivable from config
          (e.g. ``empty_cameras`` is config-determined; real cameras get
          all-True per-rank), never per-sample data that can diverge
          across ranks — otherwise DDP with
          ``find_unused_parameters=False`` will deadlock.
        - ``image``: subgoal image through PaliGemma's image tower. Same
          skip-and-emit-zeros efficiency win as ``video``, but the
          ``pad_mask`` here *is* per-sample data (``subgoal_is_pad`` can
          diverge across ranks). The cross-rank run/skip decision is computed
          once per forward by :meth:`_global_run_image_tower` and threaded in
          via ``run_image_tower`` — the collective is kept out of this
          per-item branch (see that method). When called directly with
          ``run_image_tower=None`` (single-process unit tests) it falls back
          to a rank-local decision that fires no collective.
        - ``action``: noisy-action projection (suffix only).

        The pad mask is broadcast/expanded so the returned mask matches
        the embedded sequence length: per-sample ``(B,)`` is allowed for
        ``video`` (expanded to ``(B, num_video_tokens)``) and ``image``
        (expanded to ``(B, num_image_tokens)``); all other types pass
        through ``(B, L)`` unchanged.
        """
        t = item.item_type
        if t == "text":
            emb = self.paligemma_with_expert.embed_language_tokens(item.data)
            emb = emb * math.sqrt(emb.shape[-1])
            return emb, item.pad_mask
        if t == "state":
            emb = self.state_proj(item.data.to(dtype=_preferred_dtype()))
            return emb, item.pad_mask
        if t == "discrete_action":
            emb = self.paligemma_with_expert.embed_discrete_actions(item.data)
            return emb.to(dtype=_preferred_dtype()), item.pad_mask
        if t == "action":
            emb = self.action_in_proj(item.data.to(dtype=_preferred_dtype()))
            return emb, item.pad_mask
        if t == "image":
            # Subgoal images: 4-D (B, C, H, W) in [-1, 1] through the VLM's
            # image tower (byte-identical to embed_video at T=1 on the shared
            # tower). When no sample needs this subgoal the tower is skipped and
            # zeros are emitted — a strict, mask-equivalent efficiency win (the
            # per-sample mask zeros the slot regardless), e.g. at inference or
            # whenever subgoal_is_pad is all-True.
            #
            # ``run_image_tower`` is the cross-rank decision computed once per
            # forward in ``_global_run_image_tower`` and passed down by
            # ``embed_prefix``; the collective lives there, not here, so it
            # can't hide behind this per-item branch (CLAUDE.md hard rule 5).
            # A direct call (single-process unit tests) leaves it None and we
            # fall back to a purely-local decision that fires no collective.
            image = item.data
            per_sample_mask = item.pad_mask
            if per_sample_mask.ndim != 1:
                raise ValueError(
                    f"image pad_mask must be (B,) per-sample; got shape {tuple(per_sample_mask.shape)}."
                )
            dtype = _preferred_dtype()
            if run_image_tower is None:
                run_image_tower = (
                    torch.compiler.is_compiling()
                    or torch.onnx.is_in_onnx_export()
                    or bool(per_sample_mask.any())
                )
            if run_image_tower:
                emb = self.paligemma_with_expert.embed_image(image).to(dtype=dtype)
            else:
                pg_text_cfg = self.paligemma_with_expert.config.paligemma_config.text_config
                emb = torch.zeros(
                    image.shape[0],
                    pg_text_cfg.num_image_tokens,
                    pg_text_cfg.hidden_size,
                    device=image.device,
                    dtype=dtype,
                )
            expanded = per_sample_mask[:, None].expand(emb.shape[0], emb.shape[1])
            return emb, expanded
        if t == "video":
            # The unified SigLIP video encoder accepts variable T
            # (short-circuiting the temporal sublayer at T=1).
            video = item.data
            obs_pad = item.obs_history_is_pad
            per_sample_mask = item.pad_mask
            if per_sample_mask.ndim != 1:
                raise ValueError(
                    f"video pad_mask must be (B,) per-sample; got shape {tuple(per_sample_mask.shape)}."
                )
            n_tokens = self.video_encoder.num_video_tokens
            vlm_h = self.paligemma_with_expert.config.paligemma_config.text_config.hidden_size
            dtype = _preferred_dtype()
            if bool(per_sample_mask.any()):
                emb = self.embed_video(video, obs_history_is_pad=obs_pad).to(dtype=dtype)
            else:
                emb = torch.zeros(video.shape[0], n_tokens, vlm_h, device=video.device, dtype=dtype)
            expanded = per_sample_mask[:, None].expand(emb.shape[0], emb.shape[1])
            return emb, expanded
        raise ValueError(f"Unknown ContextItem type: {t!r}")

    @staticmethod
    def _attention_pattern(mode: AttentionMode, length: int) -> list[int]:
        """1-D attention pattern for a block of ``length`` tokens.

        See ``ContextItem.attention``. The cumulative ``sum`` of these
        ints, fed to :func:`make_att_2d_masks`, defines a token's
        attention scope: tokens with the same cumulative value share
        bidirectional attention.
        """
        if mode == "continue":
            return [0] * length
        if mode == "bidirectional":
            return [1] + [0] * (length - 1)
        if mode == "causal":
            return [1] * length
        raise ValueError(f"Unknown attention mode: {mode!r}")

    def embed_prefix(self, items: list[ContextItem]) -> tuple[Tensor, Tensor, Tensor, int]:
        """Embed a layout-agnostic list of ``ContextItem``s into the prefix.

        The policy decides what blocks go into the prefix and in what
        order; this method only embeds them by type, concatenates, and
        derives the 1-D attention mask. Items flagged
        ``exclude_from_cross_attention`` are dropped from the
        ``num_cross_att_tokens`` count returned alongside the embeddings,
        so the action expert never cross-attends to them (e.g. trailing
        ``"Action: "`` indicator + discrete-action tokens).

        Args:
            items: Ordered list of ``ContextItem`` blocks. Per the
                current architecture, items flagged
                ``exclude_from_cross_attention`` must be a contiguous
                trailing run.

        Returns:
            ``(embs, pad_masks, att_masks, num_cross_att_tokens)``.
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks_flat: list[int] = []

        # Cross-rank subgoal-image run/skip decision, computed once here (every
        # rank, before the loop) and threaded into each item — see
        # _global_run_image_tower for why the collective must not live inside
        # the per-item ``image`` branch.
        run_image_tower = self._global_run_image_tower(items)

        cross_att_running = 0
        cross_att_locked = False
        for item in items:
            emb, mask = self._embed_item(item, run_image_tower=run_image_tower)
            length = emb.shape[1]
            embs.append(emb)
            pad_masks.append(mask)
            att_masks_flat += self._attention_pattern(item.attention, length)

            if item.exclude_from_cross_attention:
                cross_att_locked = True
            else:
                if cross_att_locked:
                    raise ValueError("exclude_from_cross_attention items must be a contiguous trailing run.")
                cross_att_running += length

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        bsize = embs_cat.shape[0]
        att_masks = torch.tensor(att_masks_flat, dtype=torch.bool, device=pad_masks_cat.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks_flat))
        return embs_cat, pad_masks_cat, att_masks, cross_att_running

    def embed_suffix(
        self, items: list[ContextItem], timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the action expert's suffix from a list of ``ContextItem``s.

        Mirrors :meth:`embed_prefix`. The timestep stays a separate
        argument (not a ``ContextItem``) because it conditions the
        action expert via adaRMS rather than appearing as a sequence
        token.

        Args:
            items: Ordered ``ContextItem`` blocks (today: just a single
                ``"action"`` item containing the noisy actions; the policy
                may add more in the future).
            timestep: ``(B, chunk_size)`` flow-matching time, fed through
                a sine-cosine positional encoding and a 2-layer MLP to
                produce the adaRMS conditioning vector.

        Returns:
            ``(embs, pad_masks, att_masks, adarms_cond)``.
        """
        device = timestep.device
        dtype = _preferred_dtype()

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=dtype)

        def time_mlp_func(x: Tensor) -> Tensor:
            x = self.time_mlp_in(x)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = time_mlp_func(time_emb)

        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks_flat: list[int] = []

        for item in items:
            emb, mask = self._embed_item(item)
            length = emb.shape[1]
            embs.append(emb)
            pad_masks.append(mask)
            att_masks_flat += self._attention_pattern(item.attention, length)

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        bsize = embs_cat.shape[0]
        att_masks = torch.tensor(att_masks_flat, dtype=torch.bool, device=embs_cat.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks_flat))

        return embs_cat, pad_masks_cat, att_masks, adarms_cond

    def forward(
        self,
        prefix_items: list[ContextItem],
        actions: Tensor,
        discrete_actions: Tensor,
        discrete_action_masks: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        real_action_dim: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute the loss.

        Args:
            prefix_items: Ordered ``ContextItem`` list defining the entire
                VLM prefix layout. The ``"Action: "`` indicator and the
                discrete-action item must be flagged
                ``exclude_from_cross_attention=True`` so the continuous
                action expert does not condition on them.
            actions: Continuous action targets ``(B, chunk_size, max_action_dim)``.
            discrete_actions: FAST discrete-action token IDs
                ``(B, discrete_action_max_length)``, used for the CE loss.
            discrete_action_masks: Mask for ``discrete_actions``.
            actions_is_pad: Optional ``(B, chunk_size)`` mask for padded actions.
            noise: Optional pre-sampled noise tensor.
            time: Optional pre-sampled flow-matching time tensor.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE").
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_cross_att_tokens = self.embed_prefix(
            prefix_items
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        # Now run action expert
        batch_size = actions.shape[0]
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(batch_size, actions.device)

        # handle real time inference delay
        delay = torch.randint(0, self.config.max_delay + 1, (batch_size,))
        prefix_mask = rearrange(torch.arange(self.config.chunk_size), "c -> 1 c") < rearrange(
            delay, "b -> b 1"
        )
        prefix_mask = prefix_mask.to(device=actions.device)
        time = torch.where(
            prefix_mask, 0, rearrange(time, "b -> b 1")
        )  # using diffusion time 0 instead of flow matching time 1

        time_expanded = rearrange(time, "b c -> b c 1")
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_items = self._build_suffix_items(x_t)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(suffix_items, time)

        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        # Action expert position ids continue past the cross-attended prefix
        # (skipping the trailing discrete-action block, which is excluded
        # from cross-attention via num_cross_att_tokens).
        prefix_offsets = torch.sum(prefix_pad_masks[:, :num_cross_att_tokens], dim=-1)[:, None]
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # stop gradient to avoid backpropagating from action expert to VLM
        for layer_idx in past_key_values:
            past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].detach()
            past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx]["value_states"].detach()

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # compute mse loss for velocity
        # Supervise the whole chunk the model was trained to predict. n_action_steps
        # is the inference-time execution horizon only and must not truncate the
        # training target (chunk_size is the prediction horizon).
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        # Shared masked-MSE reduction; see pi05 for the rationale.
        mse_loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            max_action_dim=self.config.max_action_dim,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            real_action_dim=real_action_dim,
        )

        # compute cross entropy loss for discrete actions
        batch_size, seq_len = discrete_actions.shape
        discrete_token_start = -self.config.discrete_action_max_length
        # The token before discrete actions predicts the first discrete action token, so slice from discrete_token_start - 1.
        # The predicted last discrete action token is unused, so exclude it from loss.
        discrete_action_slice_object = slice(discrete_token_start - 1, -1)
        discrete_action_out = prefix_out[:, discrete_action_slice_object]
        logits = self.paligemma_with_expert.da_head(discrete_action_out)

        logits = logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        logits = rearrange(logits, "b s d -> (b s) d")
        labels = rearrange(discrete_actions, "b s -> (b s)")
        discrete_action_ce_loss = F.cross_entropy(logits, labels, reduction="none")

        discrete_action_ce_loss = rearrange(discrete_action_ce_loss, "(b s) -> b s", b=batch_size, s=seq_len)

        # remove pad tokens
        discrete_action_is_pad = ~discrete_action_masks  # convert into format where value for pad is True
        discrete_action_ce_loss = discrete_action_ce_loss * ~discrete_action_is_pad

        # compute mean
        discrete_action_ce_loss = discrete_action_ce_loss.mean()

        return {"MSE": mse_loss, "CE": discrete_action_ce_loss}

    def _build_suffix_items(self, x_t: Tensor) -> list[ContextItem]:
        """Default suffix layout: a single ``"action"`` block for ``x_t``.

        Factored out so that :meth:`forward`, :meth:`sample_actions`, and
        :meth:`denoise_step` share one definition of what's in the suffix.

        Invariant: the action block must stay the *trailing* ``chunk_size``
        tokens of the suffix. :meth:`forward` and :meth:`denoise_step` recover
        the action outputs with ``suffix_out[:, -chunk_size:]``; if a future
        block is appended after the actions, that slice (and those call sites)
        must be updated accordingly.
        """
        bsize = x_t.shape[0]
        action_mask = torch.ones(bsize, x_t.shape[1], dtype=torch.bool, device=x_t.device)
        return [
            ContextItem(
                data=x_t,
                item_type="action",
                pad_mask=action_mask,
                attention="bidirectional",
            )
        ]

    def sample_actions(
        self,
        prefix_items: list[ContextItem],
        action_prefix: Tensor,
        delay: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action.

        Args:
            prefix_items: Ordered ``ContextItem`` list defining the entire
                VLM prefix layout. Inference omits the discrete-action
                block, so no items need to be excluded from cross-attention.
            action_prefix: Frozen action prefix ``(B, chunk_size, max_action_dim)``.
            delay: Number of frozen delay actions at the start of the chunk.
            noise: Optional pre-sampled noise.

        Returns:
            The sampled action tensor.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_cross_att_tokens = self.embed_prefix(
            prefix_items
        )
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        (_prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        # perform denoising steps to get the action
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = rearrange(torch.arange(self.config.chunk_size, device=device), "c -> 1 c") < delay
        while time >= -dt / 2:
            # if delay is greater than 0, then freeze the action prefix at the beginning of action chunk
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
            masked_time = torch.where(prefix_mask, 0, time)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                masked_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        # we need to ensure the frozen actions are not modified before returning the denoised actions
        x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Apply one denoising step of the noise `x_t` at a given timestep.

        Args:
            prefix_pad_masks: Prefix padding masks.
            past_key_values: Past key values from the VLM.
            x_t: Current noise tensor.
            time: Time tensor of shape (batch_size, action_chunk_length).
        Returns:
            The predicted velocity tensor (v_t).
        """
        suffix_items = self._build_suffix_items(x_t)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(suffix_items, time)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[
            :, None
        ]  # action expert position ids start after prefix
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        # Denoise the full chunk_size chunk so v_t matches x_t in the Euler step.
        # n_action_steps (execution horizon) is applied later in select_action, not
        # at decode time.
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
