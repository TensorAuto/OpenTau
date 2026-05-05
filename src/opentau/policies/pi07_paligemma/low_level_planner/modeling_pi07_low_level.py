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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import NormalizationMode
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.pi07_paligemma.low_level_planner.configuration_pi07_low_level import (
    PI07lowlevelPlannerConfig,
)
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.utils import get_safe_dtype


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


class PI07LowLevelPlannerPolicy(PreTrainedPolicy):
    """Wrapper class around PI07LowLevelPlannerFlowMatching model to train and run inference within OpenTau."""

    config_class = PI07lowlevelPlannerConfig
    name = "pi07_paligemma_low_level_planner"

    def __init__(
        self,
        config: PI07lowlevelPlannerConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the PI07LowLevelPlannerPolicy.

        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_discrete_actions = Normalize(
            config.output_features, {"ACTION": NormalizationMode.MIN_MAX}, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        # Get vocab size from processor
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI07LowLevelPlannerFlowMatching(
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

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            if missing_keys and is_main_process:
                logging.warning("Missing keys when loading state dict: %d keys", len(missing_keys))
                for key in missing_keys[:20]:
                    logging.warning("  - %s", key)
                if len(missing_keys) > 20:
                    logging.warning("  ... and %d more", len(missing_keys) - 20)

            if unexpected_keys and is_main_process:
                logging.warning("Unexpected keys when loading state dict: %d keys", len(unexpected_keys))
                for key in unexpected_keys[:20]:
                    logging.warning("  - %s", key)
                if len(unexpected_keys) > 20:
                    logging.warning("  ... and %d more", len(unexpected_keys) - 20)

            if not missing_keys and not unexpected_keys and is_main_process:
                logging.info("All keys loaded successfully!")

        except Exception as e:
            if is_main_process:
                logging.warning("Could not remap state dict keys: %s", e)

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
        assembles a batch with ``n_obs_history`` evenly-spaced frames
        (interval = ``history_interval``).  Early in an episode, missing
        history slots are zero-padded and marked in ``obs_history_is_pad``.

        Returns a new dict with:
        - ``"state"`` expanded to ``(B, T, D)``
        - each image key expanded to ``(B, T, C, H, W)``
        - ``"obs_history_is_pad"`` as ``(B, T)`` bool
        """
        assert self.config.n_obs_history is not None
        n_hist: int = self.config.n_obs_history
        interval = self.config.history_interval or 1
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

        if self.config.n_obs_history is not None and self.config.n_obs_history > 1:
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
            self._action_queue.extend(actions[delay:])
            assert len(self._action_queue) == self.config.n_action_steps, (
                f"Action queue must have {self.config.n_action_steps} actions"
            )

        action = self._action_queue.popleft()
        return action

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

        batch = self.normalize_inputs(batch)

        self._hydrate_optional_conditioning_batch(batch)

        obs_history_is_pad = batch.get("obs_history_is_pad")

        videos, vid_masks = self.prepare_videos(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        metadata_tokens, metadata_masks = self.prepare_metadata(batch)
        subgoal_videos, subgoal_vid_masks = self.prepare_subgoal_images(batch)

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=lang_tokens.device)

        if action_prefix is None:
            bsize = lang_tokens.shape[0]
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            action_prefix = torch.zeros(actions_shape, dtype=lang_tokens.dtype, device=lang_tokens.device)
        else:
            action_prefix = self.normalize_targets({"actions": action_prefix})["actions"]
            action_prefix = F.pad(
                action_prefix,
                (0, 0, 0, self.config.chunk_size - action_prefix.shape[1]),
            )

        state = self.prepare_state(batch)

        assert state.ndim == 3, f"Expected state (B, T, D) but got {state.shape}"

        t_dim = state.shape[1]
        if self.config.n_obs_history is not None and self.config.n_obs_history > 1 and t_dim == 1:
            logging.warning(
                "n_obs_history=%d but state has T=%d (single timestep). "
                "History buffering may not have been called.",
                self.config.n_obs_history,
                t_dim,
            )

        actions = self.model.sample_actions(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            action_prefix,
            delay,
            state,
            noise=noise,
            response_tokens=response_tokens,
            response_masks=response_masks,
            subgoal_videos=subgoal_videos,
            subgoal_vid_masks=subgoal_vid_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"actions": actions})["actions"]

        return actions

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Batch of data containing environment observations, actions, and targets.
            noise: Optional noise tensor.
            time: Optional time tensor.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE").
        """
        batch = self.normalize_inputs(batch)
        batch["discrete_actions"] = self.normalize_discrete_actions(dict(batch))["actions"]
        batch = self.normalize_targets(batch)

        self._hydrate_optional_conditioning_batch(batch)

        obs_history_is_pad = batch.get("obs_history_is_pad")

        videos, vid_masks = self.prepare_videos(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        metadata_tokens, metadata_masks = self.prepare_metadata(batch)
        subgoal_videos, subgoal_vid_masks = self.prepare_subgoal_images(batch)
        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(batch)
        actions = batch["actions"]
        actions_is_pad = batch.get("action_is_pad")

        state = self.prepare_state(batch)
        losses = self.model.forward(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            actions,
            actions_is_pad,
            noise,
            time,
            discrete_actions,
            discrete_action_masks,
            state=state,
            response_tokens=response_tokens,
            response_masks=response_masks,
            subgoal_videos=subgoal_videos,
            subgoal_vid_masks=subgoal_vid_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]

        return {"MSE": mse_loss, "CE": ce_loss}

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
                or if ``n_obs_history > 1`` but a 2-D state is provided.
        """
        state = batch["state"]

        if state.ndim == 2:
            if self.config.n_obs_history is not None and self.config.n_obs_history > 1:
                raise ValueError(
                    f"n_obs_history={self.config.n_obs_history} requires a "
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
        to [-1, 1] internally.  Padded history frames are handled inside the
        video encoder via temporal attention masking (not pixel zeroing).

        Args:
            batch: Batch of data containing image/video tensors.

        Returns:
            A tuple (videos, vid_masks) where each element is a list
            with one entry per camera.
        """
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
                vid = vid.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

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
        """Preprocess subgoal images for SpaceTime SigLIP embedding.

        Derives subgoal keys from ``config.image_features``: for each
        ``camera{k}`` the corresponding batch key is ``subgoal{k}``.
        If no ``subgoal{k}`` keys are present in the batch, zero-filled
        ``(B, 1, C, H, W)`` tensors with masks all ``False`` are returned
        so that the prefix sequence length stays fixed.

        Pixel values stay in ``[0, 1]``; the SpaceTime SigLIP encoder
        rescales to ``[-1, 1]`` internally.  Each image is unsqueezed to
        ``(B, 1, C, H, W)`` (single-frame video) so it can be passed
        directly to :meth:`embed_video`.

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
            A tuple ``(subgoal_videos, subgoal_vid_masks)`` of lists,
            where each video has shape ``(B, 1, C, H, W)`` in ``[0, 1]``.
        """
        subgoal_videos: list[Tensor] = []
        subgoal_vid_masks: list[Tensor] = []

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
            # the dataloader emits ``subgoal{k}`` tensors. Cleared masks yield the
            # same prefix as training with ``subgoal_is_pad=True`` everywhere
            # (comma + ``Subgoal:`` + image tokens are fully masked out).
            h, w = self.config.resize_imgs_with_padding or (224, 224)
            for _ in subgoal_keys:
                subgoal_videos.append(torch.zeros(bsize, 1, 3, h, w, device=device))
                subgoal_vid_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))
            return subgoal_videos, subgoal_vid_masks

        last_vid: Tensor | None = None
        last_mask: Tensor | None = None

        for key in present_keys:
            subgoal_img = batch[key]  # (B, C, H, W)

            if self.config.resize_imgs_with_padding is not None:
                subgoal_img = resize_with_pad(subgoal_img, *self.config.resize_imgs_with_padding, pad_value=0)

            vid = subgoal_img.unsqueeze(1)  # (B, 1, C, H, W), pixels in [0, 1]

            vid_device = vid.device
            mask = torch.ones(bsize, dtype=torch.bool, device=vid_device)

            is_pad = subgoal_is_pad.to(device=vid_device, dtype=torch.bool)
            mask = mask & ~is_pad
            vid = vid * (~is_pad)[:, None, None, None, None]

            subgoal_videos.append(vid)
            subgoal_vid_masks.append(mask)
            last_vid = vid
            last_mask = mask

        # Pad to len(subgoal_keys) regardless of empty_cameras so the prefix
        # length matches the all-missing-keys path above. With mixed
        # present/missing keys and empty_cameras=0, the prior loop produced
        # fewer slots than subgoal_keys and made the prefix length depend on
        # which subgoal{k} keys happened to be in the batch.
        if last_vid is not None and last_mask is not None:
            for _ in missing_keys:
                subgoal_videos.append(torch.zeros_like(last_vid))
                subgoal_vid_masks.append(torch.zeros_like(last_mask))

        return subgoal_videos, subgoal_vid_masks

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
        - **Mixed keys** — only some of the six keys present: fill each missing
          key with that same default so behavior matches a batch where every key
          was specified explicitly.

        Per-sample on/off is controlled only by ``*_is_pad`` after hydration;
        varying fields across rows uses the batch tensors, not missing dict keys.
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

        Builds strings ``Metadata: Speed: … Quality: … Mistake: …`` only from
        fields whose ``*_is_pad`` flag is ``False``. For each sample, if
        ``speed_is_pad`` / ``quality_is_pad`` / ``mistake_is_pad`` is ``True``,
        that field is omitted entirely (not concatenated to ``segments``). When
        every flag is pad, the row is ``""`` before tokenization.

        Always runs :meth:`_hydrate_metadata_batch` first so callers see the same
        outcomes whether they hit :meth:`sample_actions` (often no metadata
        keys), :meth:`forward` with a partial-key batch, or a dataloader batch
        with every key set.

        Values are normalized to shape ``(B,)`` on ``state``'s device, with
        scalar tensors broadcast like ``subgoal_is_pad``.

        .. note::
            ``speed_is_pad`` / ``quality_is_pad`` / ``mistake_is_pad`` each
            default to ``torch.ones(B, dtype=bool)`` (treat-as-pad) when the
            key is missing from ``batch``. This is a behavior change from
            the previous treat-as-real (``torch.zeros``) default. Hand-built
            inference batches that supply ``"speed"`` / ``"quality"`` /
            ``"mistake"`` but omit the corresponding ``_is_pad`` flag will
            now have those metadata fields silently dropped from the prefix
            string — pass ``..._is_pad=torch.zeros(...)`` explicitly when the
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
        pad_speed = _row_bool("speed_is_pad")
        pad_quality = _row_bool("quality_is_pad")
        pad_mistake = _row_bool("mistake_is_pad")

        metadata_rows: list[str] = []
        for speed, quality, mistake, speed_is_pad, quality_is_pad, mistake_is_pad in zip(
            speed_t,
            quality_t,
            mistake_t,
            pad_speed,
            pad_quality,
            pad_mistake,
            strict=True,
        ):
            segments: list[str] = []
            # *_is_pad True → omit that field from the metadata string (same as dataloader mask).
            if not speed_is_pad.item():
                segments.append(f"Speed: {speed.item()}")
            if not quality_is_pad.item():
                segments.append(f"Quality: {quality.item()}")
            if not mistake_is_pad.item():
                segments.append(f"Mistake: {str(mistake.item())}")
            metadata_rows.append(f"Metadata: {', '.join(segments)}" if segments else "")

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


class PI07LowLevelPlannerFlowMatching(nn.Module):
    """π07 Low-Level Planner — A Vision-Language-Action Flow Matching model with
    SpaceTime SigLIP video, subtask/subgoal conditioning, and episode metadata.

    Architecturally inherits the π0.5 PaliGemma + Gemma-Expert flow-matching
    backbone ([π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf))
    and extends the prefix to ingest, on top of the original (image, language,
    state, discrete actions) tokens:

    * **Image history** — encoded by :class:`SpaceTimeSiglipVideoEncoder`.
      ``n_obs_history`` frames per camera; T=1 is byte-identical to plain SigLIP,
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

    def __init__(self, config: PI07lowlevelPlannerConfig, discrete_action_vocab_size: int | None = None):
        """Initializes the PI07LowLevelPlannerFlowMatching model.

        Args:
            config: Model configuration.
            discrete_action_vocab_size: Size of the discrete action vocabulary.
        """
        super().__init__()
        self.config = config

        load_pretrained_paligemma = (
            self.config.init_strategy == "expert_only_he_init"
        )  # only load pretrained paligemma if we are He-initializing the expert only
        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            load_pretrained_paligemma=load_pretrained_paligemma,
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        n_obs_steps = self.config.n_obs_history if self.config.n_obs_history is not None else 1
        self.video_encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=self.paligemma_with_expert.paligemma.vision_tower,
            multi_modal_projector=self.paligemma_with_expert.paligemma.multi_modal_projector,
            max_num_frames=n_obs_steps,
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

        self._init_model()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using He (Kaiming) initialization.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_model(self) -> None:
        """Initialize the model weights based on the configuration."""
        if self.config.init_strategy == "no_init":
            return
        elif self.config.init_strategy == "full_he_init":
            for m in self.modules():
                self._init_weights(m)
        elif self.config.init_strategy == "expert_only_he_init":
            for m in self.paligemma_with_expert.gemma_expert.modules():
                self._init_weights(m)
        else:
            raise ValueError(f"Invalid init strategy: {self.config.init_strategy}")

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

    def embed_prefix(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        state: Tensor | None = None,
        *,
        response_tokens: Tensor,
        response_masks: Tensor,
        metadata_tokens: Tensor,
        metadata_masks: Tensor,
        subgoal_videos: list[Tensor] = (),
        subgoal_vid_masks: list[Tensor] = (),
        obs_history_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed video/images with SpaceTime SigLIP and language tokens to prepare
        for PaliGemma transformer processing.

        Args:
            videos: List of video tensors, each (B, T, C, H, W) in [0, 1].
            vid_masks: List of boolean masks, each (B,).
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            discrete_actions: Optional discrete action tensor.
            discrete_action_masks: Optional discrete action mask tensor.
            state: Optional continuous state tensor of shape ``(B, T, max_state_dim)``.
            response_tokens: Subtask response token IDs ``(B, response_max_length)``.
            response_masks: Boolean mask for response tokens (never ``None``).
            metadata_tokens: Metadata token IDs ``(B, metadata_max_length)`` (never ``None``).
            metadata_masks: Boolean mask for metadata tokens (never ``None``).
            subgoal_videos: List of subgoal video tensors, each (B, 1, C, H, W) in [0, 1].
            subgoal_vid_masks: List of boolean masks, each (B,).
            obs_history_is_pad: Optional ``(B, T)`` boolean mask where ``True``
                marks padded (missing) history timesteps.

        Returns:
            A tuple (embs, pad_masks, att_masks).
        """
        embs = []
        pad_masks = []
        att_masks = []

        for vid, vid_mask in zip(videos, vid_masks, strict=False):
            vid_emb = self.embed_video(vid, obs_history_is_pad=obs_history_is_pad)
            vid_emb = vid_emb.to(dtype=_preferred_dtype())

            bsize, num_vid_embs = vid_emb.shape[:2]
            vid_mask = vid_mask[:, None].expand(bsize, num_vid_embs)

            embs.append(vid_emb)
            pad_masks.append(vid_mask)

            att_masks += [0] * num_vid_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_indicator_ids = self.language_tokenizer.encode("State: ", add_special_tokens=False)
        state_indicator_tokens = torch.tensor([state_indicator_ids] * bsize, device=lang_tokens.device)
        state_indicator_emb = self.paligemma_with_expert.embed_language_tokens(state_indicator_tokens)
        state_indicator_emb = state_indicator_emb * math.sqrt(state_indicator_emb.shape[-1])
        state_indicator_mask = torch.ones(
            bsize, state_indicator_emb.shape[1], dtype=torch.bool, device=lang_tokens.device
        )
        embs.append(state_indicator_emb)
        pad_masks.append(state_indicator_mask)
        att_masks += [0] * state_indicator_emb.shape[1]

        # state: (B, T, max_state_dim) — one VLM token per timestep
        state_emb = self.state_proj(state.to(dtype=_preferred_dtype()))  # (B, T, vlm_hidden)
        t_steps = state_emb.shape[1]

        if obs_history_is_pad is not None:
            state_mask = ~obs_history_is_pad  # (B, T)
        else:
            # Absent → assume all history is padded; only current step is real.
            state_mask = torch.zeros(bsize, t_steps, dtype=torch.bool, device=state.device)
        state_mask[:, -1] = True

        embs.append(state_emb)
        pad_masks.append(state_mask)
        att_masks += [0] * t_steps

        # Per-sample flag: True for samples that have at least one real
        # response token, False for samples whose response was dropped.
        sample_has_response = response_masks.any(dim=1)  # (B,)

        # ", " separator — only unmasked for samples with response
        comma_ids = self.language_tokenizer.encode(", ", add_special_tokens=False)
        comma_tokens = torch.tensor([comma_ids] * bsize, device=lang_tokens.device)
        comma_emb = self.paligemma_with_expert.embed_language_tokens(comma_tokens)
        comma_emb = comma_emb * math.sqrt(comma_emb.shape[-1])
        num_comma_tokens = comma_emb.shape[1]
        comma_mask = sample_has_response[:, None].expand(bsize, num_comma_tokens)
        embs.append(comma_emb)
        pad_masks.append(comma_mask)
        att_masks += [0] * num_comma_tokens

        # Response tokens — already per-sample masked via response_masks
        response_emb = self.paligemma_with_expert.embed_language_tokens(response_tokens)
        response_emb_dim = response_emb.shape[-1]
        response_emb = response_emb * math.sqrt(response_emb_dim)
        embs.append(response_emb)
        pad_masks.append(response_masks)
        num_response_embs = response_emb.shape[1]
        att_masks += [1] + [0] * (num_response_embs - 1)

        # --- Subgoal block (per-sample masked) ---
        if subgoal_vid_masks:
            sample_has_subgoal = torch.stack(subgoal_vid_masks, dim=0).any(dim=0)  # (B,)
        else:
            sample_has_subgoal = torch.zeros(bsize, dtype=torch.bool, device=lang_tokens.device)

        # ", " separator between response/state and subgoal header
        sg_comma_ids = self.language_tokenizer.encode(", ", add_special_tokens=False)
        sg_comma_tokens = torch.tensor([sg_comma_ids] * bsize, device=lang_tokens.device)
        sg_comma_emb = self.paligemma_with_expert.embed_language_tokens(sg_comma_tokens)
        sg_comma_emb = sg_comma_emb * math.sqrt(sg_comma_emb.shape[-1])
        num_sg_comma = sg_comma_emb.shape[1]
        sg_comma_mask = sample_has_subgoal[:, None].expand(bsize, num_sg_comma)
        embs.append(sg_comma_emb)
        pad_masks.append(sg_comma_mask)
        att_masks += [0] * num_sg_comma

        # "Subgoal: " header
        sg_start_ids = self.language_tokenizer.encode("Subgoal: ", add_special_tokens=False)
        sg_start_tokens = torch.tensor([sg_start_ids] * bsize, device=lang_tokens.device)
        sg_start_emb = self.paligemma_with_expert.embed_language_tokens(sg_start_tokens)
        sg_start_emb = sg_start_emb * math.sqrt(sg_start_emb.shape[-1])
        num_sg_start = sg_start_emb.shape[1]
        sg_start_mask = sample_has_subgoal[:, None].expand(bsize, num_sg_start)
        embs.append(sg_start_emb)
        pad_masks.append(sg_start_mask)
        att_masks += [1] + [0] * (num_sg_start - 1)

        # Subgoal image embeddings via SpaceTime SigLIP. When every sample
        # masks subgoal (common with ``subgoal_drop_prob≈1`` or eval without
        # subgoal tensors), skip the vision backbone — zeros match padded slots
        # and avoid ~2× redundant SigLIP forwards per step (OOM on tight VRAM).
        any_subgoal_in_batch = bool(sample_has_subgoal.any().item())
        vlm_h = self.paligemma_with_expert.config.paligemma_config.text_config.hidden_size
        n_sg_tokens = self.video_encoder.num_video_tokens
        sg_dtype = _preferred_dtype()
        # Subgoals are single-frame (B, 1, C, H, W); the encoder accepts any
        # T in [1, max_num_frames] and short-circuits the temporal sublayer at
        # T=1, so we forward the subgoal as-is — no pad-to-history needed.

        for sg_vid, sg_vid_mask in zip(subgoal_videos, subgoal_vid_masks, strict=True):
            if any_subgoal_in_batch:
                sg_vid_emb = self.embed_video(sg_vid)
                sg_vid_emb = sg_vid_emb.to(dtype=sg_dtype)
            else:
                sg_vid_emb = torch.zeros(bsize, n_sg_tokens, vlm_h, device=lang_tokens.device, dtype=sg_dtype)

            bsize_sg, num_sg_embs = sg_vid_emb.shape[:2]
            sg_mask_expanded = sg_vid_mask[:, None].expand(bsize_sg, num_sg_embs)

            embs.append(sg_vid_emb)
            pad_masks.append(sg_mask_expanded)
            att_masks += [0] * num_sg_embs

        # Metadata (speed / quality / mistake), same layout as pi07_paligemma
        sample_has_metadata = metadata_masks.any(dim=1)  # (B,)

        md_comma_ids = self.language_tokenizer.encode(", ", add_special_tokens=False)
        md_comma_tokens = torch.tensor([md_comma_ids] * bsize, device=lang_tokens.device)
        md_comma_emb = self.paligemma_with_expert.embed_language_tokens(md_comma_tokens)
        md_comma_emb = md_comma_emb * math.sqrt(md_comma_emb.shape[-1])
        num_md_comma = md_comma_emb.shape[1]
        md_comma_mask = sample_has_metadata[:, None].expand(bsize, num_md_comma)
        embs.append(md_comma_emb)
        pad_masks.append(md_comma_mask)
        att_masks += [0] * num_md_comma

        metadata_emb = self.paligemma_with_expert.embed_language_tokens(metadata_tokens)
        metadata_emb = metadata_emb * math.sqrt(metadata_emb.shape[-1])
        embs.append(metadata_emb)
        pad_masks.append(metadata_masks)
        num_md_embs = metadata_emb.shape[1]
        att_masks += [1] + [0] * (num_md_embs - 1)

        # ":\n" always terminates the state(+response+subgoal+metadata) block
        state_end_ids = self.language_tokenizer.encode(":\n", add_special_tokens=False)
        state_end_tokens = torch.tensor([state_end_ids] * bsize, device=lang_tokens.device)
        state_end_emb = self.paligemma_with_expert.embed_language_tokens(state_end_tokens)
        state_end_emb = state_end_emb * math.sqrt(state_end_emb.shape[-1])
        state_end_mask = torch.ones(
            bsize, state_end_emb.shape[1], dtype=torch.bool, device=lang_tokens.device
        )
        embs.append(state_end_emb)
        pad_masks.append(state_end_mask)
        att_masks += [0] * state_end_emb.shape[1]

        if discrete_actions is not None:
            discrete_action_indicator_ids = self.language_tokenizer.encode(
                "Action: ", add_special_tokens=False
            )
            discrete_action_indicator_tokens = torch.tensor(
                [discrete_action_indicator_ids] * bsize, device=lang_tokens.device
            )
            discrete_action_indicator_emb = self.paligemma_with_expert.embed_language_tokens(
                discrete_action_indicator_tokens
            )
            discrete_action_indicator_emb = discrete_action_indicator_emb * math.sqrt(
                discrete_action_indicator_emb.shape[-1]
            )
            discrete_action_indicator_mask = torch.ones(
                bsize, discrete_action_indicator_emb.shape[1], dtype=torch.bool, device=lang_tokens.device
            )
            embs.append(discrete_action_indicator_emb)
            pad_masks.append(discrete_action_indicator_mask)
            att_masks += [1] * discrete_action_indicator_emb.shape[1]

            discrete_action_emb = self.paligemma_with_expert.embed_discrete_actions(discrete_actions)
            embs.append(discrete_action_emb.to(dtype=_preferred_dtype()))
            pad_masks.append(discrete_action_masks)
            att_masks += [1] * discrete_action_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            noisy_actions: Tensor containing noisy actions.
            timestep: Tensor containing timesteps of shape (batch_size, action_chunk_length).

        Returns:
            A tuple containing:
                - embs: Concatenated embeddings tensor.
                - pad_masks: Concatenated padding masks tensor.
                - att_masks: Attention masks tensor.
                - adarms_cond: AdaRMS conditioning tensor.
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = noisy_actions.shape[0]
        dtype = _preferred_dtype()
        device = noisy_actions.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )

        # Fuse timestep + action information using an MLP
        noisy_actions = noisy_actions.to(dtype=dtype)
        action_emb = self.action_in_proj(noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = time_emb.to(dtype=dtype)
        adarms_cond = time_mlp_func(time_emb)

        # Add to input tokens
        embs.append(action_emb)

        bsize, action_dim = action_emb.shape[:2]
        action_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        state: Tensor | None = None,
        *,
        response_tokens: Tensor,
        response_masks: Tensor,
        metadata_tokens: Tensor,
        metadata_masks: Tensor,
        subgoal_videos: list[Tensor] = (),
        subgoal_vid_masks: list[Tensor] = (),
        obs_history_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute the loss.

        Args:
            videos: List of video tensors, each (B, T, C, H, W) in [0, 1].
            vid_masks: List of boolean masks, each (B,).
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            actions: Action tensor.
            actions_is_pad: Optional action is padded mask tensor.
            noise: Optional noise tensor.
            time: Optional time tensor.
            discrete_actions: Optional discrete action tensor.
            discrete_action_masks: Optional discrete action mask tensor.
            state: Optional continuous state tensor of shape ``(B, T, max_state_dim)``.
            response_tokens: Subtask response token IDs ``(B, response_max_length)``.
            response_masks: Boolean mask for response tokens.
            metadata_tokens: Metadata token IDs ``(B, metadata_max_length)``.
            metadata_masks: Boolean mask for metadata tokens.
            subgoal_videos: List of subgoal video tensors, each (B, 1, C, H, W) in [0, 1].
            subgoal_vid_masks: List of boolean masks, each (B,).
            obs_history_is_pad: Optional ``(B, T)`` bool mask for padded history steps.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE").
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            discrete_actions,
            discrete_action_masks,
            state=state,
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            subgoal_videos=subgoal_videos,
            subgoal_vid_masks=subgoal_vid_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # avoids using discrete action for predicting continuous flow matching action
        num_cross_att_tokens = (
            prefix_embs.shape[1]
            - self.config.discrete_action_indicator_max_length
            - self.config.discrete_action_max_length
        )

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

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        # We should skip the discrete action tokens as well as the discrete action indicator tokens when numbering the position ids for the action expert
        prefix_offsets = torch.sum(
            prefix_pad_masks[
                :,
                : -self.config.discrete_action_indicator_max_length - self.config.discrete_action_max_length,
            ],
            dim=-1,
        )[:, None]  # action expert position ids start after prefix
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
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        mse_loss = F.mse_loss(u_t, v_t, reduction="none")

        # mask out frozen actions and padded actions
        postfix_mask = rearrange(
            torch.logical_not(prefix_mask), "b c -> b c 1"
        )  # 0 for frozen actions, 1 for non-frozen actions

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            in_episode_bound = rearrange(
                in_episode_bound, "b c -> b c 1"
            )  # 0 for padded actions, 1 for non-padded actions
            postfix_mask = torch.logical_and(postfix_mask, in_episode_bound)

        mse_loss = mse_loss * postfix_mask

        # Remove padding
        mse_loss = mse_loss[:, :, : self.config.max_action_dim]

        # Do not include frozen actions and padded actions in the mean loss calculation
        postfix_mask_expanded = repeat(postfix_mask, "b c 1 -> b c d", d=mse_loss.shape[-1])
        mse_loss = mse_loss.sum() / (postfix_mask_expanded.sum() + 1e-8)

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

    def sample_actions(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        action_prefix: Tensor,
        delay: Tensor,
        state: Tensor | None = None,
        noise: Tensor | None = None,
        *,
        response_tokens: Tensor,
        response_masks: Tensor,
        metadata_tokens: Tensor,
        metadata_masks: Tensor,
        subgoal_videos: list[Tensor] = (),
        subgoal_vid_masks: list[Tensor] = (),
        obs_history_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action.

        Args:
            videos: List of video tensors, each (B, T, C, H, W) in [0, 1].
            vid_masks: List of boolean masks, each (B,).
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            action_prefix: Action prefix tensor.
            delay: Number of delay actions.
            state: Optional continuous state tensor of shape ``(B, T, max_state_dim)``.
            noise: Optional noise tensor.
            response_tokens: Subtask response token IDs ``(B, response_max_length)``.
            response_masks: Boolean mask for response tokens.
            metadata_tokens: Metadata token IDs ``(B, metadata_max_length)``.
            metadata_masks: Boolean mask for metadata tokens.
            subgoal_videos: List of subgoal video tensors, each (B, 1, C, H, W) in [0, 1].
            subgoal_vid_masks: List of boolean masks, each (B,).

        Returns:
            The sampled action tensor.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            state=state,
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            subgoal_videos=subgoal_videos,
            subgoal_vid_masks=subgoal_vid_masks,
            obs_history_is_pad=obs_history_is_pad,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1]

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
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
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

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
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
