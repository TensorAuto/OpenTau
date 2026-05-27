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

"""π07 Low-Level Component: a Vision-Language-Action Flow Model for continuous
action generation.

The low-level component is one half of the π07 hierarchical architecture.
Given video observations (encoded by SpaceTimeSiglip), a language prompt, an
optional subtask response from the high-level planner, temporal
proprioceptive state, optional subgoal images, and optional metadata, it
produces continuous action chunks via flow matching while simultaneously
predicting discrete (FAST) action tokens through the VLM backbone.

Key differences from the base π05 policy:
  1. SpaceTimeSiglip video encoder replaces SigLIP, compressing each camera's
     temporal video into 256 tokens via a Perceiver cross-attention reducer.
  2. Temporal state sequences (B, T, D) are projected per-timestep into
     separate continuous tokens for the Gemma backbone.
  3. Supports optional subtask response, subgoal image, and metadata
     conditioning for hierarchical planning.
  4. Knowledge Insulation: action-expert gradients are detached from the
     VLM backbone to preserve language understanding capabilities.
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
from opentau.policies.normalize import resolve_num_datasets as _num_datasets
from opentau.policies.pi07.gemma3_with_expert import (
    Gemma3WithExpertModel,
)
from opentau.policies.pi07.low_level.configuration_pi07_low_level import (
    PI07LowLevelConfig,
)
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.utils import get_safe_dtype


def _preferred_dtype():
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def _global_or_branch_decisions(
    presence_locals: tuple[bool, ...],
    any_locals: tuple[bool, ...],
    field_names: tuple[str, ...],
    device: torch.device,
) -> tuple[bool, ...]:
    """Synchronize ``embed_prefix`` branch decisions across distributed ranks.

    pi07's ``embed_prefix`` chooses Python-level branches based on whether the
    local rank's micro-batch contains any subgoal images / response tokens /
    metadata tokens. Each branch fires a different number of FSDP / ZeRO-3
    all-gather collectives (one per ``embed_language_tokens`` call), so under
    stochastic dataset drop probabilities — where one rank's batch may roll
    "all dropped" while another rolls "some real" — the ranks would otherwise
    take different branches and deadlock at NCCL with mismatched all-gather
    sequence numbers.

    This helper bundles two synchronizations into a *single* SUM all-reduce
    (vs N independent 1-element MAX all-reduces before):

    1. ``any_locals`` are OR-reduced across ranks. The returned ``has_*``
       booleans drive the Python branches that fire collectives, so all
       ranks must end up taking the same branch.
    2. ``presence_locals`` are checked for cross-rank agreement. If any rank
       has a field as None while another has it as a Tensor (a future
       heterogeneous-mixture regression), this raises immediately rather
       than letting the inner ``if has_response and response_tokens is not
       None`` gates silently re-introduce the desync. Today every rank
       instantiates the same ``embed_prefix`` contract, so divergence is a
       loud bug, not normal operation.

    Outside ``torch.distributed`` (single-process runs, CPU smoke tests)
    the helper is a no-op that just returns the local ``any_locals``.

    Args:
        presence_locals: Per-field bools — ``True`` iff the local rank has a
            non-None Tensor for that field. Checked for agreement only;
            never returned.
        any_locals: Per-field bools — ``True`` iff the local rank's mask has
            any non-False entry. OR-reduced and returned. Each entry implies
            the field is present locally (``any_locals[i] => presence_locals[i]``).
        field_names: Human-readable field names parallel to the lists above
            (used in the divergence error message).
        device: Device to allocate the all-reduce buffer on; must match the
            distributed backend (typically the policy's CUDA device).

    Returns:
        A tuple of OR-reduced ``has_*`` booleans, one per ``any_locals`` entry,
        in the same order.

    Raises:
        RuntimeError: If any field's ``presence_locals`` value differs across
            ranks. Error names the divergent field(s).
    """
    n = len(any_locals)
    if not (len(presence_locals) == n and len(field_names) == n):
        raise ValueError(
            f"_global_or_branch_decisions: presence_locals/any_locals/field_names "
            f"must have the same length, got "
            f"{len(presence_locals)}/{n}/{len(field_names)}."
        )
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tuple(bool(a) for a in any_locals)
    # One SUM all-reduce over [presence | any]: SUM lets us derive both the
    # OR (sum > 0) for the any flags and the cross-rank-agreement check
    # (sum == 0 or sum == world_size) for the presence flags.
    flag = torch.tensor(
        [int(bool(p)) for p in presence_locals] + [int(bool(a)) for a in any_locals],
        device=device,
        dtype=torch.int32,
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
    world_size = torch.distributed.get_world_size()
    counts = flag.tolist()
    presence_counts = counts[:n]
    any_counts = counts[n:]
    diverged = [(name, c) for name, c in zip(field_names, presence_counts, strict=True) if 0 < c < world_size]
    if diverged:
        raise RuntimeError(
            "_global_or_branch_decisions: ranks disagree on field presence "
            "(None vs Tensor). embed_prefix assumes per-field presence is a "
            "global property of the dataset config; if a heterogeneous-mixture "
            "change makes this no longer true, the None-vs-Tensor inner gates "
            "would re-introduce the FSDP / ZeRO-3 desync. Divergent fields "
            f"(name, ranks_with_non_None / world_size={world_size}): {diverged}."
        )
    return tuple(c > 0 for c in any_counts)


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
        A tensor of shape (batch_size, action_chunk_length, dimension).
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

    Args:
        pad_masks: bool[B, N] true if its part of the input, false if padding.
        att_masks: int32[B, N] mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.
        n_cross_att_tokens: Add attention mask for cross-attention tokens if provided.
        cross_att_pad_masks: Padding masks for cross attention tokens.

    Returns:
        A 2D attention mask tensor.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

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

        cross_att_mask = cross_att_mask & pad_masks[:, :, None] & cross_att_pad_masks[:, None, :]
        att_2d_masks = torch.cat((cross_att_mask, att_2d_masks), dim=2)

    return att_2d_masks


def resize_with_pad(img: Tensor, width: int, height: int, pad_value: int = -1) -> Tensor:
    """Resizes an image to fit within the specified dimensions while maintaining aspect ratio,
    and pads the remaining area.

    Args:
        img: Input image tensor of shape (batch_size, channels, current_height, current_width).
        width: Target width.
        height: Target height.
        pad_value: Value to use for padding. Defaults to -1.

    Returns:
        The resized and padded image tensor of shape (batch_size, channels, height, width).
    """
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

    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_discrete_tokens(tokens: list[list[int]], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pads or truncates a list of discrete action token sequences to a fixed length.

    Args:
        tokens: A list of discrete action token sequences.
        max_length: The target length.

    Returns:
        A tuple of (discrete_action_tokens, discrete_action_masks) numpy arrays.
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


# Policy wrapper
class PI07LowLevelPolicy(PreTrainedPolicy):
    """Policy wrapper for the π07 low-level component.

    Handles tokenization, normalization, observation-history buffering, and
    action queue management around the inner
    :class:`PI07LowLevelFlowMatching` model.
    """

    config_class = PI07LowLevelConfig
    name = "pi07_low_level"

    def __init__(
        self,
        config: PI07LowLevelConfig,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
    ):
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

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            config.discrete_action_tokenizer_path, trust_remote_code=True
        )
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI07LowLevelFlowMatching(config, discrete_action_vocab_size=discrete_action_vocab_size)

        self.reset()

    def reset(self) -> None:
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # Observation history buffers for inference.
        self._obs_buffers: dict[str, deque] = {}
        self._state_buffer: deque | None = None

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
        """Override the from_pretrained method to handle key remapping."""
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download if resume_download is not None else False,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)

        acc = get_proc_accelerator()
        is_main_process = acc.is_main_process if acc else True
        # Populated inside the try block when skip_normalization_weights fires;
        # used outside the try/except to gate the inf-buffer guard so the
        # ValueError is not swallowed by the broad except below.
        stripped_keys: frozenset[str] = frozenset()
        try:
            if is_main_process:
                logging.info("Loading model from: %s", pretrained_name_or_path)
            try:
                from transformers.utils.hub import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                assert resolved_file is not None, "cached_file returned None"
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                if is_main_process:
                    logging.info("Loaded state dict from model.safetensors")
            except Exception as e:
                if is_main_process:
                    logging.warning("Could not load state dict from remote files: %s", e)
                    logging.info("Returning model without loading pretrained weights")
                return model

            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

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

            # Promote legacy single-dataset Normalize/Unnormalize buffers from
            # `(*feat_shape,)` to the new `(1, *feat_shape)` stacked layout so pre-PR
            # checkpoints load via `model.load_state_dict(...)`. Always run
            # (outside the `if remap_count > 0` block) — promotion is needed
            # whether or not any other keys were renamed.
            model._promote_legacy_norm_buffers_in_state_dict(remapped_state_dict)

            # Strip saved normalize/unnormalize buffers when the user opted in
            # via config.skip_normalization_weights — see PreTrainedConfig and
            # PreTrainedPolicy._strip_normalization_buffers_from_state_dict.
            remapped_state_dict, stripped_keys = cls._strip_normalization_buffers_from_state_dict(
                remapped_state_dict, model.config, is_main_process=is_main_process
            )
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
        # broad except above. The helper no-ops when ``stripped_keys`` is
        # empty (flag was off or the try block bailed before the strip ran).
        cls._assert_normalize_buffers_initialized(model, stripped_keys=stripped_keys)

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, Tensor], model_config: PreTrainedConfig
    ) -> dict[str, Tensor]:
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            # Accept legacy `paligemma_with_expert.*` prefixes from
            # `pi07_paligemma` checkpoints as a warm-start path; the rest of
            # the AdaRMS / time-mlp / patch_embedding handling below applies
            # uniformly to both prefixes.
            if key.startswith("paligemma_with_expert."):
                key = key.replace("paligemma_with_expert.", "gemma3_with_expert.", 1)

            new_key = key

            if re.match(
                r"gemma3_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.gemma3_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"gemma3_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(
                    self.model.gemma3_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            if "patch_embedding" in key:
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Currently not implemented for PI07 low-level component")

    def _build_history_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Buffer the current observation and construct a temporal batch.

        Appends the single-frame observation from ``batch`` to internal deque
        buffers, then assembles a batch with ``n_obs_steps`` evenly-spaced
        frames (interval = ``history_interval``). Early in an episode the
        buffer is partially filled, so some slots are zero-padded; the
        returned ``"obs_history_is_pad"`` (B, T) bool tensor flags those
        slots ``True`` so the model can mask them out of attention. Once the
        buffer is full (typically a handful of steps in), the mask is all
        ``False`` and the encoder uses the real history.

        Expected batch keys:
            - ``"state"``: (B, D) current proprioceptive state.
            - image keys matching ``config.image_features``: (B, C, H, W) camera frames.
            - ``"prompt"``: list[str] language instructions (passed through unchanged).
            - Any other metadata keys are forwarded unchanged.

        Returns a new dict with ``"state"`` expanded to (B, T, D), image keys
        expanded to (B, T, C, H, W), and a new ``"obs_history_is_pad"`` (B, T)
        bool tensor (``True`` = padded). T = ``n_obs_steps``.
        """
        n_hist: int = self.config.n_obs_steps
        interval = self.config.history_interval
        buf_maxlen = self.config.obs_buffer_size

        # initialise buffers on first call after reset()
        if self._state_buffer is None:
            self._state_buffer = deque(maxlen=buf_maxlen)
            self._obs_buffers = {}

        img_keys = [key for key in self.config.image_features if key in batch]
        for key in img_keys:
            if key not in self._obs_buffers:
                self._obs_buffers[key] = deque(maxlen=buf_maxlen)

        # append current observation
        self._state_buffer.append(batch["state"])  # (B, D)
        for key in img_keys:
            self._obs_buffers[key].append(batch[key])  # (B, C, H, W)

        # sample n_hist frames at the configured interval
        buf_len = len(self._state_buffer)
        missing = buf_maxlen - buf_len  # how many slots are still empty
        bsize = batch["state"].shape[0]
        device = batch["state"].device

        # Pass through all non-image, non-state keys (e.g. "prompt" and other metadata).
        temporal_batch = {key: v for key, v in batch.items() if key not in img_keys and key != "state"}

        # Build state tensor (B, T, D)
        state_frames = []
        for i in range(n_hist):
            idx = i * interval - missing  # index into current buffer
            if idx < 0:
                state_frames.append(torch.zeros_like(self._state_buffer[0]))
            else:
                state_frames.append(self._state_buffer[idx])
        temporal_batch["state"] = torch.stack(state_frames, dim=1)  # (B, T, D)

        # Build camera tensors (B, T, C, H, W)
        for key in img_keys:
            cam_frames = []
            for i in range(n_hist):
                idx = i * interval - missing
                if idx < 0:
                    cam_frames.append(torch.zeros_like(self._obs_buffers[key][0]))
                else:
                    cam_frames.append(self._obs_buffers[key][idx])
            temporal_batch[key] = torch.stack(cam_frames, dim=1)  # (B, T, C, H, W)

        # Same `idx < 0` decision as the loops above: a slot is padded iff the
        # buffer didn't have an entry to fill it. The pattern is identical
        # for state and every camera (they share the same buffer length), so
        # we emit one (B, T) mask. Broadcast across batch — every sample sees
        # the same padding pattern at any given step. Without this, the
        # encoder's None-fallback masks ALL history at inference (including
        # genuine mid-episode frames once the buffer is full); with it, only
        # the actually-padded start-of-episode slots get masked.
        pad_pattern = torch.tensor(
            [i * interval - missing < 0 for i in range(n_hist)],
            dtype=torch.bool,
            device=device,
        )
        temporal_batch["obs_history_is_pad"] = pad_pattern.unsqueeze(0).expand(bsize, n_hist)

        return temporal_batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action from the queue, regenerating if needed.

        Builds temporal observation history when configured, runs flow-matching
        inference to fill the action queue, and pops the next action.

        Args:
            batch: Environment observation dict with ``"state"``, image keys,
                ``"prompt"``, ``"response"``, and optional ``"metadata"``.
            noise: Optional pre-sampled noise for deterministic evaluation.

        Returns:
            A single action tensor of shape ``(B, action_dim)``.
        """
        self.eval()

        # Build temporal observation history if configured.
        if self.config.n_obs_steps > 1:
            batch = self._build_history_batch(batch)

        if len(self._action_queue) == 0 or len(self._action_queue) <= self.config.max_delay:
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

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        action_prefix: Tensor | None = None,
        delay: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample a full action chunk via flow-matching inference.

        Normalizes inputs, prepares all modalities (video, language, response,
        state, subgoal images, metadata), and delegates to the inner model's
        ``sample_actions`` for iterative denoising.

        Args:
            batch: Environment observation dict.
            action_prefix: Optional previously-committed actions for real-time
                inference with delay.
            delay: Number of prefix action steps already committed.
            noise: Optional pre-sampled noise for deterministic evaluation.

        Returns:
            Action chunk tensor of shape ``(B, n_action_steps, action_dim)``.
        """
        if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
            assert delay is None or 0 <= delay.item() <= self.config.max_delay, (
                f"Delay must be None or between 0 and {self.config.max_delay}"
            )

        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)

        # `_build_history_batch` (called from `select_action` upstream) emits
        # this; it's None when the caller skipped that step (e.g. n_obs_steps
        # is 1, or sample_actions is invoked directly without the buffer).
        obs_history_is_pad = batch.get("obs_history_is_pad")

        videos, vid_masks = self.prepare_videos(batch, obs_history_is_pad=obs_history_is_pad)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        state = self.prepare_state(batch)

        subgoal_images, subgoal_img_masks = self.prepare_subgoal_images(batch)

        metadata_tokens, metadata_masks = self.prepare_metadata(batch)

        # Shape checks: videos must be 5D (B, T, C, H, W), state must be 3D (B, T, D).
        for vid in videos:
            assert vid.ndim == 5, f"Expected 5D video tensor (B, T, C, H, W), got {vid.shape}"
        assert state.ndim == 3, f"Expected 3D state tensor (B, T, D), got {state.shape}"

        if self.config.n_obs_steps > 1:
            t_dim = state.shape[1]
            if t_dim == 1:
                logging.warning(
                    "Temporal dimension T=1: no historical frames included. "
                    "This should only happen at most %d time(s) at the start of an episode.",
                    self.config.history_interval,
                )

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=lang_tokens.device)

        if action_prefix is None:
            bsize = lang_tokens.shape[0]
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            action_prefix = torch.zeros(actions_shape, dtype=lang_tokens.dtype, device=lang_tokens.device)
        else:
            normalized = self.normalize_targets({"actions": action_prefix}, dataset_index)["actions"]
            action_prefix = F.pad(
                normalized,
                (0, 0, 0, self.config.chunk_size - normalized.shape[1]),
            )

        actions = self.model.sample_actions(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            state,
            action_prefix,
            delay,
            noise=noise,
            subgoal_images=subgoal_images,
            subgoal_img_masks=subgoal_img_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            response_tokens=response_tokens,
            response_masks=response_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        action_feature = self.config.action_feature
        assert action_feature is not None, "action_feature must be set in output_features"
        original_action_dim = action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"actions": actions}, dataset_index)["actions"]

        return actions

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Training forward pass: normalize, prepare modalities, and compute losses.

        Returns a dict with ``"MSE"`` (flow-matching velocity loss) and
        ``"CE"`` (discrete action cross-entropy loss).

        Args:
            batch: Training batch dict with observations, actions, and prompts.
            noise: Optional pre-sampled noise tensor.
            time: Optional pre-sampled flow-matching timesteps.

        Returns:
            Dict with ``"MSE"`` and ``"CE"`` scalar loss tensors.
        """
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        batch["discrete_actions"] = self.normalize_discrete_actions(dict(batch), dataset_index)["actions"]
        batch = self.normalize_targets(batch, dataset_index)

        obs_history_is_pad = batch.get("obs_history_is_pad")
        if obs_history_is_pad is None:
            logging.warning(
                "obs_history_is_pad is missing from the training batch. "
                "Padded observation-history timesteps will not be masked."
            )
        videos, vid_masks = self.prepare_videos(batch, obs_history_is_pad=obs_history_is_pad)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        state = self.prepare_state(batch)
        subgoal_images, subgoal_masks = self.prepare_subgoal_images(batch)
        metadata_tokens, metadata_masks = self.prepare_metadata(batch)
        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(batch)
        actions = batch["actions"]
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            actions_is_pad,
            noise,
            time,
            discrete_actions,
            discrete_action_masks,
            obs_history_is_pad=obs_history_is_pad,
            subgoal_images=subgoal_images,
            subgoal_img_masks=subgoal_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            response_tokens=response_tokens,
            response_masks=response_masks,
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]

        return {"MSE": mse_loss, "CE": ce_loss}

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepares the temporal state tensor, padding or truncating to max_state_dim.

        Args:
            batch: Batch of data containing "state" tensor of shape (B, T, D).

        Returns:
            A tensor of shape (B, T, max_state_dim).
        """
        state = batch["state"]  # (B, T, D) or (B, D) during inference
        if state.ndim == 2:
            if self.config.n_obs_steps > 1:
                raise ValueError(
                    f"Expected 3D state tensor (B, T, D) when n_obs_steps > 1, "
                    f"got shape {state.shape}. Ensure select_action() is being used."
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
        """Tokenize continuous actions into discrete FAST tokens and pad to fixed length.

        Args:
            batch: Batch dict containing ``"discrete_actions"`` (min-max normalized).

        Returns:
            A tuple ``(token_ids, token_masks)`` with shapes
            ``(B, discrete_action_max_length)``.
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

    def prepare_videos(
        self, batch: dict[str, Tensor], obs_history_is_pad: Tensor | None = None
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Apply preprocessing to the video inputs.

        Each camera key now contains a video tensor of shape (B, T, C, H, W).
        Frames are resized to 224x224 with padding. The dataset loader is
        expected to yield raw ``[0, 1]`` floats — the SpaceTime SigLIP encoder
        rescales to ``[-1, 1]`` internally (see
        ``video_encoder.SpaceTimeSiglipVideoEncoder.forward``), so callers
        should NOT pre-apply ImageNet normalization.

        Args:
            batch: Batch of data containing video tensors.
            obs_history_is_pad: Optional bool tensor (B, T) indicating which
                temporal frames are padded. Padded frames are zeroed out before
                encoding so SpaceTimeSiglip does not process clamped/repeated content.

        Returns:
            A tuple of (videos, vid_masks) lists.
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
            vid = batch[key]  # (B, T, C, H, W) or (B, C, H, W) during inference
            if vid.ndim == 4:
                if self.config.n_obs_steps > 1:
                    raise ValueError(
                        f"Expected 5D video tensor (B, T, C, H, W) when n_obs_steps > 1, "
                        f"got shape {vid.shape}. Ensure select_action() is being used."
                    )
                vid = vid.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

            if obs_history_is_pad is not None:
                frame_mask = (~obs_history_is_pad)[:, :, None, None, None]  # (B, T, 1, 1, 1)
                vid = vid * frame_mask

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

        n_empty = min(len(missing_img_keys), self.config.empty_cameras)
        if n_empty > 0:
            assert last_vid is not None and last_mask is not None
            for _ in range(n_empty):
                videos.append(torch.zeros_like(last_vid))
                vid_masks.append(torch.zeros_like(last_mask))

        return videos, vid_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the language prompt into PaliGemma token IDs.

        Wraps each prompt string as ``"Task: {task}<eos>"`` and pads/truncates
        to ``prompt_max_length``.

        Args:
            batch: Batch dict containing ``"prompt"`` (list of strings).

        Returns:
            A tuple ``(lang_tokens, lang_masks)`` with shapes
            ``(B, prompt_max_length)``.
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

    def prepare_response(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the high-level planner subtask response into PaliGemma token IDs.

        Wraps each response string as ``"Subtask: {response}, "`` and
        pads/truncates to ``response_max_length``. Uses ``add_special_tokens=False`` so
        no BOS (or other special tokens) are inserted; the prefix already encodes a
        ``"Subtask: "`` span in :meth:`PI07LowLevelFlowMatching.embed_prefix`.

        Args:
            batch: Batch dict containing ``"response"`` (list of strings).

        Returns:
            A tuple ``(response_tokens, response_masks)`` with shapes
            ``(B, response_max_length)``."""
        device = batch["state"].device
        responses = batch["response"] if "response" in batch else [""] * batch["state"].shape[0]
        response_prompt = [f"Subtask: {response}, " if response != "" else "" for response in responses]
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

    def prepare_subgoal_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess subgoal images for SigLIP embedding.

        Derives subgoal keys from ``config.image_features``: for each
        ``camera{k}`` the corresponding batch key is ``subgoal{k}`` (the
        naming convention used by ``LeRobotDataset._emit_optional_keys``).
        If no ``subgoal{k}`` keys are present, a warning is logged and
        ``([], [])`` is returned; downstream gating in ``embed_prefix``
        short-circuits the subgoal branch when the lists are empty.

        Resizes each subgoal image to 224×224 with aspect-ratio padding and
        converts the pixel range from ``[0, 1]`` to ``[-1, 1]`` as expected
        by SigLIP.  Missing cameras are filled with ``-1`` padding tensors
        up to ``empty_cameras``.

        When ``batch["subgoal_is_pad"]`` is ``True`` for a sample, all
        subgoal slots for that sample are zeroed out and their masks set to
        ``False`` so that downstream attention ignores them.

        Args:
            batch: Batch dict containing subgoal image tensors keyed as
                ``subgoal{k}`` for each ``camera{k}`` in
                ``config.image_features``. If all are absent, see warning +
                fallback above.

        Returns:
            A tuple ``(subgoal_images, subgoal_img_masks)`` of lists.
        """
        subgoal_images = []
        subgoal_img_masks = []

        # Derive subgoal keys from image_features: camera{k} -> subgoal{k}
        subgoal_keys = [key.replace("camera", "subgoal") for key in self.config.image_features]
        present_subgoal_img_keys = [key for key in subgoal_keys if key in batch]
        missing_subgoal_img_keys = [key for key in subgoal_keys if key not in batch]

        if len(present_subgoal_img_keys) == 0:
            logging.getLogger(__name__).warning(
                "All subgoal image features are missing from the batch; using zero tensors with "
                "cleared masks (no subgoal conditioning). "
                f"(batch keys: {list(batch.keys())}) (expected: {subgoal_keys})"
            )

        # Per-sample flag: True means the subgoal was dropped or absent.
        subgoal_is_pad = batch.get(
            "subgoal_is_pad", torch.ones(batch["state"].shape[0], dtype=torch.bool)
        )  # (B,) bool or None

        last_subgoal_img: Tensor | None = None
        last_mask: Tensor | None = None

        for key in present_subgoal_img_keys:
            subgoal_img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                subgoal_img = resize_with_pad(subgoal_img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            subgoal_img = subgoal_img * 2.0 - 1.0

            bsize = subgoal_img.shape[0]
            device = subgoal_img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)

            if subgoal_is_pad is not None:
                is_pad = subgoal_is_pad.to(device=device, dtype=torch.bool)
                mask = mask & ~is_pad
                subgoal_img = subgoal_img * (~is_pad)[:, None, None, None]

            subgoal_images.append(subgoal_img)
            subgoal_img_masks.append(mask)
            last_subgoal_img = subgoal_img
            last_mask = mask

        # Materialize missing-camera placeholders. Filled with -1 (not 0) so
        # they sit at the "black" end of SigLIP's [-1, 1] input range; the
        # accompanying False mask still flags these slots as padded.
        if last_subgoal_img is not None and last_mask is not None:
            for num_empty_cameras in range(len(missing_subgoal_img_keys)):
                if num_empty_cameras >= self.config.empty_cameras:
                    break
                subgoal_images.append(torch.ones_like(last_subgoal_img) * -1)
                subgoal_img_masks.append(torch.zeros_like(last_mask))

        return subgoal_images, subgoal_img_masks

    def prepare_metadata(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize episode metadata into Gemma 3 token IDs.

        Wraps non-empty per-sample metadata segments into a single
        ``"Metadata: {seg1} {seg2} ..."`` string, then pads/truncates to
        ``metadata_max_length``. Samples with no active segments emit an
        empty string.

        Args:
            batch: Batch dict that may contain any of:
                ``"speed"``, ``"quality"``, ``"mistake"``, ``"fps"`` (numeric
                tensors with a corresponding ``_is_pad`` bool tensor —
                entries marked as pad are dropped), and ``"robot_type"``,
                ``"control_mode"`` (lists of strings — empty string is the
                pad signal, no separate ``_is_pad`` flag). Missing keys are
                treated as fully padded.

        Returns:
            A tuple ``(metadata_tokens, metadata_masks)`` with shapes
            ``(B, metadata_max_length)``.
        """

        metadata = []
        batch_size = batch["state"].shape[0]
        # safety conditioning if metadata are not passed by sample actions
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
            batch.get("speed", torch.zeros(batch_size, dtype=torch.float32)),
            batch.get("quality", torch.zeros(batch_size, dtype=torch.float32)),
            batch.get("mistake", torch.zeros(batch_size, dtype=torch.float32)),
            batch.get("speed_is_pad", torch.ones(batch_size, dtype=torch.bool)),
            batch.get("quality_is_pad", torch.ones(batch_size, dtype=torch.bool)),
            batch.get("mistake_is_pad", torch.ones(batch_size, dtype=torch.bool)),
            batch.get("robot_type", [""] * batch_size),
            batch.get("control_mode", [""] * batch_size),
            batch.get("fps", torch.zeros(batch_size, dtype=torch.long)),
            batch.get("fps_is_pad", torch.ones(batch_size, dtype=torch.bool)),
            strict=True,
        ):
            segments = []
            if not speed_is_pad:
                segments.append(f"Speed: {str(speed.item())}, ")

            if not quality_is_pad:
                segments.append(f"Quality: {str(quality.item())}, ")

            if not mistake_is_pad:
                segments.append(f"Mistake: {str(mistake.item())}, ")

            if robot_type:
                segments.append(f"Robot: {robot_type}, ")

            if not fps_is_pad:
                segments.append(f"FPS: {str(fps.item())}, ")

            if control_mode:
                segments.append(f"Control: {control_mode}, ")

            metadata.append(f"Metadata: {' '.join(segments)}" if segments else "")

        device = batch["state"].device
        tokenized_metadata = self.language_tokenizer.__call__(
            metadata,
            padding="max_length",
            padding_side="right",
            max_length=self.config.metadata_max_length,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        )
        metadata_tokens = tokenized_metadata["input_ids"].to(device=device)
        metadata_masks = tokenized_metadata["attention_mask"].to(device=device, dtype=torch.bool)

        return metadata_tokens, metadata_masks


# Flow-matching model
class PI07LowLevelFlowMatching(nn.Module):
    """π07 Low-Level Component: flow-matching action generation with Knowledge Insulation.

    Architecture overview::

        ┌───────────────────────────────────────────────────┐
        │                   actions                         │
        │                   ▲                               │
        │                  ┌┴─────┐                         │
        │      kv cache    │Gemma │  (detached)             │
        │      ┌──────────►│Expert│                         │
        │      │           │      │                         │
        │     ┌┴─────────┐ │x 10  │                         │
        │     │          │ └▲─────┘                         │
        │     │PaliGemma │  │                               │
        │     │  (VLM)   │  noise                           │
        │     └▲──▲──▲──▲──▲──▲──▲──▲                       │
        │      │  │  │  │  │  └── ``Action:`` + discrete (training) │
        │      │  │  │  │  └───── ``";\n "`` + metadata        │
        │      │  │  │  └──────── subgoal images, ``Subgoal:`` │
        │      │  │  └────────── response, commas, state, ``State:`` │
        │      │  └──────────── language                        │
        │      └────────────── video (SpaceTimeSiglip)                 │
        └───────────────────────────────────────────────────┘

    The VLM processes the same prefix layout as :meth:`embed_prefix` (videos, language,
    ``State:``, state, commas, response, ``Subgoal:``, subgoal images, ``";\n "``,
    optional ``Action:``/discrete, metadata). The action expert receives
    the prefix KV-cache (detached for Knowledge Insulation) together with
    noisy continuous actions and flow-matching timestep embeddings to
    predict the velocity field.
    """

    def __init__(self, config: PI07LowLevelConfig, discrete_action_vocab_size: int | None = None):
        super().__init__()
        self.config = config

        self.config.vlm_config.discrete_action_vocab_size = discrete_action_vocab_size
        self.gemma3_with_expert = Gemma3WithExpertModel(self.config.vlm_config)

        expert_hidden = self.gemma3_with_expert.config.gemma_expert_config.hidden_size
        if config.proj_width != expert_hidden:
            raise ValueError(
                f"proj_width ({config.proj_width}) must equal the action expert's "
                f"hidden_size ({expert_hidden}) so suffix embeddings are compatible "
                f"with the expert transformer layers."
            )

        vlm_hidden_size = self.gemma3_with_expert.config.gemma3_config.text_config.hidden_size

        self.video_encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=self.gemma3_with_expert._vision_tower(),
            multi_modal_projector=self.gemma3_with_expert._multi_modal_projector(),
            max_num_frames=config.n_obs_steps,
            spacetime_layer_stride=config.spacetime_layer_stride,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        # Per-timestep state projection: each of the T state vectors becomes one token
        self.state_proj = nn.Linear(self.config.max_state_dim, vlm_hidden_size)

        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

        # Length of the "Action: " indicator tokens — constant for a given
        # tokenizer, so cache once instead of re-tokenizing on every forward
        # call. Used together with discrete_action_max_length to compute the
        # cross-attention prefix slice in forward()/sample_actions(), so the
        # action expert sees the same prefix length at train and inference.
        self._action_indicator_len = len(self.language_tokenizer.encode("Action: ", add_special_tokens=False))

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_video(self, video: Tensor, obs_history_is_pad: Tensor | None = None) -> Tensor:
        """Encode a video through SpaceTimeSiglip + Perceiver reducer + projection.

        Args:
            video: (B, T, C, H, W)
            obs_history_is_pad: Optional ``(B, T)`` bool mask — ``True`` for
                padded history frames. Threaded into the SpaceTime SigLIP
                encoder so temporal attention blocks padded frames (pixel-
                zeroing alone is insufficient — the patch embedding bias and
                temporal PE for ``t < T-1`` are non-zero, so zero pixels
                still produce non-zero hidden states the current frame would
                otherwise attend to).

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
        state: Tensor,
        response_tokens: Tensor,
        response_masks: Tensor,
        metadata_tokens: Tensor,
        metadata_masks: Tensor,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        obs_history_is_pad: Tensor | None = None,
        subgoal_images: list[Tensor] | None = None,
        subgoal_img_masks: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed all prefix modalities and build the 1-D attention pattern.

        Concatenation order -- matches π0.7 paper Fig. 19's "image goals come
        after the text prompt" rule (subgoal block sits at the tail, just
        before the optional discrete-action block):

        ``[videos | language | State: | state(T) | <state-end> |
        response? | metadata? | ";\\n "? |
        Subgoal: subgoal_images... ", "? |
        ("Action:" + discrete_actions only when training)]``

        ``<state-end>`` is ``", "`` when at least one optional middle block
        (response / subgoal / metadata) contributes real tokens, else ``":\\n"``.
        The trailing ``";\\n "`` prefix-end is only emitted in the former case;
        without optional content the state-end already serves as the separator
        before ``"Action: "``, so appending another would dangle spurious tokens.

        Note (batch-wide gate): ``has_any_optional`` is computed by OR-ing the
        masks across the whole batch, so a sample with no optional content
        sharing a batch with samples that do have optionals still receives the
        ``", "`` state-end and ``";\\n "`` prefix-end (with all-False masks on
        the absent optional block). This keeps every sample in the batch
        aligned to the same prefix layout — all-or-nothing per batch — which
        is needed because the prefix length determines per-sample position
        IDs and cross-attention offsets that must be uniform within a batch.
        Acceptable in practice for the homogeneous batches used in training;
        per-sample gating would require variable-length prefixes per sample.

        Attention pattern (via ``att_masks`` cumsums). Paper §VI.B says
        "observation tokens and subgoal image tokens use bidirectional
        attention within themselves ... the following text tokens use causal
        attention":

            - Video patches: one bidirectional block (``[0] * N``).
            - All text spans (language, ``State:``, state-end ``", "`` /
              ``":\\n"``, response, metadata, ``";\\n "``, ``Subgoal:``,
              trailing ``", "`` after subgoals, ``Action:``): one causal
              block per token (``[1] * N``).
            - State tokens (per-history-step linear projections): one
              bidirectional block (``[1] + [0] * (T-1)``).
            - Subgoal image patches: one bidirectional block per camera
              (``[1] + [0] * (N-1)``).
            - Discrete-action FAST tokens (training only): causal ``[1] * N``.

        Args:
            videos: List of video tensors, each ``(B, T, C, H, W)``.
            vid_masks: List of boolean masks, each ``(B,)``.
            lang_tokens: Language token IDs ``(B, prompt_max_length)``.
            lang_masks: Boolean mask for language tokens.
            state: Temporal state ``(B, T, max_state_dim)``.
            response_tokens: Subtask response token IDs
                ``(B, response_max_length)``.
            response_masks: Boolean mask for response tokens.
            metadata_tokens: Metadata token IDs ``(B, metadata_max_length)``.
            metadata_masks: Boolean mask for metadata tokens.
            discrete_actions: Optional FAST token IDs
                ``(B, discrete_action_max_length)``. Provided during training.
            discrete_action_masks: Boolean mask for discrete actions.
            obs_history_is_pad: Optional ``(B, T)`` bool tensor; ``True`` for
                padded timesteps. Used to mask state tokens during training.
            subgoal_images: List of subgoal image tensors ``(B, C, H, W)``.
            subgoal_img_masks: List of boolean masks ``(B,)``.

        Returns:
            A tuple ``(embs, pad_masks, att_masks)`` where:
                - embs: ``(B, total_seq_len, D)``
                - pad_masks: ``(B, total_seq_len)``
                - att_masks: ``(B, total_seq_len)`` for
                  :func:`make_att_2d_masks`.
        """
        embs = []
        pad_masks = []
        att_masks = []
        bsize = lang_tokens.shape[0]

        # Whether any optional middle block (response / subgoal / metadata) will
        # actually contribute real tokens to the prefix. When all are dropped the
        # state-end separator collapses to ":\n" and the trailing prefix-end is
        # omitted, eliminating spurious dangling tokens that would otherwise break
        # the cumsum at the indicator -> first-discrete boundary.
        #
        # Each ``has_*`` flag is OR-reduced across all ranks via
        # ``_global_or_branch_decisions``. Different ranks process different
        # micro-batches, so under stochastic ``*_drop_prob`` settings their
        # local ``.any()`` values can diverge. If we let each rank take its
        # local branch, FSDP / ZeRO-3 would launch different numbers of param
        # all-gathers per rank and deadlock at NCCL (observed empirically:
        # rank 0,1,3,4,5,7 stuck at SeqNum=274 ALLREDUCE, rank 2,6 at
        # SeqNum=279 ALLGATHER_BASE). Synchronizing the *branch decision* —
        # not the data — keeps every rank running the same set of collectives.
        # Ranks whose local data is all-dropped still embed real tokens with
        # all-False masks; downstream attention / loss respect the mask, so
        # accuracy is preserved.
        #
        # The same call also asserts cross-rank agreement on field *presence*
        # (None vs Tensor), bundled into the same all-reduce. The inner gates
        # (``if has_response and response_tokens is not None``) assume field
        # presence is a global property of the dataset config. If a future
        # heterogeneous-mixture change ever makes that not true, the helper
        # raises a loud ``RuntimeError`` here rather than letting the branches
        # silently re-desync and hang at NCCL hours into a run.
        device = lang_tokens.device
        response_present_local = response_tokens is not None and response_masks is not None
        subgoal_present_local = bool(subgoal_images) and bool(subgoal_img_masks)
        metadata_present_local = metadata_tokens is not None and metadata_masks is not None
        has_response_local = response_present_local and bool(response_masks.any())
        has_subgoal_local = subgoal_present_local and any(bool(m.any()) for m in subgoal_img_masks)
        has_metadata_local = metadata_present_local and bool(metadata_masks.any())
        has_response, has_subgoal, has_metadata = _global_or_branch_decisions(
            presence_locals=(response_present_local, subgoal_present_local, metadata_present_local),
            any_locals=(has_response_local, has_subgoal_local, has_metadata_local),
            field_names=("response", "subgoal", "metadata"),
            device=device,
        )
        has_any_optional = bool(has_response or has_subgoal or has_metadata)

        for vid, vid_mask in zip(videos, vid_masks, strict=True):
            vid_emb = self.embed_video(vid, obs_history_is_pad=obs_history_is_pad)
            vid_emb = vid_emb.to(dtype=_preferred_dtype())

            num_vid_embs = vid_emb.shape[1]
            vid_mask_expanded = vid_mask[:, None].expand(bsize, num_vid_embs)

            embs.append(vid_emb)
            pad_masks.append(vid_mask_expanded)

            att_masks += [0] * num_vid_embs

        # Gemma 3's `embed_tokens` is a `Gemma3TextScaledWordEmbedding` that
        # already multiplies by sqrt(hidden_size) internally — do NOT scale
        # again here (unlike pi05, whose PaliGemma Gemma-v1 embedding is a
        # plain nn.Embedding with the normalizer applied later in the stock
        # forward that we bypass).  Applies to every `embed_language_tokens`
        # call in this method.
        lang_emb = self.gemma3_with_expert.embed_language_tokens(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Per π0.7 paper §VI.B: "The following text tokens use causal attention".
        # Language tokens open one causal block per token after the bidirectional
        # video prefix.
        num_lang_embs = lang_emb.shape[1]
        att_masks += [1] * num_lang_embs

        state_start_indicator_ids = self.language_tokenizer.encode("State: ", add_special_tokens=False)
        state_start_tokens = torch.tensor(
            [state_start_indicator_ids] * bsize,
            device=lang_tokens.device,
            dtype=torch.long,
        )
        state_start_emb = self.gemma3_with_expert.embed_language_tokens(state_start_tokens)

        num_state_start_embs = state_start_emb.shape[1]
        state_start_mask = torch.ones(
            bsize, num_state_start_embs, dtype=torch.bool, device=lang_tokens.device
        )

        embs.append(state_start_emb)
        pad_masks.append(state_start_mask)
        # "State: " indicator is text → causal per paper §VI.B.
        att_masks += [1] * num_state_start_embs

        # Project each timestep's state into a separate VLM token
        # state: (B, T, max_state_dim) -> state_emb: (B, T, vlm_hidden_size)
        state_emb = self.state_proj(state.to(dtype=_preferred_dtype()))
        num_state_tokens = state_emb.shape[1]  # T
        if obs_history_is_pad is not None:
            state_mask = ~obs_history_is_pad  # (B, T) — `~` allocates a fresh tensor
        else:
            # Absent → assume all history is padded; only current step is real.
            state_mask = torch.zeros(bsize, num_state_tokens, dtype=torch.bool, device=state.device)
        # Current step (t = T-1) is ALWAYS real even when the dataset's
        # history_state_drop_prob augmentation flips obs_history_is_pad to
        # all-True. Without this override the policy would condition on no
        # state at all, since attention to the current state token would be
        # masked out — defeating the purpose of preserving the current frame.
        # Both branches above produce fresh tensors (`~` allocates;
        # `torch.zeros` allocates), so the `[:, -1] = True` write below does
        # not reach the caller's `obs_history_is_pad`.
        state_mask[:, -1] = True

        embs.append(state_emb)
        pad_masks.append(state_mask)
        # State tokens form their own bidirectional block (one per history step).
        # The paper §VI.B describes state as a linear-projection embedding that is
        # treated as an individual token, but does not classify it as text-causal,
        # so we keep the bidirectional-among-state behaviour while opening a fresh
        # block so it does not bleed into the now-causal "State: " indicator.
        att_masks += [1] + [0] * (num_state_tokens - 1)

        # When optional middle blocks follow, use ", " as a state -> optional separator;
        # otherwise collapse to ":\n" so the trailing prefix-end can be omitted.
        state_end_str = ", " if has_any_optional else ":\n"
        state_end_indicator_ids = self.language_tokenizer.encode(state_end_str, add_special_tokens=False)
        state_end_tokens = torch.tensor(
            [state_end_indicator_ids] * bsize,
            device=lang_tokens.device,
            dtype=torch.long,
        )
        state_end_emb = self.gemma3_with_expert.embed_language_tokens(state_end_tokens)

        num_state_end_embs = state_end_emb.shape[1]
        state_end_mask = torch.ones(bsize, num_state_end_embs, dtype=torch.bool, device=lang_tokens.device)

        embs.append(state_end_emb)
        pad_masks.append(state_end_mask)
        # state-end separator (", " or ":\n") is text → causal per paper §VI.B.
        att_masks += [1] * num_state_end_embs

        # Only embed response when at least one sample in *any rank's* batch has
        # real response tokens. The decision is taken on the globally-synced
        # ``has_response`` flag (computed above) — using a local ``.any()``
        # check here would let different ranks diverge into different code
        # paths and deadlock under FSDP / ZeRO-3.
        if has_response and response_tokens is not None and response_masks is not None:
            response_emb = self.gemma3_with_expert.embed_language_tokens(response_tokens)
            embs.append(response_emb)
            pad_masks.append(response_masks)
            # Response (Subtask) is text → fully causal per paper §VI.B (was
            # prefix-LM `[1] + [0]*(N-1)`, leaking future-token information into
            # the response loss).
            num_response_embs = response_emb.shape[1]
            att_masks += [1] * num_response_embs

        # Same global-synced gate for metadata.
        if has_metadata and metadata_tokens is not None and metadata_masks is not None:
            metadata_emb = self.gemma3_with_expert.embed_language_tokens(metadata_tokens)
            embs.append(metadata_emb)
            pad_masks.append(metadata_masks)
            # Metadata is text → causal per paper §VI.B.
            att_masks += [1] * metadata_emb.shape[1]

        # Only emit the ";\n " prefix-end when at least one optional middle block was added
        # above. With no optional content, the state-end already collapsed to ":\n" and acts
        # as the separator before "Action: " — appending another would dangle spurious tokens.
        if has_any_optional:
            prefix_end_indicator_ids = self.language_tokenizer.encode(";\n ", add_special_tokens=False)
            prefix_end_tokens = torch.tensor(
                [prefix_end_indicator_ids] * bsize,
                device=lang_tokens.device,
                dtype=torch.long,
            )
            prefix_end_emb = self.gemma3_with_expert.embed_language_tokens(prefix_end_tokens)

            num_prefix_end_embs = prefix_end_emb.shape[1]
            prefix_end_mask = torch.ones(
                bsize, num_prefix_end_embs, dtype=torch.bool, device=lang_tokens.device
            )

            embs.append(prefix_end_emb)
            pad_masks.append(prefix_end_mask)
            # ";\n " separator is text → causal per paper §VI.B.
            att_masks += [1] * num_prefix_end_embs

        # Per π0.7 paper Fig. 19 caption: "When image goals are present, we include
        # them as an additional block-causal bidirectional block, **after the text
        # prompt**." So the subgoal block sits at the tail of the prefix, after all
        # text (lang / state / response / metadata / ";\n ") and before the
        # training-only discrete-action block.
        #
        # Same global-synced gate for the subgoal block. ``has_subgoal`` is
        # globally OR-reduced above, so all ranks take this branch together
        # whenever any rank's micro-batch has at least one real subgoal image.
        # Skipping the local-``.any()`` branch and falling through to
        # all-pad data on a rank that locally has none would re-introduce
        # the FSDP NCCL desync.
        if has_subgoal and subgoal_images:
            # Per-sample availability: True iff at least one camera slot
            # has a real subgoal image for that sample. In a mixed batch
            # (some samples have subgoals, others don't), the "Subgoal: "
            # header and trailing ", " footer must follow this same mask —
            # otherwise pad-only samples would receive real (unmasked)
            # indicator tokens with no image content behind them, which
            # the model would then attend to as if they were grounded.
            sample_has_subgoal = torch.stack([m.to(dtype=torch.bool) for m in subgoal_img_masks], dim=0).any(
                dim=0
            )

            subgoal_img_start_indicator_ids = self.language_tokenizer.encode(
                "Subgoal: ", add_special_tokens=False
            )
            subgoal_img_start_tokens = torch.tensor(
                [subgoal_img_start_indicator_ids] * bsize,
                device=lang_tokens.device,
                dtype=torch.long,
            )
            subgoal_img_start_emb = self.gemma3_with_expert.embed_language_tokens(subgoal_img_start_tokens)

            num_subgoal_img_start_embs = subgoal_img_start_emb.shape[1]
            subgoal_img_start_mask = sample_has_subgoal[:, None].expand(bsize, num_subgoal_img_start_embs)

            embs.append(subgoal_img_start_emb)
            pad_masks.append(subgoal_img_start_mask)
            # "Subgoal: " indicator is text → causal per paper §VI.B.
            att_masks += [1] * num_subgoal_img_start_embs

            for (
                subgoal_img,
                subgoal_img_mask,
            ) in zip(subgoal_images, subgoal_img_masks, strict=True):
                subgoal_img_emb = self.gemma3_with_expert.embed_image(subgoal_img)
                subgoal_img_emb = subgoal_img_emb.to(dtype=_preferred_dtype())

                # Gemma 3's projector does not apply the `/ sqrt(text_hidden_size)`
                # scaling that stock PaliGemma does, so no un-normalization is
                # required here (matches `embed_image` in `gemma3_with_expert.py`).

                bsize, num_subgoal_img_embs = subgoal_img_emb.shape[:2]
                subgoal_img_mask = subgoal_img_mask[:, None].expand(bsize, num_subgoal_img_embs)

                embs.append(subgoal_img_emb)
                pad_masks.append(subgoal_img_mask)

                # Subgoal images: bidirectional within themselves, can attend everything
                # earlier in the prefix (paper §VI.B). One bidirectional block per camera.
                att_masks += [1] + [0] * (num_subgoal_img_embs - 1)

            subgoal_img_end_indicator_ids = self.language_tokenizer.encode(", ", add_special_tokens=False)
            subgoal_img_end_tokens = torch.tensor(
                [subgoal_img_end_indicator_ids] * bsize,
                device=lang_tokens.device,
                dtype=torch.long,
            )
            subgoal_img_end_emb = self.gemma3_with_expert.embed_language_tokens(subgoal_img_end_tokens)

            num_subgoal_img_end_embs = subgoal_img_end_emb.shape[1]
            subgoal_img_end_mask = sample_has_subgoal[:, None].expand(bsize, num_subgoal_img_end_embs)

            embs.append(subgoal_img_end_emb)
            pad_masks.append(subgoal_img_end_mask)
            # ", " separator after subgoal images is text → causal per paper §VI.B.
            att_masks += [1] * num_subgoal_img_end_embs

        if discrete_actions is not None:
            discrete_action_start_indicator_ids = self.language_tokenizer.encode(
                "Action: ", add_special_tokens=False
            )
            discrete_action_start_tokens = torch.tensor(
                [discrete_action_start_indicator_ids] * bsize,
                device=lang_tokens.device,
                dtype=torch.long,
            )
            discrete_action_start_emb = self.gemma3_with_expert.embed_language_tokens(
                discrete_action_start_tokens
            )

            num_discrete_action_start_embs = discrete_action_start_emb.shape[1]
            discrete_action_start_mask = torch.ones(
                bsize, num_discrete_action_start_embs, dtype=torch.bool, device=lang_tokens.device
            )

            embs.append(discrete_action_start_emb)
            pad_masks.append(discrete_action_start_mask)
            # Each "Action: " indicator token is its own causal block. Using
            # [1] + [0]*(N-1) collapses them into a single bidirectional block, which
            # shifts the cumsum at the indicator -> first-discrete boundary by N-1
            # and breaks the discrete-action CE loss.
            att_masks += [1] * num_discrete_action_start_embs

            discrete_action_emb = self.gemma3_with_expert.embed_discrete_actions(discrete_actions)
            embs.append(discrete_action_emb.to(dtype=_preferred_dtype()))
            pad_masks.append(discrete_action_masks)
            att_masks += [1] * discrete_action_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy actions and flow-matching timestep for the action expert.

        Projects actions through ``action_in_proj`` and computes an
        adaRMS-style conditioning vector from sinusoidal timestep embeddings
        via a two-layer MLP.  The suffix forms a single bidirectional block
        (``att_masks = [1, 0, …, 0]``).

        Args:
            noisy_actions: ``(B, chunk_size, max_action_dim)`` noisy action
                tensor at the current flow-matching timestep.
            timestep: ``(B, chunk_size)`` per-step timestep values.

        Returns:
            A tuple ``(embs, pad_masks, att_masks, adarms_cond)`` where
            ``adarms_cond`` is the conditioning vector for adaptive RMSNorm
            in the Gemma expert layers.
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = noisy_actions.shape[0]
        dtype = _preferred_dtype()
        device = noisy_actions.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )

        noisy_actions = noisy_actions.to(dtype=dtype)
        action_emb = self.action_in_proj(noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = time_emb.to(dtype=dtype)
        adarms_cond = time_mlp_func(time_emb)

        embs.append(action_emb)

        bsize, action_dim = action_emb.shape[:2]
        action_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_mask)

        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        obs_history_is_pad: Tensor | None = None,
        subgoal_images: list[Tensor] | None = None,
        subgoal_img_masks: list[Tensor] | None = None,
        metadata_tokens: Tensor | None = None,
        metadata_masks: Tensor | None = None,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass: embed all modalities and compute losses.

        Runs the VLM on the prefix (video, language, response, state, subgoal
        images, metadata, discrete actions), then the action expert on the
        noisy action suffix.  Returns both the flow-matching MSE loss and the
        discrete-action cross-entropy loss.

        The VLM's KV-cache is detached before being passed to the action
        expert (Knowledge Insulation).

        Args:
            videos: List of video tensors ``(B, T, C, H, W)``.
            vid_masks: List of boolean masks ``(B,)``.
            lang_tokens: Language token IDs ``(B, prompt_max_length)``.
            lang_masks: Boolean mask for language tokens.
            state: Temporal state ``(B, T, max_state_dim)``.
            actions: Ground-truth continuous actions ``(B, chunk_size, max_action_dim)``.
            actions_is_pad: Optional ``(B, chunk_size)`` bool mask for padded actions.
            noise: Optional pre-sampled noise.
            time: Optional pre-sampled flow-matching timesteps.
            discrete_actions: Optional FAST token IDs.
            discrete_action_masks: Optional mask for discrete actions.
            obs_history_is_pad: Optional ``(B, T)`` mask for padded frames.
            subgoal_images: Optional list of subgoal image tensors.
            subgoal_img_masks: Optional list of masks.
            metadata_tokens: Optional metadata token IDs.
            metadata_masks: Optional mask for metadata tokens.
            response_tokens: Optional subtask response token IDs.
            response_masks: Optional mask for response tokens.

        Returns:
            Dict with ``"MSE"`` (flow-matching loss) and ``"CE"``
            (discrete action loss) scalar tensors.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            state,
            response_tokens,
            response_masks,
            metadata_tokens,
            metadata_masks,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
            obs_history_is_pad=obs_history_is_pad,
            subgoal_images=subgoal_images,
            subgoal_img_masks=subgoal_img_masks,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Exclude both the "Action: " indicator tokens and the discrete action tokens from
        # cross-attention so the action expert sees the same prefix length at train and inference
        # (at inference, neither is in the prefix). Matches pi05's discrete_action_indicator_max_length logic.
        num_cross_att_tokens = (
            prefix_embs.shape[1] - self._action_indicator_len - self.config.discrete_action_max_length
        )

        (prefix_out, _), past_key_values = self.gemma3_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        batch_size = actions.shape[0]
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(batch_size, actions.device)

        delay = torch.randint(0, self.config.max_delay + 1, (batch_size,))
        prefix_mask = rearrange(torch.arange(self.config.chunk_size), "c -> 1 c") < rearrange(
            delay, "b -> b 1"
        )
        prefix_mask = prefix_mask.to(device=actions.device)
        time = torch.where(prefix_mask, 0, rearrange(time, "b -> b 1"))

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
        prefix_offsets = torch.sum(
            prefix_pad_masks[:, : -(self._action_indicator_len + self.config.discrete_action_max_length)],
            dim=-1,
        )[:, None]
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        assert past_key_values is not None
        kv_cache: dict = past_key_values
        for layer_idx in kv_cache:
            kv_cache[layer_idx]["key_states"] = kv_cache[layer_idx]["key_states"].detach()
            kv_cache[layer_idx]["value_states"] = kv_cache[layer_idx]["value_states"].detach()

        (_, suffix_out), _ = self.gemma3_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=kv_cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        assert suffix_out is not None
        # Supervise the whole chunk the model was trained to predict. n_action_steps
        # is the inference-time execution horizon only and must not truncate the
        # training target (chunk_size is the prediction horizon).
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        mse_loss = F.mse_loss(u_t, v_t, reduction="none")

        postfix_mask = rearrange(torch.logical_not(prefix_mask), "b c -> b c 1")

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            in_episode_bound = rearrange(in_episode_bound, "b c -> b c 1")
            postfix_mask = torch.logical_and(postfix_mask, in_episode_bound)

        mse_loss = mse_loss * postfix_mask

        mse_loss = mse_loss[:, :, : self.config.max_action_dim]

        postfix_mask_expanded = repeat(postfix_mask, "b c 1 -> b c d", d=mse_loss.shape[-1])
        mse_loss = mse_loss.sum() / (postfix_mask_expanded.sum() + 1e-8)

        assert discrete_actions is not None
        assert discrete_action_masks is not None
        assert prefix_out is not None
        batch_size, seq_len = discrete_actions.shape
        discrete_token_start = -self.config.discrete_action_max_length
        discrete_action_slice_object = slice(discrete_token_start - 1, -1)
        discrete_action_out = prefix_out[:, discrete_action_slice_object]
        logits = self.gemma3_with_expert.da_head(discrete_action_out)

        logits = logits.to(dtype=torch.float32)
        logits = rearrange(logits, "b s d -> (b s) d")
        labels = rearrange(discrete_actions, "b s -> (b s)")
        discrete_action_ce_loss = F.cross_entropy(logits, labels, reduction="none")

        discrete_action_ce_loss = rearrange(discrete_action_ce_loss, "(b s) -> b s", b=batch_size, s=seq_len)

        discrete_action_is_pad = ~discrete_action_masks
        discrete_action_ce_loss = discrete_action_ce_loss * ~discrete_action_is_pad

        discrete_action_ce_loss = discrete_action_ce_loss.mean()

        return {"MSE": mse_loss, "CE": discrete_action_ce_loss}

    def sample_actions(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        action_prefix: Tensor,
        delay: Tensor,
        noise: Tensor | None = None,
        subgoal_images: list[Tensor] | None = None,
        subgoal_img_masks: list[Tensor] | None = None,
        metadata_tokens: Tensor | None = None,
        metadata_masks: Tensor | None = None,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        obs_history_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Inference: iteratively denoise to produce a continuous action chunk.

        Embeds the prefix (without discrete actions), caches the VLM KV
        states, then runs ``num_steps`` denoising iterations through the
        action expert. Prefix action steps (up to ``delay``) are held fixed
        from ``action_prefix``.

        Args:
            videos: List of video tensors ``(B, T, C, H, W)``.
            vid_masks: List of boolean masks ``(B,)``.
            lang_tokens: Language token IDs ``(B, prompt_max_length)``.
            lang_masks: Boolean mask for language tokens.
            state: Temporal state ``(B, T, max_state_dim)``.
            action_prefix: Previously committed actions
                ``(B, chunk_size, max_action_dim)`` (zero-padded beyond delay).
            delay: Scalar tensor indicating how many prefix steps are fixed.
            noise: Optional pre-sampled noise.
            subgoal_images: Optional list of subgoal image tensors.
            subgoal_img_masks: Optional list of masks.
            metadata_tokens: Optional metadata token IDs.
            metadata_masks: Optional mask for metadata tokens.
            response_tokens: Optional subtask response token IDs.
            response_masks: Optional mask for response tokens.
            obs_history_is_pad: Optional ``(B, T)`` bool mask flagging padded
                history slots (``True`` = padded). Emitted by
                ``PI07LowLevelPolicy._build_history_batch`` so the encoder can
                use real mid-episode history while still masking out the
                start-of-episode zero-fill. ``None`` falls back to "all
                history padded except current" via ``embed_prefix`` and the
                encoder's None-fallback.

        Returns:
            Denoised action chunk ``(B, chunk_size, max_action_dim)``.
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
            state,
            response_tokens,
            response_masks,
            metadata_tokens,
            metadata_masks,
            subgoal_images=subgoal_images,
            subgoal_img_masks=subgoal_img_masks,
            obs_history_is_pad=obs_history_is_pad,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1]

        (prefix_out, _), past_kv = self.gemma3_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )
        past_key_values: list[dict[str, Tensor]] = past_kv

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = rearrange(torch.arange(self.config.chunk_size, device=device), "c -> 1 c") < delay
        while time >= -dt / 2:
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
            masked_time = torch.where(prefix_mask, 0, time)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                masked_time,
            )

            x_t += dt * v_t
            time += dt

        x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Run one action-expert forward pass to predict the velocity field.

        Embeds the suffix (noisy actions + timestep), constructs the
        cross-attention mask to the cached prefix, and runs the Gemma
        expert to produce the predicted velocity ``v_t``.

        Args:
            prefix_pad_masks: ``(B, prefix_len)`` padding mask from the
                prefix pass (used for cross-attention masking).
            past_key_values: Cached KV states from the VLM prefix pass.
            x_t: Current noisy actions ``(B, chunk_size, max_action_dim)``.
            time: Per-step timestep ``(B, chunk_size)``.

        Returns:
            Predicted velocity ``(B, n_action_steps, max_action_dim)``.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.gemma3_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        assert suffix_out is not None
        # Denoise the full chunk_size chunk so v_t matches x_t in the Euler step.
        # n_action_steps (execution horizon) is applied later in select_action, not
        # at decode time.
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
