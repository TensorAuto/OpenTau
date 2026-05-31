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

"""π05 Mem: A Vision-Language-Action Flow Model with space-time SigLIP video
encoding and temporal state sequences.

Based on π05, this variant implements the low-level memory architecture from
Torne, Pertsch, Walke et al. "MEM: Multi-Scale Embodied Memory for Vision
Language Action Models" (Section III-C + Appendix C):

  1. Extends the PaliGemma SigLIP image encoder with space-time separable
     attention every ``spacetime_layer_stride``-th ViT layer. The temporal
     sublayer re-uses each layer's existing Q/K/V/O projections — no new
     learnable parameters are introduced. Past-timestep tokens are dropped
     after the encoder so the prefix matches a single-frame VLA's 256 image
     tokens exactly.
  2. Accepts temporal state sequences (B, T, D) and projects each timestep
     into a separate continuous token for the Gemma backbone.
"""

import builtins
import logging
import math
from collections import deque
from pathlib import Path

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
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.policies.utils import flow_matching_masked_mse
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
class PI05MemPolicy(PreTrainedPolicy):
    """Wrapper class around PI05MemFlowMatching model.

    Uses a space-time SigLIP video encoder (MEM paper low-level memory) and
    temporal state sequences projected into per-timestep continuous tokens in
    the VLM embedding space.
    """

    config_class = PI05MemConfig
    name = "pi05_mem"

    def __init__(
        self,
        config: PI05MemConfig,
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

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            config.discrete_action_tokenizer_path, trust_remote_code=True
        )
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI05MemFlowMatching(config, discrete_action_vocab_size=discrete_action_vocab_size)

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
            new_key = key

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
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
        raise NotImplementedError("Currently not implemented for PI05 Mem")

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
        """Select a single action given environment observations."""
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
        """Sample actions from the policy given environment observations."""
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
        state = self.prepare_state(batch)

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
        """Do a full training forward pass to compute the loss."""
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
        state = self.prepare_state(batch)
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
            real_action_dim=batch.get("real_action_dim"),
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
        """Prepares discrete actions for the model by tokenizing and padding them."""
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
        Frames are resized to 224x224 with padding. Pixel values remain in the
        ``[0, 1]`` range as produced by the dataset loader; the video encoder
        rescales to ``[-1, 1]`` (SigLIP's expected range) inside its own
        forward pass.

        Args:
            batch: Batch of data containing video tensors.
            obs_history_is_pad: Optional bool tensor (B, T) indicating which
                temporal frames are padded. Padded frames are zeroed out before
                encoding so the video encoder does not see clamped/repeated
                content.

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
        """Tokenize the text input."""
        device = batch["state"].device
        tasks = batch["prompt"]

        prompt = [f"Task: {task}<eos>Actions:" for task in tasks]

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


# Flow-matching model
class PI05MemFlowMatching(nn.Module):
    """π05 Mem: A Vision-Language-Action Flow Model with space-time SigLIP
    video encoding and temporal state sequences.

    ┌──────────────────────────────────────────┐
    │                   actions                │
    │                   ▲                      │
    │                  ┌┴─────┐                │
    │      kv cache    │Gemma │                │
    │      ┌──────────►│Expert│                │
    │      │           │      │                │
    │     ┌┴─────────┐ │x 10  │                │
    │     │          │ └▲─────┘                │
    │     │PaliGemma │  │                      │
    │     │          │  noise                  │
    │     └▲──▲──▲──▲                          │
    │      │  │  │  └── discrete actions       │
    │      │  │  └───── state (T tokens)       │
    │      │  └──────── language tokens        │
    │      └─────────── video (SigLIP+ST)      │
    └──────────────────────────────────────────┘
    """

    def __init__(self, config: PI05MemConfig, discrete_action_vocab_size: int | None = None):
        super().__init__()
        self.config = config

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        vlm_hidden_size = self.paligemma_with_expert.config.paligemma_config.text_config.hidden_size

        # Space-time SigLIP video encoder (MEM paper low-level memory).
        # The encoder is a thin computational wrapper: it holds
        # ``paligemma_with_expert``'s ``vision_tower`` / ``multi_modal_projector``
        # by reference (no parameter duplication, no separate HF download) and
        # mutates a few encoder layers in place to add temporal self-attention.
        # Freezing and dtype-casting of these modules are already handled by
        # ``PaliGemmaWithExpertModel``. The encoder introduces no new learnable
        # parameters, so a regular pi05 checkpoint's state_dict loads directly.
        self.video_encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=self.paligemma_with_expert.paligemma.vision_tower,
            multi_modal_projector=self.paligemma_with_expert.paligemma.multi_modal_projector,
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

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_video(self, video: Tensor, obs_history_is_pad: Tensor | None = None) -> Tensor:
        """Encode a video through the space-time SigLIP video encoder.

        The encoder applies standard SigLIP spatial attention on every layer
        plus a causal temporal attention sublayer every
        ``spacetime_layer_stride``-th layer. Past-timestep tokens are dropped;
        only the current frame's 256 tokens are returned.

        Args:
            video: (B, T, C, H, W) pixel values in [0, 1].
            obs_history_is_pad: Optional ``(B, T)`` bool mask — ``True`` for
                padded history frames. Threaded into the SpaceTime SigLIP
                encoder so temporal attention blocks padded frames (pixel-
                zeroing alone is insufficient — the patch embedding bias and
                temporal PE for ``t < T-1`` are non-zero, so zero pixels
                still produce non-zero hidden states the current frame would
                otherwise attend to).

        Returns:
            (B, num_video_tokens, vlm_hidden_size) current-frame tokens.
        """
        return self.video_encoder(video, obs_history_is_pad=obs_history_is_pad)

    def embed_prefix(
        self,
        videos: list[Tensor],
        vid_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        obs_history_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed videos with the space-time SigLIP video encoder, language
        tokens with the embedding layer, and temporal state via per-timestep
        learned projection.

        Args:
            videos: List of video tensors, each (B, T, C, H, W).
            vid_masks: List of video mask tensors, each (B,).
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            state: Temporal state tensor of shape (B, T, max_state_dim).
            discrete_actions: Optional discrete action tensor.
            discrete_action_masks: Optional discrete action mask tensor.
            obs_history_is_pad: Optional bool tensor (B, T) from the dataloader.
                True for padded (clamped) timesteps, False for real ones.
                Used to mask state tokens during training; None during inference.

        Returns:
            (embs, pad_masks, att_masks) tuple.
        """
        embs = []
        pad_masks = []
        att_masks = []
        bsize = lang_tokens.shape[0]

        for vid, vid_mask in zip(videos, vid_masks, strict=False):
            vid_emb = self.embed_video(vid, obs_history_is_pad=obs_history_is_pad)
            vid_emb = vid_emb.to(dtype=_preferred_dtype())

            num_vid_embs = vid_emb.shape[1]
            vid_mask_expanded = vid_mask[:, None].expand(bsize, num_vid_embs)

            embs.append(vid_emb)
            pad_masks.append(vid_mask_expanded)

            att_masks += [0] * num_vid_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Build the state pad mask first so masked (dropped / historical) steps
        # can be zeroed *after* normalization but *before* projection.
        # state: (B, T, max_state_dim); num_state_tokens == T.
        num_state_tokens = state.shape[1]  # T
        if obs_history_is_pad is not None:
            state_mask = ~obs_history_is_pad  # (B, T) — `~` allocates a fresh tensor
        else:
            # Absent → assume all history is padded; only current step is real.
            state_mask = torch.zeros(bsize, num_state_tokens, dtype=torch.bool, device=state.device)
        # Current step (t = T-1) is ALWAYS real even when the dataset's
        # history_state_drop_prob augmentation marks obs_history_is_pad all-True.
        # Without this override the policy would condition on no state at all,
        # since attention to the current state token would be masked out —
        # defeating the purpose of preserving the current frame. Both branches
        # above produce fresh tensors (`~` allocates; `torch.zeros` allocates),
        # so the `[:, -1] = True` write below does not reach the caller's
        # `obs_history_is_pad`.
        state_mask[:, -1] = True

        # Defense-in-depth: zero the (already-normalized) state at masked steps so
        # no historical proprioception leaks even if the attention mask later
        # regresses. The current step is preserved by `state_mask[:, -1] = True`.
        # This runs AFTER normalize_inputs and BEFORE state_proj, so a masked slot
        # becomes a clean post-norm zero — never the ill `-mean/std` that zeroing a
        # *raw* state before normalization would produce.
        state = state.masked_fill(rearrange(~state_mask, "b t -> b t 1"), 0.0)

        # Project each timestep's state into a separate VLM token
        # state: (B, T, max_state_dim) -> state_emb: (B, T, vlm_hidden_size)
        state_emb = self.state_proj(state.to(dtype=_preferred_dtype()))

        embs.append(state_emb)
        pad_masks.append(state_mask)
        att_masks += [0] * num_state_tokens  # full attention with video and language

        if discrete_actions is not None:
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
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
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
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
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
        real_action_dim: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute the loss."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            videos,
            vid_masks,
            lang_tokens,
            lang_masks,
            state,
            discrete_actions,
            discrete_action_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1] - self.config.discrete_action_max_length

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
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
        prefix_offsets = torch.sum(prefix_pad_masks[:, : -self.config.discrete_action_max_length], dim=-1)[
            :, None
        ]
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        assert past_key_values is not None
        kv_cache: dict = past_key_values
        for layer_idx in kv_cache:
            kv_cache[layer_idx]["key_states"] = kv_cache[layer_idx]["key_states"].detach()
            kv_cache[layer_idx]["value_states"] = kv_cache[layer_idx]["value_states"].detach()

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
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

        # Shared masked-MSE reduction; see pi05 for the rationale.
        mse_loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            max_action_dim=self.config.max_action_dim,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            real_action_dim=real_action_dim,
        )

        assert discrete_actions is not None
        assert discrete_action_masks is not None
        assert prefix_out is not None
        batch_size, seq_len = discrete_actions.shape
        discrete_token_start = -self.config.discrete_action_max_length
        discrete_action_slice_object = slice(discrete_token_start - 1, -1)
        discrete_action_out = prefix_out[:, discrete_action_slice_object]
        logits = self.paligemma_with_expert.da_head(discrete_action_out)

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
        obs_history_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action.

        Args:
            obs_history_is_pad: Optional ``(B, T)`` bool mask flagging padded
                history slots (``True`` = padded). Emitted by
                ``PI05MemPolicy._build_history_batch`` so the encoder can use
                real mid-episode history while still masking out the
                start-of-episode zero-fill. ``None`` falls back to "all
                history padded except current" via ``embed_prefix`` and the
                encoder's None-fallback.
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
            obs_history_is_pad=obs_history_is_pad,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1]

        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
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
        """Apply one denoising step."""
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
        assert suffix_out is not None
        # Denoise the full chunk_size chunk so v_t matches x_t in the Euler step.
        # n_action_steps (execution horizon) is applied later in select_action, not
        # at decode time.
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
