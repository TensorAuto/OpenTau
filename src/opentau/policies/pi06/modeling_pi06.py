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

"""π06: a Vision-Language-Action model built on Gemma 3 4B.

Relative to π05 this policy swaps PaliGemma-3B for Gemma 3 4B (34 interleaved
sliding-window/global layers, SigLIP at 448×448), enlarges the action expert
to ~860M parameters so it matches the backbone depth, and halves the default
flow-matching denoising schedule to 5 steps.

References:
    - π0.6 Model Card, Physical Intelligence, 2025-11-17.
    - π*0.6: a VLA That Learns From Experience, arXiv:2511.14759.
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
from opentau.datasets.grounding.tokenizer_utils import ensure_loc_tokens
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.normalize import resolve_num_datasets as _num_datasets
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi06.gemma3_with_expert import (
    Gemma3WithExpertConfig,
    Gemma3WithExpertModel,
)
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.utils import get_safe_dtype

# Utility helpers — straight copies of the pi05 versions, documented here for
# locality. If the pi05 file ever evolves, we consciously choose to keep the
# pi06 version frozen at the shape the π0.6 paper assumes.


def _preferred_dtype():
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: torch.device | str = "cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: A 2-D tensor of shape `(B, action_chunk_length)`.
        dimension: The dimension of the embedding vectors. Must be divisible by 2.
        min_period: Minimum period of the sinusoidal functions.
        max_period: Maximum period of the sinusoidal functions.
        device: The device to create the tensors on.

    Returns:
        Tensor of shape `(B, action_chunk_length, dimension)` with the embedding.
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

    Tokens can attend to valid tokens whose cumulative `att_masks` is smaller
    or equal to theirs. Block semantics match π0.5 exactly — see the pi05
    docstring for the full table of examples, condensed:

        [[1 1 1 1 1 1]]: pure causal
        [[0 0 0 1 1 1]]: prefix-LM (first 3 bidirectional, last 3 causal)
        [[1 0 1 0 1 0 0 1 0 0]]: multi-block causal

    Args:
        pad_masks: bool `(B, N)` — True for real tokens.
        att_masks: int32 `(B, N)` — 1 starts a new block, 0 continues.
        n_cross_att_tokens: If set, prepend a cross-attention mask of that
            width so suffix tokens can attend back into the cached prefix.
        cross_att_pad_masks: Prefix pad mask used when building the cross
            attention block.

    Returns:
        Boolean 2-D attention mask of shape `(B, N, N)` or
        `(B, N, N + n_cross_att_tokens)` when cross attention is requested.
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
    """Resizes an image to fit within target dimensions while maintaining aspect
    ratio, padding the remainder with `pad_value`.

    Args:
        img: `(B, C, H_src, W_src)` tensor.
        width: Target width.
        height: Target height.
        pad_value: Padding value (defaults to -1 to match SigLIP's `[-1, 1]` range).

    Returns:
        Resized/padded tensor of shape `(B, C, height, width)`.
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


def flow_matching_masked_mse(
    u_t: Tensor,
    v_t: Tensor,
    prefix_mask: Tensor,
    actions_is_pad: Tensor | None,
    max_action_dim: int,
) -> Tensor:
    """Masked MSE for π0.6 flow matching.

    Zeros out (a) frozen-prefix steps from the real-time inference delay
    (`prefix_mask=True` ⇒ frozen) and (b) fully-padded action samples — e.g.
    VQA / web co-training items, where `VQADataset` sets `actions_is_pad`
    all-True so the action expert is not trained to regress to zero on items
    that have no real actions.

    Args:
        u_t: Target velocity field, shape `(B, chunk_size, D)` (D ≥ max_action_dim).
        v_t: Predicted velocity field, same shape as `u_t`.
        prefix_mask: bool `(B, chunk_size)` — True where the step is frozen
            (real-time-inference delay). Pass `torch.zeros(...)` to disable.
        actions_is_pad: optional bool `(B, chunk_size)` — True where the
            action chunk is padded (no real action target).
        max_action_dim: Number of leading action dims to score against;
            trailing dims are dropped before averaging.
    """
    mse_loss = F.mse_loss(u_t, v_t, reduction="none")
    postfix_mask = rearrange(torch.logical_not(prefix_mask), "b c -> b c 1")
    if actions_is_pad is not None:
        in_episode_bound = rearrange(~actions_is_pad, "b c -> b c 1")
        postfix_mask = torch.logical_and(postfix_mask, in_episode_bound)
    mse_loss = mse_loss * postfix_mask
    mse_loss = mse_loss[:, :, :max_action_dim]
    postfix_mask_expanded = repeat(postfix_mask, "b c 1 -> b c d", d=mse_loss.shape[-1])
    return mse_loss.sum() / (postfix_mask_expanded.sum() + 1e-8)


def pad_discrete_tokens(tokens: list[list[int]], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pads / truncates a ragged list of FAST-tokenized action chunks to a
    fixed length, returning `(B, max_length)` arrays."""
    discrete_action_tokens = []
    discrete_action_masks = []
    for token in tokens:
        if len(token) > max_length:
            logging.warning(
                f"Discrete action token length {len(token)} > max_length {max_length}; truncating."
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


# PI06Policy — the public `PreTrainedPolicy` that OpenTau instantiates.


class PI06Policy(PreTrainedPolicy):
    """Wrapper around `PI06FlowMatching` for training and inference in OpenTau."""

    config_class = PI06Config
    name = "pi06"

    def __init__(
        self,
        config: PI06Config,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
    ):
        """Initializes the PI06Policy.

        Args:
            config: `PI06Config` instance.
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

        # π0.6 uses Gemma 3's tokenizer. The same instance is shared with the
        # inner `PI06FlowMatching` so vocab extension happens exactly once and
        # token IDs cannot drift between the two layers (e.g. if anyone
        # introduces a non-deterministic adder, two independent loads at
        # different revisions, or reorders the calls). The single
        # `ensure_loc_tokens` call inside the inner ctor extends both this
        # tokenizer and resizes the model embeddings together.
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            config.discrete_action_tokenizer_path, trust_remote_code=True
        )
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI06FlowMatching(
            config,
            discrete_action_vocab_size=discrete_action_vocab_size,
            language_tokenizer=self.language_tokenizer,
        )

        self.reset()

    def reset(self) -> None:
        """Clears the rolling action queue; call on every environment reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

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
        """Load a pretrained π0.6 checkpoint.

        Mirrors `PI05Policy.from_pretrained` — the only π0.6 specific logic is
        in `_fix_pytorch_state_dict_keys`, which tolerates both a native π0.6
        checkpoint layout and weights migrated from π0.5.
        """
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

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

        model = cls(config, **kwargs)

        acc = get_proc_accelerator()
        is_main_process = acc.is_main_process if acc else True
        # Populated inside the try block when skip_normalization_weights fires;
        # used outside the try/except to gate the inf-buffer guard so the
        # ValueError is not swallowed by the broad except below.
        stripped_keys: frozenset[str] = frozenset()
        try:
            if is_main_process:
                print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

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
                    print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                if is_main_process:
                    print(f"Could not load state dict from remote files: {e}")
                    print("Returning model without loading pretrained weights")
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
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value
            if remap_count > 0 and is_main_process:
                print(f"Remapped {remap_count} state dict keys")

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
            # warning so the noisy log does not directly contradict the INFO
            # logged just above. ``stripped_keys`` is empty when the flag is
            # off, so this is a no-op for default loads.
            unintended_missing = [key for key in missing_keys if key not in stripped_keys]

            if unintended_missing and is_main_process:
                print(f"Missing keys when loading state dict: {len(unintended_missing)} keys")
                for key in unintended_missing[:20]:
                    print(f"  - {key}")
                if len(unintended_missing) > 20:
                    print(f"  ... and {len(unintended_missing) - 20} more")
            if unexpected_keys and is_main_process:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                for key in unexpected_keys[:20]:
                    print(f"  - {key}")
                if len(unexpected_keys) > 20:
                    print(f"  ... and {len(unexpected_keys) - 20} more")
            if not unintended_missing and not unexpected_keys and is_main_process:
                print("All keys loaded successfully!")

        except Exception as e:
            if is_main_process:
                print(f"Warning: Could not remap state dict keys: {e}")

        # Outside the try/except so the ValueError is not swallowed by the
        # broad except above. The helper no-ops when ``stripped_keys`` is
        # empty (flag was off or the try block bailed before the strip ran).
        cls._assert_normalize_buffers_initialized(model, stripped_keys=stripped_keys)

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, Tensor], model_config: PreTrainedConfig
    ) -> dict[str, Tensor]:
        """Tolerate both π0.6 native checkpoints and weights migrated from π0.5.

        Specifically we (a) rewrite the top-level `paligemma_with_expert.*` →
        `gemma3_with_expert.*` prefix so a user who does
        `pretrained_path="lerobot/pi05"` as warm-start gets a graceful partial
        load instead of a full miss, (b) skip layer-norm weights that can't be
        copied into the new adaRMS layout, and (c) drop `state_proj` (never
        existed in π0.5/π0.6).
        """
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # π0.5 → π0.6 top-level module rename.
            if new_key.startswith("paligemma_with_expert."):
                new_key = new_key.replace("paligemma_with_expert.", "gemma3_with_expert.", 1)

            # π0.5 used `paligemma.language_model.*`; π0.6 uses
            # `gemma3.language_model.*` (transformers exposes Gemma 3's text
            # tower under the same attribute name on the conditional-generation
            # wrapper).
            if "gemma3_with_expert.paligemma." in new_key:
                new_key = new_key.replace("gemma3_with_expert.paligemma.", "gemma3_with_expert.gemma3.")

            # Ada-RMS weight layout compatibility. When a pi05 checkpoint stores
            # a plain `.weight`, the new adaRMS expert layer expects
            # `.dense.weight` + `.dense.bias` — drop the incompatible key.
            if re.match(
                r"gemma3_with_expert\.gemma_expert\.model\.layers\.\d+\."
                r"(input_layernorm|post_attention_layernorm)\.weight",
                new_key,
            ):
                expert_uses_adarms = getattr(
                    self.model.gemma3_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {new_key}")
                    continue

            if re.match(r"gemma3_with_expert\.gemma_expert\.model\.norm\.weight", new_key):
                expert_uses_adarms = getattr(
                    self.model.gemma3_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {new_key}")
                    continue

            # pi05 called these `time_mlp_*` already; legacy checkpoints may use
            # `action_time_mlp_*`.
            if new_key.startswith("action_time_mlp_in."):
                new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif new_key.startswith("action_time_mlp_out."):
                new_key = new_key.replace("action_time_mlp_out.", "time_mlp_out.")

            # `state_proj` doesn't exist in either π0.5 or π0.6 — drop silently.
            if new_key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi06 mode: {new_key}")
                continue

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        """Return all parameters for the optimizer."""
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Action-chunk prediction API (not used for π0.6)."""
        raise NotImplementedError("Currently not implemented for PI06")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return a single action from the rolling queue, refilling when needed.

        Matches pi05 semantics — only safe for simulation loops. Real-robot
        inference should pipeline `sample_actions` in the ROS node directly.
        """
        self.eval()

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
        """Sample an action chunk.

        The provided `action_prefix` must be *unnormalized* — this method
        normalizes internally before feeding it to the flow-matching module.
        """
        if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
            assert delay is None or 0 <= delay.item() <= self.config.max_delay, (
                f"Delay must be None or between 0 and {self.config.max_delay}"
            )

        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=lang_tokens.device)

        if action_prefix is None:
            bsize = lang_tokens.shape[0]
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            action_prefix = torch.zeros(actions_shape, dtype=lang_tokens.dtype, device=lang_tokens.device)
        else:
            action_prefix = self.normalize_targets({"actions": action_prefix}, dataset_index)["actions"]
            action_prefix = F.pad(action_prefix, (0, 0, 0, self.config.chunk_size - action_prefix.shape[1]))

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, action_prefix, delay, noise=noise
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({"actions": actions}, dataset_index)["actions"]
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Full training forward pass. Returns `{"MSE": ..., "CE": ...}`."""
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        batch["discrete_actions"] = self.normalize_discrete_actions(dict(batch), dataset_index)["actions"]
        batch = self.normalize_targets(batch, dataset_index)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        response_tokens, response_masks = self.prepare_response(batch)
        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(batch)
        actions = batch["actions"]
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            actions,
            actions_is_pad,
            response_tokens,
            response_masks,
            noise,
            time,
            discrete_actions,
            discrete_action_masks,
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]
        return {"MSE": mse_loss, "CE": ce_loss}

    # Preprocessing helpers (state discretization, image resize, etc.)

    def prepare_discrete_state(self, batch: dict[str, Tensor]) -> list[str]:
        """Discretize each state dim into 256 bins and format as a space-joined
        string, matching the π0.5 / π0.6 "State:" prompt template.
        """
        state = batch["state"]
        state_cpu = state.to(device="cpu", dtype=torch.float32)
        if torch.any(state_cpu < -1.0) or torch.any(state_cpu > 1.0):
            logging.warning(
                f"State values are not normalized between -1 and 1. "
                f"Min: {state_cpu.min().item()}, Max: {state_cpu.max().item()}"
            )
        state_clipped = torch.clamp(state_cpu, -1.0, 1.0)
        bin_indices = ((state_clipped + 1.0) * 128.0).long().clamp(0, 255)
        discretized_states = bin_indices.cpu().tolist()
        return [" ".join(map(str, row)) for row in discretized_states]

    def prepare_discrete_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize continuous actions with the FAST processor and pad to
        `discrete_action_max_length`."""
        device = batch["discrete_actions"].device
        discrete_actions = batch["discrete_actions"].to(device="cpu", dtype=torch.float32)
        tokens = self.discrete_action_processor.__call__(discrete_actions)
        discrete_action_tokens, discrete_action_masks = pad_discrete_tokens(
            tokens, self.config.discrete_action_max_length
        )
        return torch.from_numpy(discrete_action_tokens).to(device=device, dtype=torch.long), torch.from_numpy(
            discrete_action_masks
        ).to(device=device, dtype=torch.bool)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Resize (with padding) each camera view to `config.resize_imgs_with_padding`
        (π0.6 default: 448×448) and normalize from `[0, 1]` to `[-1, 1]` as SigLIP
        expects. Missing views are replaced by all-`-1` tensors up to `empty_cameras`.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the task prompt together with the discretized state string.

        Format matches π0.5:
            "Task: {task}<eos>State: {state}<eos>Response:" if predict_response
            "Task: {task}<eos>State: {state}<eos>Actions:"  otherwise
        """
        device = batch["state"].device
        tasks = batch["prompt"]
        state = self.prepare_discrete_state(batch)

        if self.config.predict_response:
            prompt = [
                f"Task: {task}<eos>State: {state}<eos>Response:"
                for task, state in zip(tasks, state, strict=False)
            ]
        else:
            prompt = [
                f"Task: {task}<eos>State: {state}<eos>Actions:"
                for task, state in zip(tasks, state, strict=False)
            ]

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
        """Tokenize the response field for supervised co-training.

        Returns `(None, None)` when response prediction is disabled.
        """
        if not self.config.predict_response:
            return None, None
        device = batch["state"].device
        responses = batch["response"]
        response_prompt = [f"{response}<eos>Actions:" for response in responses]

        tokenized_response = self.language_tokenizer.__call__(
            response_prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.response_max_length,
            return_tensors="pt",
            truncation=True,
        )
        response_tokens = tokenized_response["input_ids"].to(device=device)
        response_masks = tokenized_response["attention_mask"].to(device=device, dtype=torch.bool)
        return response_tokens, response_masks


# PI06FlowMatching — the core nn.Module doing flow-matching decoding with
# a shared per-layer KV between the Gemma 3 backbone and the Gemma-v1 expert.


class PI06FlowMatching(nn.Module):
    """π06: Gemma 3 4B backbone + Gemma-v1 action expert + flow matching.

    ┌──────────────────────────────────────────┐
    │                   actions                │
    │                   ▲                      │
    │                  ┌┴─────┐                │
    │      kv cache    │Gemma │                │
    │      ┌──────────►│Expert│                │
    │      │           │ 34L  │                │
    │     ┌┴─────────┐ │AdaRMS│                │
    │     │          │ └▲─────┘                │
    │     │Gemma 3 4B│  │                      │
    │     │(SigLIP   │  noise                  │
    │     │448×448)  │                         │
    │     └▲──▲──▲──▲                          │
    │      │  │  │  └── discrete actions       │
    │      │  │  └───── robot state            │
    │      │  └──────── language tokens        │
    │      └─────────── image(s)               │
    └──────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: PI06Config,
        discrete_action_vocab_size: int | None = None,
        language_tokenizer: AutoTokenizer | None = None,
    ):
        """Initializes the PI06FlowMatching model.

        Args:
            config: `PI06Config` instance.
            discrete_action_vocab_size: FAST tokenizer vocabulary size.
            language_tokenizer: Optional pre-loaded Gemma 3 tokenizer to share
                with the enclosing `PI06Policy`. When ``None`` (e.g. unit tests
                that construct the inner module directly) the tokenizer is
                loaded here. Either way, the same instance is used by both
                layers — there is no second copy to fall out of sync.
        """
        super().__init__()
        self.config = config

        gemma3_with_expert_config = Gemma3WithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        self.gemma3_with_expert = Gemma3WithExpertModel(gemma3_with_expert_config)

        # Action projections stay float32 for numerical stability; they're small.
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        if language_tokenizer is None:
            language_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
        self.language_tokenizer = language_tokenizer
        # π0.6 uses Gemma 3, whose stock tokenizer does NOT carry the 1024
        # <loc0000>..<loc1023> grounding tokens that PaliGemma reserves. We
        # unconditionally extend the vocab here so any grounding/VQA training
        # data containing loc tokens flows through the same response_ce_loss
        # path as on PaliGemma backbones. The new embedding rows are random-
        # init under a forked, fixed-seed RNG (see `ensure_loc_tokens`); there
        # is NO PaliGemma loc-embedding transfer. The resize must happen after
        # `Gemma3WithExpertModel(...)` has already loaded the public Gemma 3
        # weights (above), so the original 256K rows survive and only the 1024
        # new rows are freshly initialized.
        ensure_loc_tokens(self.language_tokenizer, model=self.gemma3_with_expert.gemma3)

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        """Standard Gaussian noise (float32)."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        """π0-style flow-matching time sampler: `Beta(1.5, 1.0)` on (0.001, 1.000)."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    # Embedding builders — shape matches π0.5 exactly; the block pattern
    # (image + language bidirectional, response/discrete-action causal, action
    # suffix bidirectional cross-attending to prefix) is the same as the
    # π0.6 model card specifies.

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed the prefix (image + language + optional response + discrete action
        tokens) and emit the block-pattern attention masks π0.x uses."""
        embs = []
        pad_masks = []
        att_masks = []

        bsize = None
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.gemma3_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=_preferred_dtype())

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            # Image tokens share a bidirectional block with the language tokens.
            att_masks += [0] * num_img_embs

        # Gemma 3's `embed_tokens` is a `Gemma3TextScaledWordEmbedding` that
        # already multiplies by sqrt(hidden_size) internally — do NOT scale
        # again here (unlike pi05, whose PaliGemma Gemma-v1 embedding is a
        # plain nn.Embedding with the normalizer applied later in the stock
        # forward that we bypass).
        lang_emb = self.gemma3_with_expert.embed_language_tokens(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        # Language tokens use causal attention per the π0.6 model card §2:
        # "use causal attention among the text tokens" — an explicit divergence
        # from π0.5, whose language tokens shared the image block bidirectionally.
        att_masks += [1] * num_lang_embs

        if response_tokens is not None:
            response_emb = self.gemma3_with_expert.embed_language_tokens(response_tokens)

            embs.append(response_emb)
            pad_masks.append(response_masks)
            # Response starts a new causal block.
            att_masks += [1] * response_emb.shape[1]

        if discrete_actions is not None:
            discrete_action_emb = self.gemma3_with_expert.embed_discrete_actions(discrete_actions)
            embs.append(discrete_action_emb.to(dtype=_preferred_dtype()))
            pad_masks.append(discrete_action_masks)
            # Discrete action tokens start another causal block.
            att_masks += [1] * discrete_action_emb.shape[1]

        if bsize is None:
            # No images: still need a batch size from the language tokens.
            bsize = lang_tokens.shape[0]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the noisy action chunk + timestep for the expert's bidirectional block."""
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

        def time_mlp_func(t):
            x = self.time_mlp_in(t)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        # Per-token AdaRMS condition (shape `(B, n_action_steps, proj_width)`)
        # — matches the π0.5 reference implementation exactly.
        time_emb = time_emb.to(dtype=dtype)
        adarms_cond = time_mlp_func(time_emb)

        embs.append(action_emb)
        action_mask = torch.ones(bsize, action_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(action_mask)
        # Start a new bidirectional block for the action tokens. The leading `1`
        # breaks the prefix-to-suffix block boundary, the following `0`s keep
        # all action tokens inside one bidirectional group. The block spans the full
        # chunk_size (= the noise/x_t length); n_action_steps is the execution horizon
        # applied later in select_action, not the number of action tokens. Using
        # n_action_steps here would mismatch the chunk_size-length pad mask and crash
        # make_att_2d_masks when n_action_steps < chunk_size.
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    # Training forward — flow matching + discrete-action CE + optional response CE.

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        actions_is_pad: Tensor | None = None,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full training forward pass. Returns `{"MSE": ..., "CE": ...}`."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            response_tokens,
            response_masks,
            discrete_actions,
            discrete_action_masks,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # The continuous action expert must not cross-attend to the discrete
        # action tokens (they're targets, not context). Exclude them from the cache.
        num_cross_att_tokens = prefix_embs.shape[1] - self.config.discrete_action_max_length

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

        # Real-time inference delay: randomly freeze a prefix of the action chunk.
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

        # Knowledge Insulation: block gradients from the action expert into the VLM.
        for layer_idx in past_key_values:
            past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].detach()
            past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx]["value_states"].detach()

        (_, suffix_out), _ = self.gemma3_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Supervise the whole chunk the model was trained to predict. n_action_steps
        # is the inference-time execution horizon only and must not truncate the
        # training target (chunk_size is the prediction horizon).
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        mse_loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            max_action_dim=self.config.max_action_dim,
        )

        # Discrete-action cross-entropy (FAST tokens) via the dedicated head.
        batch_size_da, seq_len = discrete_actions.shape
        discrete_token_start = -self.config.discrete_action_max_length
        discrete_action_slice_object = slice(discrete_token_start - 1, -1)
        discrete_action_out = prefix_out[:, discrete_action_slice_object]
        logits = self.gemma3_with_expert.da_head(discrete_action_out)
        logits = logits.to(dtype=torch.float32)
        logits = rearrange(logits, "b s d -> (b s) d")
        labels = rearrange(discrete_actions, "b s -> (b s)")
        discrete_action_ce_loss = F.cross_entropy(logits, labels, reduction="none")
        discrete_action_ce_loss = rearrange(
            discrete_action_ce_loss, "(b s) -> b s", b=batch_size_da, s=seq_len
        )
        discrete_action_is_pad = ~discrete_action_masks
        discrete_action_ce_loss = discrete_action_ce_loss * ~discrete_action_is_pad
        discrete_action_ce_loss = discrete_action_ce_loss.mean()

        # Optional response-token cross-entropy (via Gemma 3's shared lm_head).
        if self.config.predict_response:
            batch_size_resp, seq_len_resp = response_tokens.shape
            response_token_start = -self.config.response_max_length - self.config.discrete_action_max_length
            response_token_end = -self.config.discrete_action_max_length - 1
            response_slice_object = slice(response_token_start, response_token_end)
            response_out = prefix_out[:, response_slice_object]
            response_logits = self._gemma3_lm_head()(response_out)
            response_slice = slice(1, None)
            response_logits = response_logits.to(dtype=torch.float32)
            response_logits = rearrange(response_logits, "b s d -> (b s) d")
            response_labels = rearrange(response_tokens[:, response_slice], "b s -> (b s)")
            response_ce_loss = F.cross_entropy(response_logits, response_labels, reduction="none")
            response_ce_loss = rearrange(
                response_ce_loss, "(b s) -> b s", b=batch_size_resp, s=seq_len_resp - 1
            )
            response_is_pad = ~response_masks
            response_ce_loss = response_ce_loss * ~response_is_pad[:, response_slice]
            response_ce_loss = response_ce_loss.mean()
        else:
            response_ce_loss = torch.tensor(0.0, device=mse_loss.device)

        return {"MSE": mse_loss, "CE": discrete_action_ce_loss + response_ce_loss}

    def _gemma3_lm_head(self):
        """Return the language-modeling head of the Gemma 3 backbone, regardless
        of which transformers version layout is in use."""
        gemma3 = self.gemma3_with_expert.gemma3
        if hasattr(gemma3, "lm_head"):
            return gemma3.lm_head
        return gemma3.model.lm_head

    # Inference — flow-matching denoising, optional response autoregression.

    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        action_prefix: Tensor,
        delay: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Inference: encode prefix once, run `num_steps` Euler steps."""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None] - 1
        num_cross_att_tokens = prefix_embs.shape[1]

        (prefix_out, _), past_key_values = self.gemma3_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        response_tokens = torch.empty((bsize, 0), device=device, dtype=torch.long)
        if self.config.predict_response:
            for auto_step in range(self.config.response_max_length):
                (
                    prefix_out,
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    prefix_offsets,
                    response_tokens,
                    past_key_values,
                ) = self.infer_response(
                    prefix_out,
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    past_key_values,
                    prefix_offsets,
                    response_tokens,
                    auto_step,
                    bsize,
                    device,
                )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = rearrange(torch.arange(self.config.chunk_size, device=device), "c -> 1 c") < delay
        while time >= -dt / 2:
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
            masked_time = torch.where(prefix_mask, 0, time)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, masked_time)
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
        """Apply one Euler step of the flow-matching denoiser."""
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
        # Denoise the full chunk_size chunk so v_t matches x_t in the Euler step.
        # n_action_steps (execution horizon) is applied later in select_action, not
        # at decode time.
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t

    def infer_response(
        self,
        prefix_out: Tensor,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        prefix_offsets: Tensor,
        response_tokens: Tensor,
        auto_step: int,
        bsize: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
        """Autoregressive response-token generation for one step."""
        eos_token_id = self.language_tokenizer.convert_tokens_to_ids(self.language_tokenizer.eos_token)
        if auto_step == 0:
            response_token = torch.full(
                (bsize, 1), self.language_tokenizer.bos_token_id, device=device, dtype=torch.long
            )
        else:
            response_token = prefix_out[:, -1:]
            response_token = self._gemma3_lm_head()(response_token).argmax(dim=-1)

        pad_token_id = self.language_tokenizer.pad_token_id
        if response_tokens.shape[1] > 1:
            prev_tokens = response_tokens
            has_eos = (prev_tokens == eos_token_id).any(dim=1, keepdim=True)
            has_pad = (prev_tokens == pad_token_id).any(dim=1, keepdim=True)
            response_pad_masks = ~(has_eos | has_pad)
            response_token = torch.where(
                response_pad_masks,
                response_token,
                torch.tensor(pad_token_id, device=device, dtype=response_token.dtype),
            )
        else:
            response_pad_masks = torch.ones((bsize, 1), device=device, dtype=torch.bool)

        response_tokens = torch.cat([response_tokens, response_token], dim=1)

        # Gemma 3's `embed_tokens` already scales by sqrt(hidden_size); see
        # the note in `embed_prefix`.
        response_emb = self.gemma3_with_expert.embed_language_tokens(response_token)

        response_att_masks = torch.ones((bsize, 1), device=device, dtype=response_emb.dtype)

        prefix_embs = torch.cat([prefix_embs, response_emb], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, response_pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, response_att_masks], dim=1)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        response_att_2d_masks = make_att_2d_masks(
            response_pad_masks,
            response_att_masks,
            n_cross_att_tokens=num_cross_att_tokens - 1,
            cross_att_pad_masks=prefix_pad_masks[:, : num_cross_att_tokens - 1],
        )
        prefix_offsets = prefix_offsets + response_pad_masks.long()
        prefix_position_ids = prefix_offsets

        (prefix_out, _), past_key_values = self.gemma3_with_expert.forward(
            attention_mask=response_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[response_emb, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=True,
            fill_kv_cache=True,
        )

        return (
            prefix_out,
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_offsets,
            response_tokens,
            past_key_values,
        )
