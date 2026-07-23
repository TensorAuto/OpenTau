#!/usr/bin/env python

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

"""cosmos3: a Vision-Language-Action flow-matching policy on a frozen Qwen3-VL-32B reasoner.

cosmos3 follows the π0.5 flow-matching recipe (see ``policies/pi05/modeling_pi05.py``)
but swaps the PaliGemma backbone for a **frozen Qwen3-VL-32B** vision-language model --
the **reasoning tower of NVIDIA Cosmos3-Super** (extracted to a standalone Qwen3-VL-32B
checkpoint by ``opentau.scripts.extract_cosmos3_reasoner``) -- and pairs it with a custom
sub-1B Qwen3-style action expert (``qwen3vl_with_expert.py``).

Pipeline:
  * The frozen reasoner encodes the camera images + language prompt once (the prefix)
    via the stock ``Qwen3VLModel.forward``, producing a per-layer key/value cache.
  * The trainable expert denoises a continuous action chunk (the suffix) by flow
    matching, cross-attending to that cache at every layer. Proprioceptive state is
    projected into a single token prepended to the expert's action chunk, so actions
    are conditioned on state while the backbone sees only images + language.

Continuous actions only (MSE flow matching) -- there is no FAST discrete-action branch
and no response/subtask head, so cosmos3 always returns a zero ``CE`` term for loss-dict
compatibility with ``scripts/train.py``.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers import AutoProcessor, Qwen3VLConfig

from opentau.policies.cosmos3.configuration_cosmos3 import Cosmos3Config
from opentau.policies.cosmos3.qwen3vl_with_expert import Qwen3VLWithExpertModel
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.normalize import resolve_num_datasets as _num_datasets
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.utils import PerSampleLoss, flow_matching_masked_mse


def _preferred_dtype():
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: torch.device | str = "cpu"
) -> Tensor:
    """Sine-cosine positional embedding for scalar positions ``time`` of shape (B, N).

    Returns (B, N, dimension). Mirrors the helper in ``pi05/modeling_pi05.py``.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 2:
        raise ValueError("`time` is expected to be of shape `(batch_size, n)`.")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = rearrange(scaling_factor, "d -> 1 1 d") * rearrange(time.float(), "b n -> b n 1")
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=2)


def make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
    n_cross_att_tokens: int | None = None,
    cross_att_pad_masks: Tensor | None = None,
) -> Tensor:
    """Build a 2-D attention mask from 1-D pad + block masks (π0.5 convention).

    ``pad_masks`` bool (B, N): True = real token. ``att_masks`` int (B, N): 1 opens a new
    causal block, 0 shares the previous token's block. Returns (B, N, N) or, when
    ``n_cross_att_tokens`` is given, (B, N, n_cross + N) with full cross-attention to the
    (valid) prefix prepended. Mirrors ``pi05/modeling_pi05.py::make_att_2d_masks``.
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d = att_2d & pad_2d
    if n_cross_att_tokens is not None:
        assert cross_att_pad_masks is not None
        cross = torch.ones(
            (att_masks.size(0), att_masks.size(1), n_cross_att_tokens),
            dtype=torch.bool,
            device=att_masks.device,
        )
        cross = cross & pad_masks[:, :, None] & cross_att_pad_masks[:, None, :]
        att_2d = torch.cat((cross, att_2d), dim=2)
    return att_2d


def _first_tensor(batch: dict) -> Tensor:
    """Return the first ``torch.Tensor`` value in ``batch`` for device/batch-size inference.

    Skips non-tensor entries such as the ``prompt`` string list, which would raise on
    ``.device`` / ``.shape`` if ``next(iter(batch.values()))`` happened to hit it first.
    """
    for value in batch.values():
        if isinstance(value, Tensor):
            return value
    raise ValueError("batch contains no tensor values to infer device / batch size from")


class Cosmos3FlowMatching(nn.Module):
    """Flow-matching head: frozen Qwen3-VL prefix + trainable Qwen3 action expert."""

    def __init__(self, config: Cosmos3Config, qwen3vl_config: Qwen3VLConfig | None = None):
        super().__init__()
        self.config = config

        if qwen3vl_config is None:
            if not config.load_pretrained_backbone:
                raise ValueError(
                    "Cosmos3FlowMatching needs a Qwen3VLConfig: either set "
                    "config.load_pretrained_backbone=True (load from "
                    f"'{config.pretrained_backbone_repo_id}') or pass an explicit qwen3vl_config "
                    "(e.g. a tiny config for CPU tests)."
                )
            qwen3vl_config = Qwen3VLConfig.from_pretrained(config.pretrained_backbone_repo_id)

        self.qwen3vl_with_expert = Qwen3VLWithExpertModel(
            qwen3vl_config,
            expert_hidden_size=config.expert_hidden_size,
            expert_intermediate_size=config.expert_intermediate_size,
            expert_num_hidden_layers=config.expert_num_hidden_layers,
            expert_num_attention_heads=config.expert_num_attention_heads,
            expert_num_key_value_heads=config.expert_num_key_value_heads,
            expert_head_dim=config.expert_head_dim,
            expert_adarms_cond_dim=config.expert_adarms_cond_dim,
            expert_rms_norm_eps=config.expert_rms_norm_eps,
            dropout=config.dropout,
            attention_implementation=config.attention_implementation,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            gradient_checkpointing=config.gradient_checkpointing,
            load_pretrained_backbone_repo=(
                config.pretrained_backbone_repo_id if config.load_pretrained_backbone else None
            ),
            condition_on_layer=config.condition_on_layer,
        )

        expert_hidden = config.expert_hidden_size
        proj_width = config.proj_width
        # Action <-> expert-hidden projections, time embedding MLP, AdaRMS conditioning,
        # and the proprioceptive-state token projection. Kept in float32 then cast to the
        # backbone dtype below (so GPU bf16 runs share a dtype across the cross-attention).
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, config.max_action_dim)
        self.time_mlp_in = nn.Linear(proj_width, proj_width)
        self.time_mlp_out = nn.Linear(proj_width, proj_width)
        self.adarms_proj = nn.Linear(proj_width, config.expert_adarms_cond_dim)
        self.state_proj = nn.Linear(config.max_state_dim, expert_hidden)

        backbone_dtype = next(self.qwen3vl_with_expert.backbone.parameters()).dtype
        for module in (
            self.action_in_proj,
            self.action_out_proj,
            self.time_mlp_in,
            self.time_mlp_out,
            self.adarms_proj,
            self.state_proj,
        ):
            module.to(dtype=backbone_dtype)

    # ----- flow-matching sampling utilities (identical to pi05) -----

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        beta = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        return beta.sample((bsize,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001

    # ----- suffix embedding -----

    def _time_to_adarms(self, time: Tensor) -> Tensor:
        """(B, N) timesteps -> (B, N, adarms_cond_dim) AdaRMS conditioning vectors."""
        dtype = _preferred_dtype() if self.action_in_proj.weight.is_cuda else self.action_in_proj.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            time, self.config.proj_width, min_period=4e-3, max_period=4.0, device=time.device
        ).to(dtype)
        x = F.silu(self.time_mlp_in(time_emb))
        x = F.silu(self.time_mlp_out(x))
        return self.adarms_proj(x)

    def embed_suffix(
        self, noisy_actions: Tensor, timestep: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the proprioceptive-state token + the noisy action chunk for the expert.

        Returns (embs, pad_masks, att_masks, adarms_cond) with sequence length
        ``chunk_size + 1`` (state token prepended). The state token uses a fixed time of
        1.0 for its AdaRMS conditioning; the action tokens use ``timestep``.
        """
        dtype = self.action_in_proj.weight.dtype
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device

        action_emb = self.action_in_proj(noisy_actions.to(dtype))
        state_emb = rearrange(self.state_proj(state.to(dtype)), "b d -> b 1 d")
        embs = torch.cat([state_emb, action_emb], dim=1)

        # State token gets time=1.0; actions get their (per-step) flow-matching time.
        state_time = torch.ones(bsize, 1, dtype=torch.float32, device=device)
        time_full = torch.cat([state_time, timestep.to(torch.float32)], dim=1)
        adarms_cond = self._time_to_adarms(time_full)

        seq_len = embs.shape[1]
        pad_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        # One bidirectional block over [state, actions]; they all attend to each other
        # and (via the cross mask) to the whole prefix.
        att_masks = torch.tensor([1] + [0] * (seq_len - 1), dtype=torch.long, device=device)
        att_masks = repeat(att_masks, "n -> b n", b=bsize)
        return embs, pad_masks, att_masks, adarms_cond

    def _suffix_position_ids(self, prefix_position_ids: Tensor, suffix_len: int) -> Tensor:
        """Text-style 3-D MRoPE positions for the suffix, continuing past the prefix max.

        ``prefix_position_ids`` is (3, B, S_prefix). Returns (3, B, suffix_len) where each
        suffix token's position is identical across the temporal/height/width axes
        (the Qwen3-VL text convention), continuing from ``prefix.max() + 1`` per sample.
        """
        offset = prefix_position_ids.amax(dim=(0, 2)) + 1  # (B,)
        ar = torch.arange(suffix_len, device=prefix_position_ids.device)
        suffix = offset[:, None] + ar[None, :]  # (B, suffix_len)
        return repeat(suffix, "b n -> three b n", three=3)

    # ----- training forward -----

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        state: Tensor,
        actions: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        real_action_dim: Tensor | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Full flow-matching training forward; returns {"MSE", "CE"(=0)} (+ per-sample)."""
        device = actions.device
        batch_size = actions.shape[0]

        # 1) Frozen prefix forward -> per-layer KV cache + prefix MRoPE positions.
        prefix_position_ids, _ = self.qwen3vl_with_expert.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
        cached_kv = self.qwen3vl_with_expert.run_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # 2) Flow-matching interpolation x_t and target velocity u_t.
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(batch_size, device)

        delay = torch.randint(0, self.config.max_delay + 1, (batch_size,), device=device)
        prefix_mask = rearrange(torch.arange(self.config.chunk_size, device=device), "c -> 1 c") < rearrange(
            delay, "b -> b 1"
        )
        time = torch.where(prefix_mask, 0.0, rearrange(time, "b -> b 1"))  # (B, chunk)
        time_expanded = rearrange(time, "b c -> b c 1")
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 3) Expert (suffix) forward, cross-attending to the cached prefix KV.
        v_t = self._run_expert(cached_kv, prefix_position_ids, attention_mask.bool(), x_t, time, state)

        mse_result = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            max_action_dim=self.config.max_action_dim,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            real_action_dim=real_action_dim,
            return_per_sample=return_per_sample,
        )
        mse_loss, mse_per_sample = mse_result if return_per_sample else (mse_result, None)

        ce_loss = torch.zeros((), device=device, dtype=mse_loss.dtype)
        out: dict[str, Tensor | PerSampleLoss] = {"MSE": mse_loss, "CE": ce_loss}
        if return_per_sample:
            out["MSE_per_sample"] = mse_per_sample
            out["CE_per_sample"] = PerSampleLoss(
                sum=torch.zeros(batch_size, device=device),
                count=torch.zeros(batch_size, device=device),
            )
        return out

    def _run_expert(
        self,
        cached_kv: list[tuple[Tensor, Tensor]],
        prefix_position_ids: Tensor,
        prefix_pad: Tensor,
        x_t: Tensor,
        time: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Embed the suffix, build masks/positions, run the expert, project to velocity."""
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, time, state)
        n_cross = cached_kv[0][0].shape[2]
        attn_mask = make_att_2d_masks(
            suffix_pad, suffix_att, n_cross_att_tokens=n_cross, cross_att_pad_masks=prefix_pad
        )
        suffix_pos = self._suffix_position_ids(prefix_position_ids, suffix_embs.shape[1])
        cos, sin = self.qwen3vl_with_expert.compute_rope(
            suffix_pos, dtype=suffix_embs.dtype, device=suffix_embs.device
        )
        suffix_out = self.qwen3vl_with_expert.run_expert(
            suffix_embs, cached_kv, cos, sin, attn_mask, adarms_cond
        )
        # Drop the prepended state token; project the action chunk to velocity.
        v_t = self.action_out_proj(suffix_out[:, -self.config.chunk_size :])
        return v_t.to(dtype=torch.float32)

    # ----- inference sampling -----

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        state: Tensor,
        action_prefix: Tensor,
        delay: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Euler-integrate the flow from noise to an action chunk (π0.5 sampler)."""
        device = input_ids.device
        bsize = input_ids.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_position_ids, _ = self.qwen3vl_with_expert.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
        cached_kv = self.qwen3vl_with_expert.run_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        prefix_pad = attention_mask.bool()

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = rearrange(torch.arange(self.config.chunk_size, device=device), "c -> 1 c") < delay
        while time >= -dt / 2:
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
            masked_time = torch.where(prefix_mask, torch.zeros_like(time), time).expand(
                bsize, self.config.chunk_size
            )
            v_t = self._run_expert(cached_kv, prefix_position_ids, prefix_pad, x_t, masked_time, state)
            x_t = x_t + dt * v_t
            time = time + dt

        x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix, x_t)
        return x_t


class Cosmos3Policy(PreTrainedPolicy):
    """OpenTau wrapper around ``Cosmos3FlowMatching`` (normalization, processor, action queue)."""

    config_class = Cosmos3Config
    name = "cosmos3"
    # Leave torch.compile off until bit-identical seeded runs are verified (MRoPE /
    # dynamic shapes); the model still trains/infers eagerly.
    supports_torch_compile = False

    def __init__(
        self,
        config: Cosmos3Config,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
        qwen3vl_config: Qwen3VLConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        num_datasets = _num_datasets(per_dataset_stats, dataset_names, config)
        zero_range_center = config.zero_range_centers_on_zero()
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
        )

        # The Qwen3-VL processor (tokenizer + image processor) builds the multimodal
        # prefix. Only needed when running with the real backbone; CPU tests pass a tiny
        # ``qwen3vl_config`` and call the inner model directly with pre-built tensors.
        self.processor = None
        if config.load_pretrained_backbone:
            self.processor = AutoProcessor.from_pretrained(config.pretrained_backbone_repo_id)

        self.model = Cosmos3FlowMatching(config, qwen3vl_config=qwen3vl_config)
        self.reset()

    def reset(self) -> None:
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> list[nn.Parameter]:
        # Only the trainable expert + projections (the 32B backbone is frozen) -- never
        # hand the optimizer the frozen backbone params.
        return [p for p in self.parameters() if p.requires_grad]

    # ----- input preparation -----

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the proprioceptive state padded to ``max_state_dim`` (zeros if absent)."""
        if "state" not in batch:
            ref = _first_tensor(batch)
            return torch.zeros(
                ref.shape[0], self.config.max_state_dim, device=ref.device, dtype=torch.float32
            )
        state = batch["state"]
        state_dim = state.shape[-1]
        if state_dim > self.config.max_state_dim:
            raise ValueError(f"State dim ({state_dim}) exceeds max_state_dim ({self.config.max_state_dim}).")
        if state_dim < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state_dim))
        return state

    def prepare_multimodal_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Build Qwen3-VL ``input_ids``/``attention_mask``/``pixel_values``/``image_grid_thw``.

        Resizes every present camera image to ``image_resize`` and runs the Qwen3-VL chat
        template + processor so the language prompt is interleaved with the image tokens.
        """
        if self.processor is None:
            raise RuntimeError(
                "Cosmos3Policy.processor is None (constructed with load_pretrained_backbone=False). "
                "Call the inner model with pre-built tensors, or enable the real backbone."
            )
        image_keys = sorted(k for k in self.config.image_features if k in batch)
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        bsize = len(prompts)
        side = self.config.image_resize

        texts, images_per_sample = [], []
        for b in range(bsize):
            content = []
            sample_imgs = []
            for key in image_keys:
                img = batch[key][b]  # (C, H, W) in [0, 1]
                img = F.interpolate(
                    img[None].float(), size=(side, side), mode="bilinear", align_corners=False
                )[0]
                img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
                sample_imgs.append(rearrange(img, "c h w -> h w c").cpu().numpy())
                content.append({"type": "image"})
            # Bound the language prompt to prompt_max_length tokens. We truncate the text
            # alone (not the assembled multimodal sequence) so image placeholder tokens are
            # never clipped.
            prompt = prompts[b]
            tokenizer = self.processor.tokenizer
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[: self.config.prompt_max_length]
            prompt = tokenizer.decode(prompt_ids)
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
            images_per_sample.append(sample_imgs)

        flat_images = [img for sample in images_per_sample for img in sample] or None
        inputs = self.processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        device = batch[image_keys[0]].device if image_keys else self.prepare_state(batch).device
        return {k: v.to(device) for k, v in inputs.items()}

    # ----- training / inference -----

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        batch = self.normalize_targets(batch, dataset_index)

        mm = self.prepare_multimodal_inputs(batch)
        state = self.prepare_state(batch)
        return self.model(
            input_ids=mm["input_ids"],
            attention_mask=mm["attention_mask"],
            pixel_values=mm.get("pixel_values"),
            image_grid_thw=mm.get("image_grid_thw"),
            state=state,
            actions=batch["actions"],
            actions_is_pad=batch.get("action_is_pad"),
            noise=noise,
            time=time,
            real_action_dim=batch.get("real_action_dim"),
            return_per_sample=return_per_sample,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0 or len(self._action_queue) <= self.config.max_delay:
            action_prefix = None
            delay = 0
            if len(self._action_queue) > 0:
                prefix_actions = list(self._action_queue)
                delay = min(len(prefix_actions), self.config.max_delay)
                prefix_actions = prefix_actions[-delay:]
                action_prefix = torch.stack(prefix_actions, dim=1)
            ref = _first_tensor(batch)
            delay = torch.tensor(delay, dtype=torch.long, device=ref.device)
            actions = self.sample_actions(batch, noise=noise, action_prefix=action_prefix, delay=delay)
            actions = rearrange(actions, "b c d -> c b d")
            self._action_queue.extend(actions[delay : delay + self.config.n_action_steps])
        return self._action_queue.popleft()

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        action_prefix: Tensor | None = None,
        delay: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        mm = self.prepare_multimodal_inputs(batch)
        state = self.prepare_state(batch)
        device = mm["input_ids"].device
        bsize = mm["input_ids"].shape[0]

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=device)
        if action_prefix is None:
            action_prefix = torch.zeros(
                bsize, self.config.chunk_size, self.config.max_action_dim, dtype=torch.float32, device=device
            )
        else:
            action_prefix = self.normalize_targets({"actions": action_prefix}, dataset_index)["actions"]
            action_prefix = F.pad(action_prefix, (0, 0, 0, self.config.chunk_size - action_prefix.shape[1]))

        actions = self.model.sample_actions(
            input_ids=mm["input_ids"],
            attention_mask=mm["attention_mask"],
            pixel_values=mm.get("pixel_values"),
            image_grid_thw=mm.get("image_grid_thw"),
            state=state,
            action_prefix=action_prefix,
            delay=delay,
            noise=noise,
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return self.unnormalize_outputs({"actions": actions}, dataset_index)["actions"]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Use select_action / sample_actions for cosmos3.")
