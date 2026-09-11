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

"""xr1: an OpenTau port of Xiaomi-Robotics-1 (Qwen3-VL-4B + a 36-layer DiT).

Pipeline, per inference step:

1. Four frames per camera at stride 2 are pulled from the observation buffer (**clamped**
   at the start of an episode, never zero-padded -- see ``obs_adapter.py``), centre-cropped
   at 0.95 and resized back to 256.
2. They become three two-frame *videos*; the prompt labels them Left / Right / **Wrist**
   positionally and the Qwen3-VL backbone encodes prompt + video in one pass, leaving a
   per-layer KV cache.
3. A 21-token DiT query ``[sink, state x 4, noisy_action x 16]`` -- causal among itself,
   fully attending to the cache -- is Euler-integrated for 5 steps with **ascending** tau
   (0 -> 1, ``dt = +0.2``), one DiT layer per VLM layer.
4. The 60-wide chunk is sliced to the real 12-D RoboCasa action and executed **open-loop**.

Normalization is identity throughout (the reference's RoboCasa365 statistics are mean 0 /
std 1), and the state is *not* normalized at all -- the adapter consumes raw quaternions.
``XR1Config.__post_init__`` refuses any other mapping.

Continuous actions only. There is no discrete-action branch, so ``forward`` always returns
a real zero for ``"CE"`` rather than omitting the key: ``scripts/train.py`` indexes both
``"MSE"`` and ``"CE"`` unconditionally, and a zero ``loss_weighting`` entry hard-fails
under DeepSpeed -- so the weight stays 1.0 and the *value* is zero.

Note on CLAUDE.md rule 5's "composite forward units must be a single ``nn.Module``": that
requirement targets *interleaved* execution, where an FSDP all-gather hook has to prefetch
a backbone layer and an expert layer together. xr1, like cosmos3, runs two **disjoint**
passes -- the whole VLM, then the whole DiT against its cache -- so there is nothing to
interleave and no ``InterleavedDecoderLayer`` equivalent is needed. What the rule does bind
here is honoured: no sub-component of a wrapped layer is called from outside it, and the
async-prefix augmentation is shape-static (see :meth:`XR1FlowMatching.forward`).
"""

import builtins
import logging
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from transformers import AutoTokenizer, Qwen3VLConfig

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.accel import AccelMeter, executed_row_mask
from opentau.policies.accel import build_provenance as build_accel_provenance
from opentau.policies.accel import make_meter as make_accel_meter
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.normalize import resolve_num_datasets as _num_datasets
from opentau.policies.pretrained import (
    CheckpointWeightsNotFoundError,
    PreTrainedPolicy,
    T,
    resolve_checkpoint_provenance,
    resolve_pretrained_weights_file,
)
from opentau.policies.utils import PerSampleLoss
from opentau.policies.xr1.configuration_xr1 import XR1Config
from opentau.policies.xr1.losses import (
    build_flow_mask,
    flow_and_freq_loss,
    fold_repeat_per_sample,
)
from opentau.policies.xr1.obs_adapter import XR1ObservationBuffer, adapt_state
from opentau.policies.xr1.processing_xr1 import (
    build_action_mask,
    center_crop_resize,
    expand_video_placeholders,
    patchify_videos,
    render_chat_text,
)
from opentau.policies.xr1.qwen3vl_with_dit import Qwen3VLWithDiT
from opentau.policies.xr1.state_dict_remap import (
    assert_full_coverage,
    remap_reference_state_dict,
)
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.hub import format_repo_revision, split_repo_revision


def _first_tensor(batch: dict) -> Tensor:
    """First ``torch.Tensor`` value in ``batch``, for device / batch-size inference.

    Skips non-tensor entries such as the ``prompt`` string list, which would raise on
    ``.device`` if ``next(iter(batch.values()))`` happened to hit it first.
    """
    for value in batch.values():
        if isinstance(value, Tensor):
            return value
    raise ValueError("batch contains no tensor values to infer device / batch size from")


def resolve_qwen3vl_config(repo_id: str) -> Qwen3VLConfig:
    """Read a ``Qwen3VLConfig`` from an XR-1 checkpoint or a stock Qwen3-VL repo.

    The XR-1 checkpoint's ``config.json`` is a ``mibot`` config with the Qwen3-VL config
    nested under ``vlm_config``; a stock repo has it at the top level. Only the JSON is
    fetched -- no weights.
    """
    config_dict, _ = Qwen3VLConfig.get_config_dict(repo_id)
    if "vlm_config" in config_dict:
        config_dict = config_dict["vlm_config"]
    return Qwen3VLConfig.from_dict(config_dict)


class XR1FlowMatching(Qwen3VLWithDiT):
    """Flow-matching head: Qwen3-VL prefix + the DiT action decoder.

    Subclasses :class:`~opentau.policies.xr1.qwen3vl_with_dit.Qwen3VLWithDiT` rather than
    containing it so ``vlm`` / ``dit`` / the projectors stay **direct** children -- that is
    what keeps the reference-checkpoint remap down to a single ``model.`` prefix.
    """

    def __init__(self, config: XR1Config, qwen3vl_config: Qwen3VLConfig | None = None):
        if qwen3vl_config is None:
            if not config.load_pretrained_backbone:
                raise ValueError(
                    "XR1FlowMatching needs a Qwen3VLConfig: either set "
                    "config.load_pretrained_backbone=True (read it from "
                    f"'{config.pretrained_backbone_repo_id}') or pass an explicit qwen3vl_config "
                    "(e.g. a tiny config for CPU tests)."
                )
            qwen3vl_config = resolve_qwen3vl_config(config.pretrained_backbone_repo_id)

        super().__init__(
            qwen3vl_config,
            dit_num_hidden_layers=config.dit_num_hidden_layers,
            dit_hidden_size=config.dit_hidden_size,
            dit_intermediate_size=config.dit_intermediate_size,
            dit_num_key_value_heads=config.dit_num_key_value_heads,
            dit_head_dim=config.dit_head_dim,
            dit_time_embed_dim=config.dit_time_embed_dim,
            dit_rms_norm_eps=config.dit_rms_norm_eps,
            state_token_dim=config.state_token_dim,
            max_action_dim=config.max_action_dim,
            attention_implementation=config.attention_implementation,
            freeze_input_embeddings=config.freeze_input_embeddings,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            train_vision_encoder_only=config.train_vision_encoder_only,
            train_state_action_representation_only=config.train_state_action_representation_only,
            gradient_checkpointing=config.gradient_checkpointing,
            backbone_weights_repo_id=config.backbone_weights_repo_id,
            enable_choice_heads=config.enable_choice_heads,
            n_choices=config.n_choices,
        )
        self.config = config

    # ----- flow-matching utilities -----

    def sample_noise(self, shape: tuple[int, ...], device, dtype: torch.dtype) -> Tensor:
        """Draw the starting noise.

        The dtype is the caller's, not float32: the reference draws with
        ``torch.randn_like(action_mask)`` and the action mask is bfloat16 on the real
        model, so the noise is bf16-quantized before the first DiT call.
        """
        return torch.randn(shape, device=device, dtype=dtype)

    def sample_time(self, bsize: int, device, dtype: torch.dtype) -> Tensor:
        """``(1 - Beta(a, b).sample()) * 0.999`` -- the reference's training timestep draw.

        Note the ``1 -``: with ``Beta(1.5, 1)`` mass concentrated near 1, the *complement*
        concentrates near 0, which is where the ascending flow starts.
        """
        beta = torch.distributions.Beta(
            concentration1=self.config.time_beta_alpha, concentration0=self.config.time_beta_beta
        )
        sample = beta.sample((bsize,)).to(device=device)
        return ((1 - sample) * 0.999).to(dtype)

    def _dit_geometry(
        self,
        cached_kv: list[tuple[Tensor, Tensor]],
        prefix_position_ids: Tensor,
        prefix_pad: Tensor,
        query_len: int,
        n_suffix: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(attn_mask, position_ids)`` for a ``query_len``-token DiT query."""
        position_ids = self.dit_position_ids(prefix_position_ids, prefix_pad, query_len)
        if self.config.suffix_position_offset and n_suffix:
            position_ids = position_ids.clone()
            position_ids[:, :, -n_suffix:] += self.config.suffix_position_offset
        attn_mask = self.dit_attention_mask(prefix_pad, query_len)
        return attn_mask, position_ids

    def _run_prefix_and_geometry(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values_videos: Tensor | None,
        video_grid_thw: Tensor | None,
        state: Tensor,
        n_prefix_rows: int,
        dtype: torch.dtype,
        choice_prefix_len: int | None = None,
    ):
        """Backbone prefix + everything the DiT loop needs that does not depend on tau.

        When ``choice_prefix_len`` is given the sequence carries a trailing choice turn:
        the backbone runs over the whole thing (so the auxiliary heads see it) but the DiT's
        cache and cross-attention mask are cut to the observation prefix.
        """
        use_choice = choice_prefix_len is not None and self.choice_heads is not None
        prefix_position_ids, _ = self.get_rope_index(
            input_ids=input_ids, video_grid_thw=video_grid_thw, attention_mask=attention_mask
        )
        hidden_states = None
        if use_choice:
            # `Qwen3VLModel.forward` takes exactly one of input_ids / inputs_embeds, so the
            # choice-token embeddings are spliced into `inputs_embeds` here and `input_ids`
            # is dropped -- see `choice_heads.py` for why the video rows stay untouched.
            inputs_embeds = self.choice_heads.build_inputs_embeds(
                input_ids, self.vlm.model.get_input_embeddings(), state.to(dtype)
            )
            cached_kv, hidden_states = self.run_prefix(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=prefix_position_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                prefix_len=choice_prefix_len,
                return_hidden_states=True,
            )
        else:
            cached_kv = self.run_prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=prefix_position_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        prefix_pad = attention_mask.bool()
        if use_choice:
            prefix_pad = prefix_pad[:, :choice_prefix_len]
            prefix_position_ids = prefix_position_ids[:, :, :choice_prefix_len]
        query_len = 1 + state.shape[1] + self.config.chunk_size
        attn_mask, dit_position_ids = self._dit_geometry(
            cached_kv,
            prefix_position_ids,
            prefix_pad,
            query_len,
            self.config.chunk_size - n_prefix_rows,
            dtype,
        )
        state_embed = self.state_projector(state.to(dtype))
        if use_choice:
            return cached_kv, attn_mask, dit_position_ids, state_embed, hidden_states
        return cached_kv, attn_mask, dit_position_ids, state_embed

    def _position_embeds(self, dit_position_ids: Tensor, dtype: torch.dtype, device) -> tuple[Tensor, Tensor]:
        return self.compute_rope(dit_position_ids, dtype=dtype, device=device)

    def _euler_rollout(
        self,
        x: Tensor,
        *,
        action_mask: Tensor,
        state_embed: Tensor,
        position_embeds: tuple[Tensor, Tensor],
        cached_kv: list[tuple[Tensor, Tensor]],
        attn_mask: Tensor,
        num_steps: int,
        prefix_mask: Tensor | None = None,
    ) -> Tensor:
        """Ascending-tau Euler integration used by the **training** weight rollout.

        ``prefix_mask`` ``(B, chunk)`` marks committed rows whose velocity is zeroed, so
        they stay pinned to the ground-truth actions they were seeded with -- the reference
        does the same by writing ``output[:, :prefix_length] = 0``. It is a per-sample mask
        rather than a scalar length because the async-prefix draw here is per-sample.

        ``sample_actions`` deliberately writes its own copy of this loop rather than
        calling here: the accel-wiring registry AST-checks that ``accel.update(v_t)`` sits
        immediately before the state update *inside* ``sample_actions``, and a pin whose
        subject lives in a helper is not a pin. The two loops are otherwise identical, and
        ``tests/policies/test_xr1_cpu.py`` asserts they agree.
        """
        dt = 1.0 / num_steps
        freeze = None if prefix_mask is None else rearrange(prefix_mask, "b c -> b c 1")
        for step in range(num_steps):
            t = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / num_steps
            v_t = self.dit_forward(
                noisy_action=x,
                t=t,
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=cached_kv,
                attn_mask=attn_mask,
            )
            if freeze is not None:
                v_t = torch.where(freeze, torch.zeros_like(v_t), v_t)
            x = x + v_t * dt
        return x

    # ----- training forward -----

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values_videos: Tensor | None,
        video_grid_thw: Tensor | None,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        actions_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        real_action_dim: Tensor | None = None,
        return_per_sample: bool = False,
        choice_prefix_len: int | None = None,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Flow + frequency training forward. Returns ``{"MSE", "CE", "L1"}``.

        ``training_repeat`` flow timesteps are evaluated per sample against **one** VLM
        prefix pass (the 4.4B backbone is ~88 % of the FLOPs; the 21-token DiT is noise).
        The repeat is folded into the batch with ``repeat_interleave`` -- the reference's
        own scheme -- rather than carried as a separate axis, because
        ``F.scaled_dot_product_attention`` is not rank-invariant and a 5-D call would put
        training on different numerics from ``sample_actions``.

        **The async action prefix is shape-static.** The reference draws one Python-level
        ``prefix_length`` per batch and *slices* the DiT query, which makes the sequence
        length a per-rank random variable -- ranks then disagree on all-gather sizes under
        ZeRO-3 / FSDP (CLAUDE.md rule 5) and the graph is untraceable. Here the prefix is a
        per-sample **mask** over a full-length chunk instead: rows below ``prefix_len`` are
        clamped to the ground-truth action with ``time = 0`` and dropped from the loss, and
        every tensor keeps its shape whatever the draw.
        """
        device = actions.device
        batch_size = actions.shape[0]
        dtype = self.action_projector.layers[0].weight.dtype
        repeat_n = self.config.training_repeat if self.training else 1

        use_choice = choice_prefix_len is not None and self.choice_heads is not None
        geometry = self._run_prefix_and_geometry(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            state=state,
            n_prefix_rows=0,
            dtype=dtype,
            choice_prefix_len=choice_prefix_len,
        )
        if use_choice:
            cached_kv, attn_mask, dit_position_ids, state_embed, hidden_states = geometry
        else:
            cached_kv, attn_mask, dit_position_ids, state_embed = geometry
            hidden_states = None

        # --- async action prefix (per-sample, shape-static) ---------------------------
        chunk = self.config.chunk_size
        if self.training and self.config.train_prefix_max > 0 and self.config.train_prefix_prob > 0:
            enabled = torch.rand(batch_size, device=device) < self.config.train_prefix_prob
            drawn = torch.randint(1, self.config.train_prefix_max + 1, (batch_size,), device=device)
            prefix_len = torch.where(enabled, drawn, torch.zeros_like(drawn))
        else:
            prefix_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        prefix_mask = rearrange(torch.arange(chunk, device=device), "c -> 1 c") < rearrange(
            prefix_len, "b -> b 1"
        )

        # --- fold the repeat axis into the batch ---------------------------------------
        actions_r = actions.repeat_interleave(repeat_n, dim=0)
        action_mask_r = action_mask.repeat_interleave(repeat_n, dim=0)
        prefix_mask_r = prefix_mask.repeat_interleave(repeat_n, dim=0)
        state_embed_r = state_embed.repeat_interleave(repeat_n, dim=0)
        attn_mask_r = attn_mask.repeat_interleave(repeat_n, dim=0)
        dit_position_ids_r = dit_position_ids.repeat_interleave(repeat_n, dim=1)
        is_pad_r = actions_is_pad.repeat_interleave(repeat_n, dim=0) if actions_is_pad is not None else None
        real_dim_r = (
            real_action_dim.repeat_interleave(repeat_n, dim=0) if real_action_dim is not None else None
        )
        position_embeds = self._position_embeds(dit_position_ids_r, dtype=dtype, device=device)

        if noise is None:
            noise = self.sample_noise(actions_r.shape, device, actions_r.dtype)
        else:
            noise = noise.repeat_interleave(repeat_n, dim=0) if noise.shape[0] == batch_size else noise
        if time is None:
            if self.training or not self.config.val_deterministic_time:
                time = self.sample_time(actions_r.shape[0], device, actions_r.dtype)
            else:
                # Deterministic tau grid in validation: otherwise the curve is dominated by
                # timestep variance rather than by the model.
                grid = (torch.arange(actions_r.shape[0], device=device) + 0.5) / actions_r.shape[0]
                time = grid.to(actions_r.dtype)
        time = rearrange(time, "b -> b 1 1")
        # Frozen prefix rows sit at tau = 0 and carry the ground-truth action.
        time = torch.where(rearrange(prefix_mask_r, "b c -> b c 1"), torch.zeros_like(time), time)

        noisy_action = (1 - time) * noise + time * actions_r
        noisy_action = torch.where(rearrange(prefix_mask_r, "b c -> b c 1"), actions_r, noisy_action)
        target = actions_r - noise

        pred = self.dit_forward(
            noisy_action=noisy_action,
            t=time,
            action_mask=action_mask_r,
            state_embed=state_embed_r,
            position_embeds=position_embeds,
            past_key_values=cached_kv,
            attn_mask=attn_mask_r,
        )

        # --- per-sample loss weight from a no-grad rollout of the prefix ----------------
        # Runs unconditionally (5 extra DiT passes over 21 tokens, reusing the SAME cache)
        # so there is no data-dependent branch; rows with no prefix come out at weight 1.
        weight = self._prefix_rollout_weight(
            actions=actions_r,
            noise=noise,
            prefix_mask=prefix_mask_r,
            action_mask=action_mask_r,
            state_embed=state_embed_r,
            position_embeds=position_embeds,
            cached_kv=cached_kv,
            attn_mask=attn_mask_r,
        )

        mask = build_flow_mask(
            batch_size=actions_r.shape[0],
            chunk_size=chunk,
            max_action_dim=self.config.max_action_dim,
            device=device,
            prefix_mask=prefix_mask_r,
            actions_is_pad=is_pad_r,
            real_action_dim=real_dim_r,
        )
        result = flow_and_freq_loss(
            pred,
            target,
            mask=mask,
            weight=weight,
            freq_excluded_dims=tuple(self.config.freq_loss_excluded_dims),
            weight_clamp=tuple(self.config.weight_clamp),
            return_per_sample=return_per_sample,
        )
        if return_per_sample:
            loss_mse, loss_freq, per_sample = result
        else:
            loss_mse, loss_freq = result
            per_sample = None

        mse_term = self.config.mse_loss_scale * loss_mse + self.config.freq_loss_weight * loss_freq

        # The "CE" key carries the VLM-side auxiliary half of the reference's scalar
        # (0.5 * choice + 0.5 * score). It is a **real zero tensor** when the heads are off
        # rather than an absent key: train.py indexes both loss keys unconditionally, and a
        # zero `loss_weighting` entry hard-fails under DeepSpeed -- so the weight stays 1.0
        # and the value is what goes to zero.
        loss_choice = loss_score = None
        if use_choice:
            from opentau.policies.xr1.choice_heads import choice_loss

            action_pred, score_pred = self.choice_heads.predict(hidden_states, input_ids)
            choice_mask = build_flow_mask(
                batch_size=batch_size,
                chunk_size=chunk,
                max_action_dim=self.config.max_action_dim,
                device=device,
                actions_is_pad=actions_is_pad,
                real_action_dim=real_action_dim,
            )
            loss_choice, loss_score = choice_loss(action_pred, score_pred, actions, choice_mask)
            ce_term = 0.5 * loss_choice + 0.5 * loss_score
        else:
            ce_term = torch.zeros((), device=device, dtype=mse_term.dtype)
        with torch.no_grad():
            l1 = (pred.float() - target.float()).abs()
            l1_term = (l1 * mask).sum() / (mask.sum() + 1e-8)

        out: dict[str, Tensor | PerSampleLoss] = {
            "MSE": mse_term,
            "CE": ce_term,
            "L1": l1_term,
            "loss_mse": loss_mse.detach(),
            "loss_freq": loss_freq.detach(),
        }
        if loss_choice is not None:
            out["loss_choice"] = loss_choice.detach()
            out["loss_score"] = loss_score.detach()
        if return_per_sample:
            # Reduce the repeat axis INTO the sample: train.py gathers per-sample losses in
            # one call alongside (B,) provenance tensors, so a (B * R,) vector silently
            # misattributes every row rather than erroring.
            out["MSE_per_sample"] = fold_repeat_per_sample(per_sample, batch_size, repeat_n)
            out["CE_per_sample"] = PerSampleLoss(
                sum=torch.zeros(batch_size, device=device),
                count=torch.zeros(batch_size, device=device),
            )
        return out

    @torch.no_grad()
    def _prefix_rollout_weight(
        self,
        *,
        actions: Tensor,
        noise: Tensor,
        prefix_mask: Tensor,
        action_mask: Tensor,
        state_embed: Tensor,
        position_embeds: tuple[Tensor, Tensor],
        cached_kv: list[tuple[Tensor, Tensor]],
        attn_mask: Tensor,
    ) -> Tensor:
        """``|rollout(prefix, noise) - action|``: how hard each slot is given the prefix.

        The reference runs the full 5-step sampler with the committed prefix clamped in,
        and weights the flow MSE by how far that free-running rollout lands from the truth.
        Rows with no prefix get exactly 1 (the reference's ``ones_like``), which this
        reproduces by ``where``-ing the rollout result against ones rather than by
        branching -- so the graph shape never depends on the draw.
        """
        has_prefix = reduce(prefix_mask, "b c -> b", "any")
        x = torch.where(rearrange(prefix_mask, "b c -> b c 1"), actions, noise)
        rolled = self._euler_rollout(
            x,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            cached_kv=cached_kv,
            attn_mask=attn_mask,
            num_steps=self.config.num_steps,
            prefix_mask=prefix_mask,
        )
        weight = (rolled.float() - actions.float()).abs()
        return torch.where(rearrange(has_prefix, "b -> b 1 1"), weight, torch.ones_like(weight))

    # ----- inference sampling -----

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values_videos: Tensor | None,
        video_grid_thw: Tensor | None,
        state: Tensor,
        action_mask: Tensor,
        action_prefix: Tensor | None = None,
        delay: Tensor | None = None,
        noise: Tensor | None = None,
        accel: AccelMeter | None = None,
    ) -> Tensor:
        """Euler-integrate noise -> action chunk. Returns ``(B, chunk_size, max_action_dim)``.

        Args:
            input_ids: Prompt token ids with the video placeholders already expanded.
            attention_mask: 1 = real token.
            pixel_values_videos: Flattened Qwen3-VL video patches, or ``None``.
            video_grid_thw: Per-video ``(t, h, w)`` patch grid.
            state: ``(B, n_obs_steps, state_token_dim)`` adapted state tokens.
            action_mask: ``(B, chunk_size, max_action_dim)`` float mask; also fixes the
                noise dtype/shape, exactly as the reference's ``randn_like`` does.
            action_prefix: Committed actions freezing the first ``delay`` rows, or ``None``.
            delay: Number of frozen rows (real-time chunking). ``None`` ⇒ 0.
            noise: Optional starting noise; drawn here when ``None``. Threaded through so
                the parity fixtures can inject the reference's own draw.
            accel: Optional denoising-acceleration meter. ``None`` (the default) makes
                every added branch a compile-time constant, so the traced graph is
                unchanged. Passed in rather than stashed on ``self`` because writing to an
                ``nn.Module`` attribute inside a traced region is a Dynamo side-effect.
        """
        device = input_ids.device
        dtype = self.action_projector.layers[0].weight.dtype
        chunk = self.config.chunk_size

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=device)
        prefix_mask = rearrange(torch.arange(chunk, device=device), "c -> 1 c") < delay
        # Kept as a mask rather than an int length: `int(delay.max().item())` would force a
        # device sync every call (and break a traced graph), and a per-sample delay would
        # over-freeze every row up to the batch maximum.
        freeze_rows = rearrange(prefix_mask, "b c -> b c 1") if action_prefix is not None else None

        cached_kv, attn_mask, dit_position_ids, state_embed = self._run_prefix_and_geometry(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            state=state,
            n_prefix_rows=0,
            dtype=dtype,
        )
        position_embeds = self._position_embeds(dit_position_ids, dtype=action_mask.dtype, device=device)

        if noise is None:
            noise = self.sample_noise(action_mask.shape, device, action_mask.dtype)
        x_t = noise
        if action_prefix is not None:
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix.to(x_t.dtype), x_t)

        if accel is not None:
            accel.set_row_mask(
                executed_row_mask(
                    prefix_mask=prefix_mask,
                    delay=delay,
                    chunk_size=chunk,
                    n_action_steps=self.config.n_action_steps,
                    device=device,
                )
            )

        # The Euler loop is written out here rather than delegated to `_euler_rollout` so
        # the accel-wiring invariants (update reads the loop's own velocity, and runs
        # before the state advances) are checkable in this function -- see the helper's
        # docstring. tau ASCENDS: dt is +1/num_steps and x moves *with* the velocity.
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones((x_t.shape[0], 1, 1), device=device, dtype=x_t.dtype) * step / num_steps
            v_t = self.dit_forward(
                noisy_action=x_t,
                t=t,
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=cached_kv,
                attn_mask=attn_mask,
            )
            if freeze_rows is not None:
                # Freeze the committed rows: their velocity is zeroed so the already-executed
                # actions cannot drift over the remaining steps.
                v_t = torch.where(freeze_rows, torch.zeros_like(v_t), v_t)
            if accel is not None:
                # Must read `v_t` here: with frozen prefix rows `(x_{k+1} - x_k) / dt` is not
                # `v_t` on those rows, so accel cannot be recovered from the returned chunk.
                accel.update(v_t)
            x_t = x_t + v_t * dt
        if action_prefix is not None:
            x_t = torch.where(rearrange(prefix_mask, "b c -> b c 1"), action_prefix.to(x_t.dtype), x_t)
        return x_t


class XR1Policy(PreTrainedPolicy):
    """OpenTau wrapper around :class:`XR1FlowMatching`.

    Owns the identity normalization modules, the tokenizer, the prompt/video front end and
    the observation-history + action queues.

    On ``accel``: the structural wiring is present and pinned by
    ``tests/policies/test_accel_wiring_registry.py``, but the meter can never actually arm
    for this policy -- ``opentau.policies.accel.make_meter`` refuses IDENTITY-normalized
    actions, and xr1's normalization is identity by construction. The wiring is kept so a
    future non-identity xr1 variant inherits a correct sampler rather than a subtly wrong
    one bolted on later.
    """

    config_class = XR1Config
    name = "xr1"
    # Leave torch.compile off until bit-identical seeded runs are verified (MRoPE, dynamic
    # prompt lengths); the model still trains and infers eagerly.
    supports_torch_compile = False

    def __init__(
        self,
        config: XR1Config,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
        qwen3vl_config: Qwen3VLConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        num_datasets = _num_datasets(per_dataset_stats, dataset_names, config)
        zero_range_center = config.zero_range_centers_on_zero()
        eps = config.normalization_epsilon()
        norm_kwargs = {
            "per_dataset_stats": per_dataset_stats,
            "dataset_names": dataset_names,
            "num_datasets": num_datasets,
            "zero_range_center": zero_range_center,
            "eps": eps,
        }
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, **norm_kwargs)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, **norm_kwargs
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, **norm_kwargs
        )

        # Only the tokenizer is needed at runtime: the prompt text is a fixed template
        # (`processing_xr1.CHAT_TEMPLATE`) and the video patching is done here rather than
        # by a `Qwen3VLVideoProcessor`, so no `trust_remote_code` module has to be vendored.
        self.tokenizer = None
        if config.load_pretrained_backbone:
            tokenizer_kwargs = {}
            if config.enable_choice_heads:
                # The heads index hidden states by token-id *range*, so the extra special
                # tokens have to exist and land on the reference's ids.
                from opentau.policies.xr1.choice_heads import SPECIAL_TOKENS, assert_reference_token_ids

                tokenizer_kwargs["extra_special_tokens"] = SPECIAL_TOKENS
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.pretrained_backbone_repo_id, **tokenizer_kwargs
            )
            if config.enable_choice_heads:
                assert_reference_token_ids(self.tokenizer)

        self.model = XR1FlowMatching(config, qwen3vl_config=qwen3vl_config)

        # Denoising-acceleration uncertainty proxy; off by default (see the class docstring
        # for why it can never arm on an identity-normalized policy).
        self.accel_prefix: int | None = None

        self._obs_buffer = XR1ObservationBuffer(
            n_obs_steps=config.n_obs_steps,
            history_interval=config.history_interval,
            buffer_size=config.obs_buffer_size,
        )
        self.reset()

    def reset(self) -> None:
        """Clear the action queue, the observation history and the per-plan accel state."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._obs_buffer.reset()
        self.last_accel = None
        self.last_accel_provenance = None

    def get_optim_params(self) -> list[nn.Parameter]:
        """Trainable parameters only.

        Filtering on ``requires_grad`` rather than returning ``self.parameters()`` is worth
        ~4.7 GB: the frozen input-embedding table is 389 M parameters, and a fused AdamW
        handed a frozen tensor still allocates its fp32 master copy plus both moments.
        """
        return [p for p in self.parameters() if p.requires_grad]

    # ----- input preparation -----

    def _image_keys(self) -> list[str]:
        """Camera keys in **positional** order (``camera0``, ``camera1``, ...).

        Deliberately not ``sorted()``: lexicographic order puts ``camera10`` before
        ``camera2``, and the prompt labels cameras by position, so a 10-camera config
        would silently tell the model the wrong view.
        """
        return [f"camera{i}" for i in range(self.config.num_cams)]

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Adapt and pad the state to ``(B, n_obs_steps, state_token_dim)``.

        Accepts either a ``(B, T, D)`` history batch or a ``(B, D)`` single frame (which is
        broadcast across the ``n_obs_steps`` slots, matching the buffer's clamping at the
        very first step of an episode).
        """
        state = batch["state"]
        if state.ndim == 2:
            state = repeat(state, "b d -> b t d", t=self.config.n_obs_steps)
        return adapt_state(
            state,
            state_adapter=self.config.state_adapter,
            state_token_dim=self.config.state_token_dim,
            quat_order=self.config.quat_order,
        )

    def prepare_multimodal_inputs(
        self, batch: dict[str, Tensor], include_choice_turn: bool = True
    ) -> dict[str, Tensor]:
        """Build ``input_ids`` / ``attention_mask`` / ``pixel_values_videos`` / ``video_grid_thw``.

        Each camera's ``n_obs_steps`` frames become one video; the frames are centre-cropped
        at ``center_crop_ratio`` and resized back before patching.

        Args:
            batch: The (already normalized) observation batch.
            include_choice_turn: Append the ``<state>`` / ``<a_i>`` / ``<score>`` turn when
                the choice heads are enabled. ``False`` on the inference path -- the heads
                are a training-only auxiliary signal and the reference's evaluator never
                sends the turn, so including it there would change the prompt the
                parity gates were verified against.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "XR1Policy.tokenizer is None (constructed with load_pretrained_backbone=False). "
                "Call the inner model with pre-built tensors, or enable the real backbone."
            )
        keys = self._image_keys()
        missing = [key for key in keys if key not in batch]
        if missing:
            raise ValueError(f"xr1 expects cameras {keys}; missing {missing} from the batch.")

        frames = []
        for key in keys:
            cam = batch[key]
            if cam.ndim == 4:  # (B, C, H, W) -> broadcast a single frame over the history
                cam = repeat(cam, "b c h w -> b t c h w", t=self.config.n_obs_steps)
            frames.append(center_crop_resize(cam, self.config.center_crop_ratio))
        # (B, num_cams, T, C, H, W) -> one video per (sample, camera), camera-major within
        # each sample so the flattened patch rows line up with the prompt's marker order.
        stacked = torch.stack(frames, dim=1)
        bsize = stacked.shape[0]
        videos = rearrange(stacked, "b n t c h w -> (b n) t c h w")

        pixel_values_videos, video_grid_thw = patchify_videos(
            videos,
            patch_size=self.config.vision_patch_size,
            temporal_patch_size=self.config.vision_temporal_patch_size,
            merge_size=self.config.vision_merge_size,
            image_mean=tuple(self.config.vision_image_mean),
            image_std=tuple(self.config.vision_image_std),
        )

        prompts = batch.get("prompt", [""] * bsize)
        if isinstance(prompts, str):
            prompts = [prompts]
        frame_indices = [list(range(self.config.n_obs_steps))] * self.config.num_cams
        per_sample_grid = video_grid_thw[: self.config.num_cams]
        texts = []
        for b in range(bsize):
            text = render_chat_text(prompts[b], self.config.camera_prompt_labels[: self.config.num_cams])
            texts.append(
                expand_video_placeholders(
                    text,
                    per_sample_grid,
                    self.config.vision_merge_size,
                    frame_indices,
                    video_fps=self.config.video_timestamp_fps,
                )
            )
        # LEFT padding, always. Two reasons: `get_rope_index` writes real positions only
        # where the mask is set, so a contiguous run of real tokens is what it expects; and
        # when the choice turn is appended below, a uniform observation-prefix length is
        # what makes the DiT's cache cut a single slice rather than a per-sample gather.
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        device = stacked.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        if self.config.enable_choice_heads and include_choice_turn:
            from opentau.policies.xr1.choice_heads import render_choice_turn

            suffix = self.tokenizer(
                [render_choice_turn(self.config.chunk_size)],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)
            suffix = repeat(suffix, "1 s -> b s", b=bsize)
            out["choice_prefix_len"] = int(input_ids.shape[1])
            out["input_ids"] = torch.cat([input_ids, suffix], dim=1)
            out["attention_mask"] = torch.cat(
                [attention_mask, torch.ones_like(suffix, dtype=attention_mask.dtype)], dim=1
            )
        return out

    def _action_mask(self, batch: dict[str, Tensor], batch_size: int, device, dtype: torch.dtype) -> Tensor:
        """The DiT's action mask, from the per-sample dataset signal when there is one.

        The two paths disagree about what ``config.action_feature`` means, which is why this
        cannot just read it:

        * **Inference.** The eval config declares the robot's true action width (12 for
          RoboCasa), no dataset is involved, and no ``real_action_dim`` reaches the batch.
        * **Training.** ``make_policy`` overwrites ``output_features`` from the dataset, and
          a heterogeneous mixture reports ``actions`` as ``(max_action_dim,)`` -- 60 -- for
          every dataset in it. Reading the declared width there yields an **all-ones** mask,
          so ``noisy_action * action_mask`` stops zeroing the padding columns and the DiT
          trains on noise where inference feeds it zeros. Nothing fails: the loss is masked
          from ``real_action_dim`` separately, so only the input distribution silently
          diverges from the one every parity gate verified.

        Hence: prefer the per-sample ``real_action_dim`` the dataset emits, and refuse a
        training batch that lacks it rather than falling back to a width that is wrong there.
        """
        real_action_dim = batch.get("real_action_dim")
        if real_action_dim is not None:
            return build_action_mask(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                real_action_dim=real_action_dim,
                device=device,
                dtype=dtype,
            )
        if self.training:
            raise ValueError(
                "A training batch must carry `real_action_dim` (LeRobotDataset emits it): the "
                "DiT's action mask is per-sample, and `config.action_feature` is not a usable "
                "stand-in during training — `make_policy` sets it from the dataset mixture, "
                f"which reports every action as max_action_dim ({self.config.max_action_dim}) "
                "wide, so the mask would come out all-ones and the padded columns would carry "
                "noise instead of zeros."
            )
        return build_action_mask(
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
            self.config.action_feature.shape[0],
            device=device,
            dtype=dtype,
        )

    # ----- training / inference -----

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Training forward. Always returns both ``"MSE"`` and ``"CE"``."""
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        batch = self.normalize_targets(batch, dataset_index)

        mm = self.prepare_multimodal_inputs(batch)
        state = self.prepare_state(batch)
        actions = batch["actions"]
        if actions.shape[-1] < self.config.max_action_dim:
            actions = F.pad(actions, (0, self.config.max_action_dim - actions.shape[-1]))
        dtype = self.model.action_projector.layers[0].weight.dtype
        action_mask = self._action_mask(batch, actions.shape[0], actions.device, dtype)

        return self.model(
            input_ids=mm["input_ids"],
            attention_mask=mm["attention_mask"],
            pixel_values_videos=mm["pixel_values_videos"],
            video_grid_thw=mm["video_grid_thw"],
            state=state,
            actions=actions.to(dtype),
            action_mask=action_mask,
            actions_is_pad=batch.get("action_is_pad"),
            noise=noise,
            time=time,
            real_action_dim=batch.get("real_action_dim"),
            return_per_sample=return_per_sample,
            choice_prefix_len=mm.get("choice_prefix_len"),
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Pop one action, re-planning when the queue runs dry. Returns ``(B, action_dim)``."""
        self.eval()

        # Cleared every step so a consumer can distinguish "this step re-planned, here is
        # its score" from "this step popped a queued action".
        self.last_accel = None

        batch = self._buffer_observation(batch)

        if len(self._action_queue) == 0 or len(self._action_queue) <= self.config.max_delay:
            action_prefix = None
            delay = 0
            if len(self._action_queue) > 0:
                prefix_actions = list(self._action_queue)
                delay = min(len(prefix_actions), self.config.max_delay)
                prefix_actions = prefix_actions[-delay:]
                action_prefix = torch.stack(prefix_actions, dim=1)
            ref = _first_tensor(batch)
            delay_t = torch.tensor(delay, dtype=torch.long, device=ref.device)
            actions = self.sample_actions(batch, noise=noise, action_prefix=action_prefix, delay=delay_t)
            actions = rearrange(actions, "b c d -> c b d")
            self._action_queue.extend(actions[delay : delay + self.config.n_action_steps])
        return self._action_queue.popleft()

    def _buffer_observation(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Append this step's frame to the history buffer and return the stacked window.

        Unlike the pi07 / pi05_mem buffers this **clamps** rather than zero-pads at the
        start of an episode -- the reference's convention, and the same one OpenTau's
        dataset uses at episode boundaries, so train and eval see the same distribution.
        """
        keys = [key for key in self._image_keys() if key in batch]
        if "state" not in batch:
            raise ValueError("xr1 requires a `state` entry in the observation batch.")
        self._obs_buffer.append(batch["state"], {key: batch[key] for key in keys})
        state, images, is_pad = self._obs_buffer.window()
        windowed = {k: v for k, v in batch.items() if k not in {*keys, "state"}}
        windowed["state"] = state
        windowed.update(images)
        windowed["obs_history_is_pad"] = is_pad
        return windowed

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        action_prefix: Tensor | None = None,
        delay: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Produce one action chunk, ``(B, n_action_steps_or_chunk, action_dim)``."""
        dataset_index = self._resolve_dataset_index(batch)
        batch = self.normalize_inputs(batch, dataset_index)
        mm = self.prepare_multimodal_inputs(batch, include_choice_turn=False)
        state = self.prepare_state(batch)
        device = mm["input_ids"].device
        bsize = mm["input_ids"].shape[0]
        dtype = self.model.action_projector.layers[0].weight.dtype
        action_mask = self._action_mask(batch, bsize, device, dtype)

        if delay is None:
            delay = torch.tensor(0, dtype=torch.long, device=device)
        if action_prefix is not None:
            action_prefix = self.normalize_targets({"actions": action_prefix}, dataset_index)["actions"]
            action_prefix = F.pad(
                action_prefix,
                (
                    0,
                    self.config.max_action_dim - action_prefix.shape[-1],
                    0,
                    self.config.chunk_size - action_prefix.shape[1],
                ),
            )

        accel_meter = make_accel_meter(self, batch_size=bsize, device=device, dataset_index=dataset_index)

        actions = self.model.sample_actions(
            input_ids=mm["input_ids"],
            attention_mask=mm["attention_mask"],
            pixel_values_videos=mm["pixel_values_videos"],
            video_grid_thw=mm["video_grid_thw"],
            state=state,
            action_mask=action_mask,
            action_prefix=action_prefix,
            delay=delay,
            noise=noise,
            accel=accel_meter,
        )

        if accel_meter is not None:
            # `.to_list()` rather than a tensor: anything allocated inside
            # `torch.inference_mode()` stays an inference tensor even after `.clone()`.
            self.last_accel = accel_meter.to_list()
            self.last_accel_provenance = build_accel_provenance(
                self, accel_meter, dataset_index=dataset_index
            )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim].float()
        return self.unnormalize_outputs({"actions": actions}, dataset_index)["actions"]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Not implemented; use :meth:`select_action` / :meth:`sample_actions`."""
        raise NotImplementedError("Use select_action / sample_actions for xr1.")

    # ----- checkpoint loading -----

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
        """Load an xr1 checkpoint, or a native Xiaomi-Robotics-1 one.

        The only xr1-specific step is :func:`remap_reference_state_dict`, which prefixes the
        reference's top-level module names with ``model.``. Everything else mirrors
        ``PI06Policy.from_pretrained``: resolve the weights *before* building a 5 B-parameter
        model so a typo'd repo fails in milliseconds, and resolve the checkpoint's
        ``config_version`` before ``cls(config)`` builds the Normalize modules off it.
        """
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # "<repo_id>@<revision>" selects a published step by its git tag; a local path
        # (even one containing "@") is returned untouched.
        repo_id, revision = split_repo_revision(pretrained_name_or_path, revision)

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=repo_id,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        acc = get_proc_accelerator()
        is_main_process = acc.is_main_process if acc else True

        try:
            weights_file: str | None = resolve_pretrained_weights_file(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
        except CheckpointWeightsNotFoundError as e:
            if is_main_process:
                logging.warning(
                    "%s Building the policy WITHOUT pretrained weights — expected only when "
                    "resuming a DeepSpeed/ZeRO run, where accelerator.load_state restores them "
                    "next. Otherwise this policy is randomly initialized.",
                    e,
                )
            weights_file = None

        # Key the config to the conventions these weights were trained under, BEFORE
        # `cls(config)` builds the Normalize modules off `config_version`.
        resolve_checkpoint_provenance(
            config,
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            local_files_only=local_files_only,
        )

        model = cls(config, **kwargs)
        if weights_file is None:
            return model

        if is_main_process:
            print(f"Loading model from: {format_repo_revision(repo_id, revision)}")
        from safetensors.torch import load_file

        original_state_dict = load_file(weights_file)

        stripped_keys: frozenset[str] = frozenset()
        load_result: tuple[list[str], list[str]] | None = None
        try:
            remapped_state_dict = remap_reference_state_dict(original_state_dict)
            model._promote_legacy_norm_buffers_in_state_dict(remapped_state_dict)
            remapped_state_dict, stripped_keys = cls._strip_normalization_buffers_from_state_dict(
                remapped_state_dict, model.config, is_main_process=is_main_process
            )
            load_result = model.load_state_dict(remapped_state_dict, strict=False)
        except Exception as e:
            if is_main_process:
                print(f"Warning: Could not remap state dict keys: {e}")

        # Deliberately OUTSIDE the broad catch above. The other policies keep that catch
        # because they have documented partial-load warm-start paths (e.g. pi06 from a
        # pi05 checkpoint) where `strict=False` absorbing key mismatches is the intended
        # behaviour. xr1 has no such path: it loads Xiaomi-Robotics-1 checkpoints, whose
        # coverage is total. Raising inside the `try` would have the `except` print the
        # failure as a warning and hand back a half-random 5B model -- which is exactly the
        # contract `state_dict_remap.assert_full_coverage` exists to prevent, so the gate
        # has to sit where nothing can swallow it.
        if load_result is None:
            raise ValueError(
                "Loading the xr1 checkpoint failed before `load_state_dict` completed (see the "
                "warning above). Refusing to return a randomly-initialized 5B model: unlike the "
                "pi05/pi06 lineage, xr1 has no partial-load warm-start path, so an incomplete "
                "load is always a bug rather than a supported mode."
            )
        assert_full_coverage(*load_result, stripped_keys=stripped_keys)
        # Re-establish the input-embedding / lm_head alias. A checkpoint carries only one end
        # of it (see TIED_WEIGHT_KEYS), and `load_state_dict` writes into the existing
        # storage -- so whichever end arrived, the other is only correct once the tie is
        # restored. Cheap and idempotent when the tie is already intact.
        model.model.vlm.tie_weights()
        if is_main_process:
            print("All keys loaded successfully!")

        cls._assert_normalize_buffers_initialized(model, stripped_keys=stripped_keys)

        return model
