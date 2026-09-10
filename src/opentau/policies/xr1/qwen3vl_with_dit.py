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

"""Stock Qwen3-VL backbone + the Xiaomi-Robotics-1 DiT head.

The xr1 analogue of ``cosmos3/qwen3vl_with_expert.py``, and structured the same way: the
backbone runs as a **black box** through stock ``transformers`` (vision encoding, video
token scatter, DeepStack injection, 3-D MRoPE, QK-norm, causal masking), and only its
per-layer key/value cache crosses into the hand-written action head. The prefix forward
and the DiT forward never run in the same pass, so nothing about the 4.4B backbone is
reimplemented here.

Several blocks are lifted from ``cosmos3/qwen3vl_with_expert.py`` (attribution inline):
``compute_rope`` (:521-533 there), the ``get_rope_index`` delegation, and the
``no_grad`` / ``nullcontext`` + ``.detach()`` structure of ``run_prefix`` (:535-609),
re-keyed on the same question -- *is any backbone parameter trainable?*

**Module naming is load-bearing.** The children here are named ``vlm``, ``dit``,
``state_projector``, ``action_projector``, ``action_output_layer``, ``t_embedder``,
``t_projector`` and ``sink`` because those are the reference checkpoint's top-level keys.
``XR1FlowMatching`` *subclasses* this, so the whole state-dict remap collapses to "insert
the ``model.`` prefix" (see ``state_dict_remap.py``). Renaming any of them turns a strict
load into a silent partial load.

.. warning::
   **Never call ``gradient_checkpointing_enable()`` on the text tower.**
   ``transformers.utils.generic.check_model_inputs`` flips ``use_cache`` to ``False``
   whenever ``gradient_checkpointing and self.training``, and
   ``GradientCheckpointingLayer.__call__`` nulls ``past_key_values`` -- so the prefix
   returns **no cache**, the DiT is fed nothing, and the only signal is a
   ``warning_once``. That is exactly why the reference's own trainer checkpoints the ViT
   wholesale but wraps only each LLM layer's **MLP**. :meth:`run_prefix` raises rather
   than letting a ``None`` cache through.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

from opentau.policies.xr1.dit import XR1DiT, XR1MLPProjector, XR1TimestepEmbedder


class _CheckpointedQwen3VLTextMLP(Qwen3VLTextMLP):
    """``Qwen3VLTextMLP`` whose forward is activation-checkpointed.

    Installed by a **class swap** (``layer.mlp.__class__ = ...``), not by rebinding
    ``layer.mlp.forward``. Rebinding leaves FSDP's pre-forward hook registered on the
    original module while the call bypasses it; a class swap leaves module identity,
    ``_modules``, ``_parameters`` and every hook untouched, and the state-dict keys stay
    ``...mlp.gate_proj.weight``.
    """

    def forward(self, x):  # noqa: D102 - inherited contract
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(super().forward, x, use_reentrant=False)
        return super().forward(x)


class Qwen3VLWithDiT(nn.Module):
    """Backbone + DiT + projectors, named exactly as the reference checkpoint names them."""

    def __init__(
        self,
        qwen3vl_config: Qwen3VLConfig,
        *,
        dit_num_hidden_layers: int,
        dit_hidden_size: int,
        dit_intermediate_size: int,
        dit_num_key_value_heads: int,
        dit_head_dim: int,
        dit_time_embed_dim: int,
        dit_rms_norm_eps: float,
        state_token_dim: int,
        max_action_dim: int,
        attention_implementation: str,
        freeze_input_embeddings: bool = True,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        train_vision_encoder_only: bool = False,
        train_state_action_representation_only: bool = False,
        gradient_checkpointing: bool = False,
        backbone_weights_repo_id: str | None = None,
        enable_choice_heads: bool = False,
        n_choices: int = 5,
    ):
        super().__init__()
        self.freeze_input_embeddings = freeze_input_embeddings
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_vision_encoder_only = train_vision_encoder_only
        self.train_state_action_representation_only = train_state_action_representation_only

        if train_state_action_representation_only and (train_expert_only or train_vision_encoder_only):
            raise ValueError(
                "`train_state_action_representation_only=True` is mutually exclusive with "
                "`train_expert_only=True` and `train_vision_encoder_only=True`."
            )

        qwen3vl_config = copy.deepcopy(qwen3vl_config)
        text_cfg = qwen3vl_config.text_config
        backbone_depth = text_cfg.num_hidden_layers

        # Hard cross-attention constraints: the DiT's keys/values are concatenated with the
        # backbone's cached KV at every layer, so head geometry has to match exactly.
        if dit_head_dim != text_cfg.head_dim:
            raise ValueError(
                f"dit_head_dim ({dit_head_dim}) must equal the backbone head_dim ({text_cfg.head_dim})."
            )
        if dit_num_key_value_heads != text_cfg.num_key_value_heads:
            raise ValueError(
                f"dit_num_key_value_heads ({dit_num_key_value_heads}) must equal the backbone "
                f"num_key_value_heads ({text_cfg.num_key_value_heads})."
            )
        if dit_num_hidden_layers > backbone_depth:
            raise ValueError(
                f"dit_num_hidden_layers ({dit_num_hidden_layers}) exceeds the backbone depth "
                f"({backbone_depth}); DiT layer i reads backbone cache layer start + i."
            )
        self.num_layers = backbone_depth

        text_cfg._attn_implementation = attention_implementation
        qwen3vl_config._attn_implementation = attention_implementation
        if backbone_weights_repo_id is not None:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                backbone_weights_repo_id,
                dtype=torch.bfloat16,
                attn_implementation=attention_implementation,
            )
        else:
            self.vlm = Qwen3VLForConditionalGeneration(qwen3vl_config)

        self.dit = XR1DiT(
            num_hidden_layers=dit_num_hidden_layers,
            hidden_size=dit_hidden_size,
            intermediate_size=dit_intermediate_size,
            head_dim=dit_head_dim,
            num_key_value_heads=dit_num_key_value_heads,
            rms_norm_eps=dit_rms_norm_eps,
        )

        self.state_projector = XR1MLPProjector(
            input_dim=state_token_dim, output_dim=dit_hidden_size, inter_dim=dit_hidden_size, num_layers=2
        )
        self.action_projector = XR1MLPProjector(
            input_dim=max_action_dim, output_dim=dit_hidden_size, inter_dim=dit_hidden_size, num_layers=2
        )
        self.action_output_layer = XR1MLPProjector(
            input_dim=dit_hidden_size, output_dim=max_action_dim, inter_dim=dit_hidden_size, num_layers=2
        )
        self.t_embedder = XR1TimestepEmbedder(dit_hidden_size, frequency_embedding_size=dit_time_embed_dim)
        self.t_projector = XR1MLPProjector(
            input_dim=dit_hidden_size, output_dim=6 * dit_hidden_size, bias=True
        )
        self.sink = nn.Embedding(1, dit_hidden_size)

        # VLM-side auxiliary heads. Off by default: the released RoboCasa365 checkpoint
        # dropped them, so a parity-verified eval must not grow them.
        self.choice_heads = None
        if enable_choice_heads:
            from opentau.policies.xr1.choice_heads import XR1ChoiceHeads

            self.choice_heads = XR1ChoiceHeads(
                hidden_size=text_cfg.hidden_size,
                state_dim=state_token_dim,
                action_dim=max_action_dim,
                n_choices=n_choices,
            )

        # Match the head's dtype to the (possibly bf16) backbone so the cross-attention
        # matmuls share a dtype on GPU; on CPU tests both stay fp32.
        backbone_dtype = next(self.vlm.parameters()).dtype
        for module in (
            self.dit,
            self.state_projector,
            self.action_projector,
            self.action_output_layer,
            self.t_embedder,
            self.t_projector,
            self.sink,
        ):
            module.to(dtype=backbone_dtype)
        if self.choice_heads is not None:
            self.choice_heads.to(dtype=backbone_dtype)

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.enable_gradient_checkpointing()
        self.set_requires_grad()

    # ----- checkpointing -----

    def enable_gradient_checkpointing(self) -> None:
        """Arm the three checkpointing mechanisms the reference trainer uses.

        1. The **ViT** wholesale -- it holds no KV cache and is where the activations are
           (24 blocks x 3 cameras x 4 frames x 256 patches).
        2. Each LLM layer's **MLP only**, by a class swap. Checkpointing the whole text
           layer (or calling ``gradient_checkpointing_enable()`` on the tower) silently
           empties the KV cache; see the module warning.
        3. The **DiT** layers, which emit no cache and can be checkpointed whole.
        """
        self.vlm.model.visual.gradient_checkpointing_enable()
        for layer in self.text_model.layers:
            layer.mlp.__class__ = _CheckpointedQwen3VLTextMLP
        self.dit.gradient_checkpointing = True

    # ----- freezing -----

    def set_requires_grad(self) -> None:
        """Apply the freezing matrix. Called from ``__init__`` and after any flag change."""
        if self.freeze_input_embeddings:
            # 151936 x 2560 ~ 389M parameters that never receive a gradient. Handing them
            # to a fused AdamW would still allocate fp32 master + m + v (~4.7 GB) for them,
            # which is why `get_optim_params` filters on `requires_grad`.
            self.vlm.get_input_embeddings().weight.requires_grad = False
        if self.freeze_vision_encoder:
            self.vlm.model.visual.eval()
            for p in self.vlm.model.visual.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False
        if self.train_vision_encoder_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False
            for module in (
                self.dit,
                self.state_projector,
                self.action_projector,
                self.action_output_layer,
                self.t_embedder,
                self.t_projector,
                self.sink,
            ):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
            for p in self.vlm.model.visual.parameters():
                p.requires_grad = True
        if self.train_state_action_representation_only:
            # Only the robot-shaped projections train: state_projector, action_projector
            # and action_output_layer. Everything else -- backbone, ViT, DiT and the flow
            # time conditioning (t_embedder / t_projector) and the sink embedding -- is
            # frozen, exactly mirroring cosmos3's reading of this mode.
            self.vlm.eval()
            self.dit.eval()
            for p in self.parameters():
                p.requires_grad = False
            for module in (self.state_projector, self.action_projector, self.action_output_layer):
                module.train(self.training)
                for p in module.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):  # noqa: D102 - inherited contract
        super().train(mode)
        if self.train_expert_only:
            self.vlm.eval()
        elif self.train_vision_encoder_only:
            self.vlm.eval()
            self.dit.eval()
            self.vlm.model.visual.train(mode)
        elif self.train_state_action_representation_only:
            self.vlm.eval()
            self.dit.eval()
        elif self.freeze_vision_encoder:
            self.vlm.model.visual.eval()
        return self

    # ----- backbone helpers -----

    @property
    def text_model(self):
        """The Qwen3-VL text tower (``Qwen3VLTextModel``)."""
        return self.vlm.model.language_model

    def get_rope_index(
        self, input_ids: Tensor, video_grid_thw: Tensor | None, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Delegate to the backbone's MRoPE index builder (video branch).

        ``get_rope_index`` splits ``video_grid_thw`` into per-frame ``t = 1`` entries, which
        is exactly XR-1's two-block-per-camera layout. Lifted from
        ``cosmos3/qwen3vl_with_expert.py:521-525``, extended with ``video_grid_thw=``.
        """
        return self.vlm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def compute_rope(
        self, position_ids: Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """MRoPE ``(cos, sin)`` for ``position_ids`` ``(3, B, S)``.

        Reuses the *backbone's* rotary module rather than instantiating a second one as the
        reference does: ``inv_freq`` is a non-persistent, config-derived buffer, so the two
        are numerically identical and one fewer module means one fewer thing to keep in
        sync. From ``cosmos3/qwen3vl_with_expert.py:527-533``.
        """
        dummy = torch.zeros(1, dtype=dtype, device=device)
        return self.text_model.rotary_emb(dummy, position_ids)

    def run_prefix(
        self,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        prefix_len: int | None = None,
        return_hidden_states: bool = False,
    ) -> list[tuple[Tensor, Tensor]] | tuple[list[tuple[Tensor, Tensor]], Tensor]:
        """Run the backbone over the observation prefix; return its per-layer ``(K, V)``.

        Exactly one of ``input_ids`` / ``inputs_embeds`` may be given --
        ``Qwen3VLModel.forward`` raises on both or neither. The ``inputs_embeds`` path
        exists for the Stage-2 choice heads, which have to splice their own embeddings in
        while leaving the ``<|video_pad|>`` rows untouched.

        The forward runs under ``no_grad`` and the cache is detached exactly when **no**
        backbone parameter is trainable; otherwise the graph is kept so the head's loss can
        reach the (unfrozen part of the) backbone -- unfreezing must never be a silent
        no-op. Structure and reasoning from ``cosmos3/qwen3vl_with_expert.py:535-609``.

        Raises:
            RuntimeError: if the backbone returns no cache -- the ``use_cache`` x
                gradient-checkpointing interaction described in the module warning.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds`.")

        backbone_is_frozen = self.train_expert_only or self.train_state_action_representation_only
        ctx = torch.no_grad() if backbone_is_frozen else nullcontext()
        with ctx:
            out = self.vlm.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                use_cache=True,
            )
        pkv = out.past_key_values
        if pkv is None:
            raise RuntimeError(
                "The Qwen3-VL backbone returned `past_key_values=None`, so the DiT would be "
                "conditioned on nothing. This is the documented `use_cache` x gradient-"
                "checkpointing interaction: `transformers.utils.generic.check_model_inputs` "
                "silently sets `use_cache=False` when the text tower has checkpointing enabled "
                "and is in training mode. Do not call `gradient_checkpointing_enable()` on the "
                "text tower — use `Qwen3VLWithDiT.enable_gradient_checkpointing()`, which "
                "checkpoints each LLM layer's MLP only."
            )
        cached = []
        for i in range(self.num_layers):
            key, value = pkv[i]
            if backbone_is_frozen:
                key, value = key.detach(), value.detach()
            if prefix_len is not None:
                # Exclude the choice turn from what the DiT can read. Without this cut the
                # action head can copy the VLM's own action guess through the cache, which
                # the reference's paper attributes a real regression to. The cut is a single
                # uniform slice because the observation prompt is LEFT-padded to a common
                # length before the (fixed-length) choice turn is appended.
                key, value = key[:, :, :prefix_len], value[:, :, :prefix_len]
            cached.append((key, value))
        if return_hidden_states:
            return cached, out.last_hidden_state
        return cached

    # ----- DiT geometry -----

    def dit_position_ids(self, prefix_position_ids: Tensor, prefix_pad: Tensor, query_len: int) -> Tensor:
        """MRoPE positions for the DiT's ``query_len`` tokens: ``arange + per-axis max + 1``.

        The offset is the **per-MRoPE-axis** maximum of the prefix positions -- ``(3, B)``,
        one per temporal/height/width axis. cosmos3 collapses all three with
        ``amax(dim=(0, 2))``, which agrees only when the three axes happen to share a
        maximum; that is a genuine divergence, not a stylistic one.

        Unlike the reference this masks padded columns out of the maximum before reducing.
        ``get_rope_index`` initializes ``position_ids`` to ``1`` and writes real ids only
        where the mask is set, so on an unpadded batch (every fixture, and every batch the
        reference ever ran) the two agree exactly; under padding the masked form is the
        correct one.
        """
        masked = prefix_position_ids.masked_fill(~rearrange(prefix_pad.bool(), "b s -> 1 b s"), -1)
        offset = masked.max(dim=-1).values + 1  # (3, B)
        ar = torch.arange(query_len, device=prefix_position_ids.device)
        return rearrange(ar, "n -> 1 1 n") + rearrange(offset, "three b -> three b 1")

    def dit_attention_mask(self, prefix_pad: Tensor, query_len: int) -> Tensor:
        """``(B, 1, query_len, S_prefix + query_len)`` bool mask.

        Full attention to every valid prefix token, and **causal** among the DiT's own 21
        queries. The causality is worth pinning: cosmos3's suffix is one *bidirectional*
        block, so "fixing" this to bidirectional is a one-character change that trains and
        evaluates without complaint.
        """
        bsize = prefix_pad.shape[0]
        device = prefix_pad.device
        cache_mask = repeat(prefix_pad.bool(), "b s -> b q s", q=query_len)
        self_mask = repeat(
            torch.tril(torch.ones(query_len, query_len, dtype=torch.bool, device=device)),
            "q k -> b q k",
            b=bsize,
        )
        return rearrange(torch.cat([cache_mask, self_mask], dim=-1), "b q k -> b 1 q k")

    # ----- DiT forward -----

    def dit_forward(
        self,
        noisy_action: Tensor,
        t: Tensor,
        action_mask: Tensor,
        state_embed: Tensor,
        position_embeds: tuple[Tensor, Tensor],
        past_key_values: list[tuple[Tensor, Tensor]],
        attn_mask: Tensor,
    ) -> Tensor:
        """One DiT evaluation: ``(noisy_action, tau) -> velocity``.

        Verbatim in structure from ``MiBoTForActionGeneration.dit_forward``. Two details
        that look like noise and are not:

        * ``t[:, 0, 0] * 1000`` multiplies in ``t``'s **own** dtype before the sinusoid
          casts to fp32, so on the real model the bf16 tau is scaled in bf16 (0.2 arrives
          as 0.2001953125 and 200.1953125 rounds to 200.0). Promoting to fp32 first gives a
          different embedding.
        * ``noisy_action * action_mask`` is applied **here**, inside every call.
        """
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        noisy_action = noisy_action * action_mask
        noisy_action = self.action_projector(noisy_action)

        sink = repeat(self.sink.weight, "one d -> b one d", b=state_embed.shape[0])
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        return self.action_output_layer(hidden_states)
