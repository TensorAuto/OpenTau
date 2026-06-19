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

"""Qwen3-VL backbone + custom flow-matching action expert (cosmos3).

This is the cosmos3 analog of ``paligemma_with_expert.py`` / ``gemma3_with_expert.py``,
adapted to the Qwen3-VL architecture and to a **frozen** backbone.

Design (π0.5 dual-stream, specialized for a frozen reasoner)
-----------------------------------------------------------
In the π0.5 recipe the backbone encodes images + language *once* (the "prefix")
and the action expert cross-attends to the backbone's per-layer key/value cache
to denoise the action chunk (the "suffix"). The two streams never run in the same
forward pass — the prefix forward populates the KV cache; the suffix forward only
runs the expert, reading that cache. The expert reads the backbone's keys/values
but the backbone never reads the expert.

That structure lets cosmos3 run the **entire stock** ``Qwen3VLModel.forward`` as a
black box for the prefix (``Qwen3VLWithExpertModel.run_prefix``). Stock transformers
handles vision encoding, image-token scatter, deepstack injection, the 3-D
multimodal RoPE (MRoPE), QK-norm and the native causal mask — so the frozen
Cosmos-Reason2 reasoner behaves exactly as trained, with zero reimplementation of
the 32B backbone. Only the small (<1B) action expert is hand-written here.

Hard cross-attention constraints (validated at build time)
----------------------------------------------------------
The expert's per-layer keys/values are concatenated with the backbone's cached KV,
so the expert's ``num_key_value_heads`` and ``head_dim`` **must** equal the backbone
text tower's (8 / 128 for Qwen3-VL-32B), and the expert must have the same number of
layers (64) as the backbone so layer ``i`` of the expert reads layer ``i`` of the
cache. The expert's *query* head count is free (any multiple of the KV head count)
because the backbone's queries are never consumed here. The shared MRoPE ``(cos, sin)``
are produced by the backbone's ``rotary_emb`` and reused by the expert, which is why
``expert_head_dim`` must match.
"""

from contextlib import nullcontext

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand GQA key/value heads. ``x`` is (B, n_kv, S, hd) -> (B, n_kv*n_rep, S, hd)."""
    if n_rep == 1:
        return x
    return repeat(x, "b h s d -> b (h r) s d", r=n_rep)


class ExpertRMSNorm(nn.Module):
    """Plain RMSNorm used for the expert's per-head QK normalization.

    Mirrors ``Qwen3VLTextRMSNorm`` (variance in fp32, learned ``weight``), kept as a
    standalone module so the frozen backbone's norm class is never monkey-patched.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype)).to(dtype)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm (DiT adaLN-Zero style), conditioned on the flow-matching time.

    From the conditioning vector ``cond`` (the time embedding) a single dense layer
    produces per-channel ``(scale, shift, gate)``; the normalized input is modulated
    as ``norm(x) * (1 + scale) + shift`` and the ``gate`` is returned for the gated
    residual ``x + gate * sublayer(x)``. The dense layer is **zero-initialized** so
    the expert starts as an exact identity/no-op residual on top of the frozen
    backbone — the stable adaLN-Zero initialization.

    Mirrors ``opentau.utils.transformers_patch.PatchedGemmaRMSNorm`` (the AdaRMS used
    by the pi05/pi06 Gemma action experts) but as a standalone Qwen3 variant.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _norm(self, x: Tensor) -> Tensor:
        var = torch.mean(x.float().square(), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        scale, shift, gate = torch.chunk(self.dense(cond), 3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class Qwen3ExpertMLP(nn.Module):
    """Qwen3 gated-SiLU MLP (gate/up/down), matching ``Qwen3VLTextMLP``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3ExpertAttention(nn.Module):
    """Expert self/cross attention: expert queries over ``[cached_backbone_KV ; expert_KV]``.

    Matches ``Qwen3VLTextAttention`` (QK-norm on the head dim before RoPE, GQA, scaling
    ``head_dim**-0.5``) but the keys/values are *prefixed* with the frozen backbone's
    cached, already-RoPE'd KV for this layer, so the expert cross-attends to the whole
    observation prefix as well as to the action chunk.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_implementation: str,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.n_rep = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.attention_implementation = attention_implementation

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = ExpertRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = ExpertRMSNorm(head_dim, eps=rms_norm_eps)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        """q (B,Hq,Sq,hd); k/v (B,Hkv,Sk,hd); attn_mask bool (B,Sq,Sk) True=attend."""
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        bsz, _, sq, _ = q.shape
        mask = rearrange(attn_mask, "b sq sk -> b 1 sq sk")

        if self.attention_implementation == "sdpa":
            out = nn.functional.scaled_dot_product_attention(
                q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask, dropout_p=0.0, scale=self.scaling
            )
        else:  # eager, fp32 scores for stability (matches pi05/pi06)
            qf = q.to(torch.float32)
            kf = k.to(torch.float32)
            attn = torch.matmul(qf, kf.transpose(2, 3)) * self.scaling
            attn = torch.where(mask, attn, torch.finfo(torch.float32).min)
            probs = nn.functional.softmax(attn, dim=-1).to(v.dtype)
            out = torch.matmul(probs, v)
        return rearrange(out, "b h s d -> b s (h d)", b=bsz, s=sq)

    def forward(
        self,
        hidden: Tensor,
        cached_kv: tuple[Tensor, Tensor],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        b, s, _ = hidden.shape
        q = self.q_norm(rearrange(self.q_proj(hidden), "b s (h d) -> b s h d", d=self.head_dim))
        k = self.k_norm(rearrange(self.k_proj(hidden), "b s (h d) -> b s h d", d=self.head_dim))
        v = rearrange(self.v_proj(hidden), "b s (h d) -> b s h d", d=self.head_dim)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        cached_k, cached_v = cached_kv
        k = torch.cat([cached_k.to(k.dtype), k], dim=2)
        v = torch.cat([cached_v.to(v.dtype), v], dim=2)

        out = self._attend(q, k, v, attn_mask)
        return self.o_proj(out.to(hidden.dtype))


class Qwen3ExpertDecoderLayer(nn.Module):
    """One expert decoder layer: AdaRMS pre-norms + gated residuals around attn / MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        adarms_cond_dim: int,
        rms_norm_eps: float,
        dropout: float,
        attention_implementation: str,
    ):
        super().__init__()
        self.input_layernorm = AdaRMSNorm(hidden_size, adarms_cond_dim, eps=rms_norm_eps)
        self.post_attention_layernorm = AdaRMSNorm(hidden_size, adarms_cond_dim, eps=rms_norm_eps)
        self.self_attn = Qwen3ExpertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_implementation=attention_implementation,
        )
        self.mlp = Qwen3ExpertMLP(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        cached_kv: tuple[Tensor, Tensor],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
        adarms_cond: Tensor,
    ) -> Tensor:
        normed, gate_attn = self.input_layernorm(hidden, adarms_cond)
        attn_out = self.self_attn(normed, cached_kv, cos, sin, attn_mask)
        hidden = hidden + gate_attn * self.dropout(attn_out)

        normed, gate_mlp = self.post_attention_layernorm(hidden, adarms_cond)
        mlp_out = self.mlp(normed)
        hidden = hidden + gate_mlp * self.dropout(mlp_out)
        return hidden


class Qwen3ActionExpert(nn.Module):
    """The trainable flow-matching action expert: a stack of AdaRMS Qwen3 decoder layers.

    Operates on action-token embeddings (the suffix). Each layer ``i`` cross-attends to
    the frozen backbone's cached KV for layer ``i`` plus the action chunk itself.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        adarms_cond_dim: int,
        rms_norm_eps: float,
        dropout: float,
        attention_implementation: str,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Qwen3ExpertDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    adarms_cond_dim=adarms_cond_dim,
                    rms_norm_eps=rms_norm_eps,
                    dropout=dropout,
                    attention_implementation=attention_implementation,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = AdaRMSNorm(hidden_size, adarms_cond_dim, eps=rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden: Tensor,
        cached_kv: list[tuple[Tensor, Tensor]],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
        adarms_cond: Tensor,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, cached_kv[i], cos, sin, attn_mask, adarms_cond, use_reentrant=False
                )
            else:
                hidden = layer(hidden, cached_kv[i], cos, sin, attn_mask, adarms_cond)
        hidden, _ = self.norm(hidden, adarms_cond)
        return hidden


class Qwen3VLWithExpertModel(nn.Module):
    """A frozen Qwen3-VL backbone paired with a trainable flow-matching action expert."""

    def __init__(
        self,
        qwen3vl_config: Qwen3VLConfig,
        *,
        expert_hidden_size: int,
        expert_intermediate_size: int,
        expert_num_hidden_layers: int,
        expert_num_attention_heads: int,
        expert_num_key_value_heads: int,
        expert_head_dim: int,
        expert_adarms_cond_dim: int,
        expert_rms_norm_eps: float,
        dropout: float,
        attention_implementation: str,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        gradient_checkpointing: bool = False,
        load_pretrained_backbone_repo: str | None = None,
    ):
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        text_cfg = qwen3vl_config.text_config
        # Hard cross-attention constraints (see module docstring).
        if expert_head_dim != text_cfg.head_dim:
            raise ValueError(
                f"expert_head_dim ({expert_head_dim}) must equal the backbone head_dim ({text_cfg.head_dim})."
            )
        if expert_num_key_value_heads != text_cfg.num_key_value_heads:
            raise ValueError(
                f"expert_num_key_value_heads ({expert_num_key_value_heads}) must equal the backbone "
                f"num_key_value_heads ({text_cfg.num_key_value_heads})."
            )
        if expert_num_hidden_layers != text_cfg.num_hidden_layers:
            raise ValueError(
                f"expert_num_hidden_layers ({expert_num_hidden_layers}) must equal the backbone "
                f"num_hidden_layers ({text_cfg.num_hidden_layers}) so each expert layer reads the matching "
                "backbone KV cache layer."
            )
        self.num_layers = text_cfg.num_hidden_layers

        qwen3vl_config.text_config._attn_implementation = attention_implementation
        if load_pretrained_backbone_repo is not None:
            self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                load_pretrained_backbone_repo,
                dtype=torch.bfloat16,
                attn_implementation=attention_implementation,
            )
        else:
            self.backbone = Qwen3VLForConditionalGeneration(qwen3vl_config)

        self.expert = Qwen3ActionExpert(
            num_hidden_layers=expert_num_hidden_layers,
            hidden_size=expert_hidden_size,
            intermediate_size=expert_intermediate_size,
            num_attention_heads=expert_num_attention_heads,
            num_key_value_heads=expert_num_key_value_heads,
            head_dim=expert_head_dim,
            adarms_cond_dim=expert_adarms_cond_dim,
            rms_norm_eps=expert_rms_norm_eps,
            dropout=dropout,
            attention_implementation=attention_implementation,
        )
        self.expert.gradient_checkpointing = gradient_checkpointing

        # Match the expert dtype to the (possibly bf16-loaded) backbone so cross-attention
        # matmuls share a dtype on GPU; on CPU tests both stay fp32.
        backbone_dtype = next(self.backbone.parameters()).dtype
        self.expert.to(dtype=backbone_dtype)

        self.set_requires_grad()

    # ----- freezing / dtype plumbing -----

    def set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.backbone.model.visual.eval()
            for p in self.backbone.model.visual.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_expert_only:
            self.backbone.eval()
        elif self.freeze_vision_encoder:
            self.backbone.model.visual.eval()
        return self

    # ----- backbone helpers -----

    @property
    def text_model(self):
        return self.backbone.model.language_model

    def get_rope_index(self, input_ids, image_grid_thw, attention_mask):
        return self.backbone.model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

    def compute_rope(
        self, position_ids: Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Compute MRoPE ``(cos, sin)`` for ``position_ids`` (3, B, S) using the backbone rotary."""
        dummy = torch.zeros(1, dtype=dtype, device=device)
        return self.text_model.rotary_emb(dummy, position_ids)

    def run_prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Run the backbone over the observation prefix; return the per-layer (K, V) cache.

        Each entry is ``(key, value)`` of shape ``(B, num_kv_heads, S_prefix, head_dim)``.

        When ``train_expert_only`` (the default), the backbone is fully frozen: the forward
        runs under ``no_grad`` and the cached KV is ``.detach()``'d, so the expert reads it
        as a constant. When ``train_expert_only`` is False (partial unfreeze, e.g. a
        trainable text tower with a frozen vision encoder), the forward keeps its graph and
        the KV is **not** detached, so the expert's loss backpropagates into the (unfrozen)
        backbone — otherwise unfreezing would be a silent no-op.
        """
        ctx = torch.no_grad() if self.train_expert_only else nullcontext()
        with ctx:
            out = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True,
            )
        pkv = out.past_key_values
        cached = []
        for i in range(self.num_layers):
            key, value = pkv[i]
            if self.train_expert_only:
                key, value = key.detach(), value.detach()
            cached.append((key, value))
        return cached

    def run_expert(
        self,
        action_embs: Tensor,
        cached_kv: list[tuple[Tensor, Tensor]],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor,
        adarms_cond: Tensor,
    ) -> Tensor:
        return self.expert(action_embs, cached_kv, cos, sin, attn_mask, adarms_cond)
