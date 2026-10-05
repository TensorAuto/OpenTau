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

"""The Xiaomi-Robotics-1 DiT action head.

A faithful port of ``MiBoTForActionGeneration``'s DiT tail (``modeling_mibot.py``
lines 1549-1960 of ``XiaomiRobotics/Xiaomi-Robotics-1`` @ ``4da1db0``, Apache-2.0).
Module and parameter names mirror the reference exactly, so loading its checkpoint is a
single ``model.`` prefix insertion (see ``state_dict_remap.py``).

Four details are easy to "clean up" into silent divergence, so they are called out here
and pinned by ``tests/policies/test_xr1_cpu.py``:

1. **The timestep sinusoid is the classic DiT ladder** -- ``1 / 10000 ** (2i/d)`` with
   ``[cos, sin]`` concatenation and a ``t * 1000`` input scale. It is *not*
   ``opentau.policies.cosmos3.modeling_cosmos3.create_sinusoidal_pos_embedding``, which
   uses a ``min_period``/``max_period`` ladder and concatenates ``[sin, cos]``. Reusing
   that helper is the single most tempting wrong reuse in this port.
2. **adaLN semantics.** ``modulate(x, shift, scale) = x * (1 + scale) + shift`` over a
   plain RMSNorm, and the residual is ``x + gate * sublayer(x)`` -- **not**
   ``x + (1 + gate) * sublayer(x)``. The six chunks come from a *shared* projector plus a
   *per-layer* ``adaln_table`` parameter, not from a per-layer Linear.
3. **``noisy_action * action_mask`` is applied inside every DiT call**, not once outside
   the Euler loop. The sampler's own ``x`` keeps non-zero padded columns from step 1
   onward (the output layer writes all 60), so hoisting the mask out of the loop changes
   the DiT's *input* on steps 1-4.
4. **Attention is always SDPA with an explicit boolean mask**, whatever
   ``attention_implementation`` says -- that field selects the *backbone's* kernel only.

Shape convention: this module runs on a flat leading batch ``N``. Training folds the
``training_repeat`` flow timesteps into it (``N = B * R``) rather than carrying a separate
axis, because ``F.scaled_dot_product_attention`` is **not** rank-invariant -- a
``(B, 1, H, L, D)`` call and a ``(B, H, L, D)`` call select different kernels and do not
agree bitwise, which would put the training and inference paths on different numerics.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """adaLN modulation, verbatim from the reference: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply Qwen3-VL MRoPE to ``(N, H, L, D)`` queries/keys.

    Identical arithmetic and operation order to
    ``transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb`` with
    ``unsqueeze_dim=1``; written out only so the head axis is inserted explicitly rather
    than by a magic index.
    """
    cos = rearrange(cos, "n l d -> n 1 l d")
    sin = rearrange(sin, "n l d -> n 1 l d")
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_batch(x: Tensor, batch_size: int) -> Tensor:
    """Interleave-expand ``x``'s batch to ``batch_size``.

    The training path evaluates ``training_repeat`` flow timesteps against **one** VLM
    prefix pass, folding the repeat axis into the batch with
    ``repeat_interleave`` -- so the cached keys/values have to be expanded the same way
    (AABB, not ABAB) or every repeat reads another sample's observation. A no-op at
    inference, where the batches already match.

    Raises:
        ValueError: if ``batch_size`` is not a multiple of ``x``'s batch.
    """
    if x.shape[0] == batch_size:
        return x
    if batch_size % x.shape[0] != 0:
        raise ValueError(f"Cannot repeat a batch of size {x.shape[0]} to {batch_size}.")
    return x.repeat_interleave(batch_size // x.shape[0], dim=0)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand GQA key/value heads: ``(N, n_kv, S, D) -> (N, n_kv * n_rep, S, D)``.

    A no-op at the shipped geometry (1024 / 128 = 8 query heads against 8 KV heads), kept
    so a re-sized DiT stays correct.
    """
    if n_rep == 1:
        return x
    return repeat(x, "n h s d -> n (h r) s d", r=n_rep)


class XR1RMSNorm(nn.Module):
    """RMSNorm matching ``Qwen3VLTextRMSNorm`` (variance accumulated in fp32)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class XR1TimestepEmbedder(nn.Module):
    """Flow-time embedder: classic DiT sinusoid -> ``Linear -> SiLU -> Linear`` (no biases).

    The reference hard-codes ``dtype=torch.bfloat16`` for the sinusoid cast. Here the cast
    follows the module's own weight dtype instead, which is bf16 on the real (GPU) model --
    so GPU numerics are unchanged -- and float32 in CPU tests, where a bf16 cast would
    inject rounding the reference never saw on that path.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @property
    def dtype(self) -> torch.dtype:
        return self.mlp[0].weight.dtype

    def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """``(N,) -> (N, dim)``; ``[cos, sin]`` concatenation, frequencies built in fp32."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t: Tensor) -> Tensor:
        """``(N,) -> (N, 1, hidden_size)``."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb[:, None]


class XR1MLPProjector(nn.Module):
    """The reference ``MLPProjector``: an ``nn.Sequential`` of Linears separated by GELU(tanh).

    Kept structurally identical (``self.layers`` as an ``nn.Sequential``, Linears at even
    indices) because the checkpoint's keys are ``<name>.layers.<even>.weight``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inter_dim: int | None = None,
        num_layers: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"input_dim and output_dim must be positive, got {input_dim} and {output_dim}")
        if num_layers > 1 and (inter_dim is None or inter_dim <= 0):
            raise ValueError(f"inter_dim must be positive when num_layers > 1, got {inter_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inter_dim = inter_dim
        self.bias = bias
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, inter_dim, bias=bias))
            for _ in range(1, num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)])
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class XR1DiTAttention(nn.Module):
    """Fused-QKV attention over ``[cached VLM prefix | DiT queries]``.

    ``qkv_proj`` is a single ``Linear(hidden, 3 * hidden, bias=True)`` -- note the bias,
    which cosmos3's separate bias-free q/k/v projections do not have -- followed by
    per-head RMSNorm on Q and K, MRoPE, then SDPA against the concatenation of the
    backbone's cached keys/values with the DiT's own.
    """

    def __init__(self, hidden_size: int, head_dim: int, num_key_value_heads: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.num_key_value_groups = self.num_heads // num_key_value_heads
        self.dropout = 0.0

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.q_norm = XR1RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = XR1RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_state: Tensor,
        past_key_value: tuple[Tensor, Tensor],
        position_embeds: tuple[Tensor, Tensor],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, q_len, _ = hidden_state.size()
        qkv_states = self.qkv_proj(hidden_state).view(batch_size, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv_states.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeds
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        k_cache, v_cache = past_key_value
        k_cache = repeat_kv(repeat_batch(k_cache, batch_size), self.num_key_value_groups)
        v_cache = repeat_kv(repeat_batch(v_cache, batch_size), self.num_key_value_groups)

        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        return self.o_proj(attn_output)


class XR1DiTMLP(nn.Module):
    """SwiGLU MLP; the reference fixes ``intermediate = 4 * hidden``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class XR1DiTLayer(nn.Module):
    """One DiT block: adaLN-modulated attention + MLP with gated residuals.

    Note the second norm is named ``post_layernorm``, **not**
    ``post_attention_layernorm`` -- the checkpoint's key spelling.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = XR1DiTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = XR1DiTMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.input_layernorm = XR1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_layernorm = XR1RMSNorm(hidden_size, eps=rms_norm_eps)

        # Per-layer adaLN offsets, added to the SHARED time projection before chunking.
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: tuple[Tensor, Tensor],
        position_embeds: tuple[Tensor, Tensor],
        t_embeds: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + t_embeds
        ).chunk(6, dim=1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.attn(hidden_states, past_key_value, position_embeds, attn_mask)
        # Gated residual: `x + gate * sublayer(x)`, NOT `x + (1 + gate) * sublayer(x)`.
        hidden_states = residual + gate_msa * hidden_states

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states

        return hidden_states


class XR1DiT(nn.Module):
    """The 36-layer DiT stack. Layer ``i`` reads cache layer ``start + i``.

    ``start = len(past_key_values) - len(self.layers)`` -- so a DiT shallower than the
    backbone reads the backbone's *last* layers, which is the reference's own convention.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                XR1DiTLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    head_dim=head_dim,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]],
        attn_mask: Tensor,
        position_embeds: tuple[Tensor, Tensor],
        t_embeds: Tensor,
    ) -> Tensor:
        start_index = len(past_key_values) - len(self.layers)
        if start_index < 0:
            raise ValueError(
                f"The DiT has {len(self.layers)} layers but the backbone cache only has "
                f"{len(past_key_values)}; layer i reads cache layer start + i, so the cache must be "
                "at least as deep as the DiT."
            )
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    past_key_values[start_index + i],
                    position_embeds,
                    t_embeds,
                    attn_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    past_key_values[start_index + i],
                    position_embeds,
                    t_embeds,
                    attn_mask=attn_mask,
                )
        return hidden_states
