# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""JointSingleSeq DiT. Module tree matches the production checkpoint layout so safetensors keys load without renaming."""

import importlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

ATTN_MODES = ("torch", "cudnn", "flash")
# ``cudnn``: the cuDNN fused kernel first (the reference trainer forces it), torch's own kernels for
# the shapes cuDNN declines (very short sequences such as the two-token dummy streams).
_CUDNN_FIRST = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
_FLASH_CANDIDATES = (
    (
        "flash_attn.cute.interface",
        "flash-attn 4",
    ),  # CuTe DSL kernels (Blackwell), the reference's varlen kernel
    ("flash_attn_interface", "flash-attn 3"),  # Hopper
    ("flash_attn", "flash-attn 2"),
)
_flash: tuple | None = None
_cu_seqlens_cache: dict[tuple, Tensor] = {}
_order_cache: dict[tuple, tuple[Tensor, Tensor, list[int]]] = {}


def _default_attn_mode() -> str:
    """Default attention backend.

    Plain torch SDPA everywhere; ``cudnn`` (cuDNN fused attention with torch fallbacks) and ``flash``
    (flash-attn 4 / 3 / 2, dense and variable-length kernels) are opt-in speed-ups.
    """
    return "torch"


def flash_kernels() -> tuple:
    """``(flash_attn_func, flash_attn_varlen_func, name)`` of the first flash-attn generation that imports.

    Resolved on first use so that importing this module never pulls in the CuTe DSL.
    """
    global _flash
    if _flash is None:
        errors = []
        for module_name, label in _FLASH_CANDIDATES:
            try:
                module = importlib.import_module(module_name)
                _flash = (module.flash_attn_func, module.flash_attn_varlen_func, label)
                break
            except Exception as e:  # ImportError, or a DSL that fails to initialise on this machine
                errors.append(f"{module_name}: {e!r}")
        else:
            raise RuntimeError("attn_mode='flash' needs flash-attn 4, 3 or 2; " + "; ".join(errors))
    return _flash


def cu_seqlens(lens: Sequence[int], device: torch.device) -> Tensor:
    """``[0, l0, l0 + l1, ...]`` as int32 on ``device``; cached, the same window layout recurs every block."""
    key = (tuple(lens), str(device))
    hit = _cu_seqlens_cache.get(key)
    if hit is None:
        if len(_cu_seqlens_cache) > 64:
            _cu_seqlens_cache.clear()
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        torch.cumsum(torch.as_tensor(list(lens), dtype=torch.int32), 0, out=cu[1:])
        hit = _cu_seqlens_cache[key] = cu.to(device)
    return hit


def window_major_order(
    segments: Sequence[Sequence[int]], device: torch.device
) -> tuple[Tensor, Tensor, list[int]]:
    """Reorder a sequence laid out segment-major into window-major.

    ``segments[s][i]`` is the token count of window ``i`` in segment ``s`` (context, then each content
    stream); the sequence holds segment 0 for all windows, then segment 1, ... Returns ``(order,
    inverse, window_lens)``: ``x[:, order]`` puts every window's tokens together (segment 0 of window 0,
    segment 1 of window 0, ..., then window 1), ``y[:, inverse]`` undoes it, ``window_lens[i]`` counts
    the tokens of window ``i``.
    """
    key = (tuple(tuple(int(n) for n in seg) for seg in segments), str(device))
    hit = _order_cache.get(key)
    if hit is None:
        if len(_order_cache) > 16:
            _order_cache.clear()
        n_windows = len(segments[0])
        if any(len(seg) != n_windows for seg in segments):
            raise ValueError("every segment must list one token count per window")
        starts, base = [], 0
        for seg in segments:
            offsets = [base]
            for n in seg[:-1]:
                offsets.append(offsets[-1] + int(n))
            starts.append(offsets)
            base += int(sum(seg))
        pieces = [
            torch.arange(starts[s][i], starts[s][i] + int(segments[s][i]))
            for i in range(n_windows)
            for s in range(len(segments))
        ]
        order = torch.cat(pieces)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        window_lens = [int(sum(seg[i] for seg in segments)) for i in range(n_windows)]
        hit = _order_cache[key] = (order.to(device), inverse.to(device), window_lens)
    return hit


@dataclass
class JointSingleSeqParams:
    in_channels: dict = field(
        default_factory=lambda: {
            "video": 96,
            "video_cond": 96,
        }
    )
    sequence: dict = field(
        default_factory=lambda: {
            "x_video": "video",
            "x_video_cond": "video_cond",
        }
    )
    vec_in_dim: int | None = 768
    context_in_dim: int = 20480
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    depth_late_blocks: int = 0
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 10000
    mlp_ratio: float = 3.0
    qkv_bias: bool = False
    gate_type: str | None = None
    attn_mode: str = field(default_factory=_default_attn_mode)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0, "RoPE dimension must be even"
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def dense_attention(q: Tensor, k: Tensor, v: Tensor, attn_mode: str) -> Tensor:
    """Full attention within each batch entry: ``(B, H, L, D)`` -> ``(B, L, H * D)``."""
    if attn_mode == "flash":
        flash_attn_func, _, _ = flash_kernels()
        x = flash_attn_func(*(t.transpose(1, 2) for t in (q, k, v)))  # (B, L, H, D)
        x = x[0] if isinstance(x, tuple) else x
        return x.flatten(2)
    if attn_mode == "cudnn":
        if q.shape[-2] == 1:
            # the single-token state stream: cuDNN declines length-1 sequences; PyTorch's own kernels keep
            # the (zero) gradients flowing to the query and key projections like every other mode
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            with sdpa_kernel(_CUDNN_FIRST, set_priority=True):
                x = F.scaled_dot_product_attention(q, k, v)
    elif attn_mode == "torch":
        x = F.scaled_dot_product_attention(q, k, v)
    else:
        raise NotImplementedError(f"attn_mode={attn_mode!r} not supported")
    return rearrange(x, "b h n d -> b n (h d)")


def packed_attention(q: Tensor, k: Tensor, v: Tensor, attn_mode: str, seqlens: Sequence[int]) -> Tensor:
    """Attention within each of the windows packed along ``L``: ``(1, H, L, D)`` -> ``(1, L, H * D)``.

    ``seqlens`` are the windows' token counts in order (``sum == L``); a token attends to its own
    window only. ``flash`` runs the variable-length kernel over the whole sequence. ``torch`` and
    ``cudnn`` batch windows of equal length into one dense call (a reshape when all windows match,
    the usual case: equal canvases and, mostly, equal caption buckets) and run one call per distinct
    length otherwise; the maths is that of separate per-window forwards.
    """
    if q.shape[0] != 1:
        raise ValueError(f"packed windows form one sequence of batch size 1, got {q.shape[0]}")
    lens = [int(n) for n in seqlens]
    _, heads, total, dim = q.shape
    if sum(lens) != total or any(n < 1 for n in lens):
        raise ValueError(f"window token counts {lens} do not add up to the sequence length {total}")
    if attn_mode == "flash":
        _, flash_attn_varlen_func, _ = flash_kernels()
        cu = cu_seqlens(lens, q.device)
        longest = max(lens)
        x = flash_attn_varlen_func(
            *(t[0].transpose(0, 1) for t in (q, k, v)),  # (L, H, D)
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=longest,
            max_seqlen_k=longest,
        )
        x = x[0] if isinstance(x, tuple) else x
        return x.reshape(1, total, heads * dim)
    if attn_mode not in ("torch", "cudnn"):
        raise NotImplementedError(f"attn_mode={attn_mode!r} not supported")
    n_windows = len(lens)
    if len(set(lens)) == 1:
        per = total // n_windows
        qw, kw, vw = (t.reshape(heads, n_windows, per, dim).transpose(0, 1) for t in (q, k, v))
        return dense_attention(qw, kw, vw, attn_mode).reshape(1, total, heads * dim)
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    by_length: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(lens):
        by_length[n].append(i)
    out = q.new_empty(1, total, heads * dim)
    for n, windows in by_length.items():
        group = [
            torch.stack([t[0, :, offsets[i] : offsets[i] + n] for i in windows]) for t in (q, k, v)
        ]  # (n_group, H, n, D)
        result = dense_attention(*group, attn_mode)
        for j, i in enumerate(windows):
            out[:, offsets[i] : offsets[i] + n] = result[j]
    return out


class Attention(nn.Module):
    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mode: str = "cudnn", seqlens: Sequence[int] | None = None
    ) -> Tensor:
        """``(B, H, L, D)`` -> ``(B, L, H * D)``; with ``seqlens`` the windows packed along ``L`` attend separately."""
        if seqlens is not None:
            return packed_attention(q, k, v, attn_mode, seqlens)
        return dense_attention(q, k, v, attn_mode)


# Perturbs the whole call; the two-pass CFG denoise loops enable PAG on the
# uncond forward only, so the CFG combine guides against the perturbed
# prediction.
def apply_pag(attn: Tensor, v: Tensor, pag_strength: float) -> Tensor:
    if pag_strength <= 0.0:
        return attn

    v_passthrough = rearrange(v, "b h n d -> b n (h d)")
    if pag_strength >= 1.0:
        return v_passthrough
    return (1 - pag_strength) * attn + pag_strength * v_passthrough


def _bf16_if_required(module: nn.Module, x: Tensor) -> Tensor:
    if getattr(module, "_requires_bf16_input", False) and x.dtype != torch.bfloat16:
        return x.to(torch.bfloat16)
    return x


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(out_dtype if out_dtype is not None else t.dtype)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        s = self.adaLN_modulation[0](vec)  # type: ignore
        shift_w, scale_w = self.adaLN_modulation[1].weight.chunk(2)

        # x = (1 + scale) * self.norm_final(x) + shift
        # using in place ops and avoid materializing full (seq, 2*hidden) projection to reduce memory
        x = self.norm_final(x)
        x.mul_(F.linear(s, scale_w).add_(1))
        x.add_(F.linear(s, shift_w))
        return self.linear(x)


class ModeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        gate_type: str | None = None,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1
        self.gate_type = gate_type

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        if gate_type is not None:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")
        self.attention = Attention()
        self.attn_mode = "torch"

    def forward(
        self, x: Tensor, pe: Tensor, mod: tuple[Tensor, ...], seqlens: Sequence[int] | None = None
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        q = rearrange(
            self.q_proj(_bf16_if_required(self.q_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        k = rearrange(
            self.k_proj(_bf16_if_required(self.k_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        v = rearrange(
            self.v_proj(_bf16_if_required(self.v_proj, x_mod)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        mlp = self.mlp_in(_bf16_if_required(self.mlp_in, x_mod))

        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        attn = self.attention(q, k, v, self.attn_mode, seqlens)

        output = self.attn_out(_bf16_if_required(self.attn_out, attn)) + self.mlp_out(
            _bf16_if_required(self.mlp_out, self.mlp_act(mlp))
        )
        if self.gate_type is not None:
            output = output * F.silu(self.gate_proj(x_mod))
        return x + mod_gate * output


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        gate_type: str | None = None,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1
        self.gate_type = gate_type

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        if gate_type is not None:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")
        self.attention = Attention()
        self.attn_mode = "torch"

    def forward(
        self,
        x: Tensor,
        ctx: Tensor | None,
        pe: Tensor,
        mod_x: tuple[Tensor, ...],
        mod_ctx: tuple[Tensor, ...] | None,
        pag_strength: float = 0.0,
        seqlens: Sequence[int] | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        """``ctx is None``: ``x`` already holds the whole sequence (context and content of every window,
        window-major, see ``JointSingleSeq.forward``) with its per-token modulation ``mod_x``; the block
        then returns ``(None, x_out)``. ``seqlens`` keeps attention within each packed window."""
        mod_shift_x, mod_scale_x, mod_gate_x = mod_x
        if ctx is None:
            combined = (1 + mod_scale_x) * self.pre_norm(x) + mod_shift_x
        else:
            mod_shift_ctx, mod_scale_ctx, mod_gate_ctx = mod_ctx
            x_mod = (1 + mod_scale_x) * self.pre_norm(x) + mod_shift_x
            ctx_mod = (1 + mod_scale_ctx) * self.pre_norm(ctx) + mod_shift_ctx
            combined = torch.cat((ctx_mod, x_mod), dim=1)

        q = rearrange(
            self.q_proj(_bf16_if_required(self.q_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        k = rearrange(
            self.k_proj(_bf16_if_required(self.k_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        v = rearrange(
            self.v_proj(_bf16_if_required(self.v_proj, combined)),
            "B L (H D) -> B H L D",
            H=self.num_heads,
        )
        mlp = self.mlp_in(_bf16_if_required(self.mlp_in, combined))

        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        attn = self.attention(q, k, v, self.attn_mode, seqlens)
        attn = apply_pag(attn, v, pag_strength)

        output = self.attn_out(_bf16_if_required(self.attn_out, attn)) + self.mlp_out(
            _bf16_if_required(self.mlp_out, self.mlp_act(mlp))
        )
        if self.gate_type is not None:
            output = output * F.silu(self.gate_proj(combined))

        if ctx is None:
            return None, x + mod_gate_x * output
        ctx_out, x_out = torch.split(output, [ctx.shape[1], x.shape[1]], dim=1)
        return ctx + mod_gate_ctx * ctx_out, x + mod_gate_x * x_out


class JointSingleSeq(nn.Module):
    def __init__(self, params: JointSingleSeqParams = JointSingleSeqParams()):
        super().__init__()
        self.params = params

        self.sequence = params.sequence
        for _, v in params.sequence.items():
            assert v in params.in_channels, (params.sequence, params.in_channels)

        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.depth = params.depth
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_late_blocks = params.depth_late_blocks
        self.mlp_ratio = params.mlp_ratio
        self.qkv_bias = params.qkv_bias
        self.gate_type = params.gate_type
        self.attn_mode = params.attn_mode

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.emb_in = nn.ModuleDict(
            {m: nn.Linear(c, params.hidden_size, bias=False) for m, c in params.in_channels.items()}
        )
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size, bias=False)

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size, disable_bias=True)
        self.vector_in: nn.Module = (
            MLPEmbedder(params.vec_in_dim, params.hidden_size, disable_bias=True)
            if params.vec_in_dim is not None
            else nn.Identity()
        )

        content_modalities = sorted(params.in_channels.keys())
        self.early_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modalities
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modalities
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )
        if params.depth_late_blocks > 0:
            self.late_stream_modulations = nn.ModuleDict(
                {
                    **{
                        m: Modulation(params.hidden_size, double=False, disable_bias=True)
                        for m in content_modalities
                    },
                    "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
                }
            )

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        ModeBlock(
                            params.hidden_size,
                            params.num_heads,
                            mlp_ratio=params.mlp_ratio,
                            qkv_bias=params.qkv_bias,
                            gate_type=params.gate_type,
                        )
                        for _ in range(params.depth)
                    ]
                )
                for m in content_modalities
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                ModeBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    gate_type=params.gate_type,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    gate_type=params.gate_type,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if params.depth_late_blocks > 0:
            self.late_content_mode_blocks = nn.ModuleDict(
                {
                    m: nn.ModuleList(
                        [
                            ModeBlock(
                                params.hidden_size,
                                params.num_heads,
                                mlp_ratio=params.mlp_ratio,
                                qkv_bias=params.qkv_bias,
                                gate_type=params.gate_type,
                            )
                            for _ in range(params.depth_late_blocks)
                        ]
                    )
                    for m in content_modalities
                }
            )
            self.late_txt_mode_blocks = nn.ModuleList(
                [
                    ModeBlock(
                        params.hidden_size,
                        params.num_heads,
                        mlp_ratio=params.mlp_ratio,
                        qkv_bias=params.qkv_bias,
                        gate_type=params.gate_type,
                    )
                    for _ in range(params.depth_late_blocks)
                ]
            )

        self.final_layer = nn.ModuleDict(
            {m: LastLayer(params.hidden_size, c) for m, c in params.in_channels.items()}
        )

        # Propagate the selected attention backend to every block (default: torch SDPA,
        # see _default_attn_mode; "flash" and "cudnn" are opt-in).
        for module in self.modules():
            if hasattr(module, "attn_mode"):
                module.attn_mode = params.attn_mode

    def embed_timesteps(self, ts: Tensor, out_dtype: torch.dtype) -> Tensor:
        """Sinusoid + ``time_in`` in the precision of ``time_in``'s own weights, outside autocast.

        The reference trainer keeps this MLP in fp32 under bf16 mixed precision (its own FSDP group)
        and casts the result where the modulations consume it; :func:`shard_dit` does the same, and a
        bf16 inference model computes it in bf16 as before.
        """
        weight_dtype = self.time_in.in_layer.weight.dtype
        with torch.autocast(device_type=ts.device.type, enabled=False):
            vec = self.time_in(timestep_embedding(ts, 256, out_dtype=weight_dtype))
        return vec.to(out_dtype)

    def forward(
        self,
        ctx: Tensor,
        ctx_ids: Tensor,
        vector: Tensor,
        timesteps_ctx: Tensor,
        pag_single_layers: list[int] | None = None,
        pag_strength: float = 1.0,
        seqlens: dict[str, Sequence[int]] | None = None,
        **kwargs: Tensor,
    ) -> dict[str, Tensor]:
        """Denoise every stream given in ``kwargs`` (``x_<stream>``, ``x_<stream>_ids``, ``x_<stream>_timesteps``).

        ``seqlens`` packs several windows into one call of batch size 1: ``seqlens["ctx"]`` and
        ``seqlens["x_<stream>"]`` list the token counts of each window in order, and every stream and the
        context hold the windows' tokens concatenated in that order. Attention then stays within each
        window (mode blocks per stream, the joint blocks over context plus content of a window), so the
        result equals separate per-window forwards while the whole micro-batch runs in one pass: one set of
        FSDP gathers, large matmuls, and captions of different lengths without padding.
        """
        assert vector.ndim == 2, "invalid vector shape"
        batch_size = ctx.shape[0]

        x_by_mod: dict[str, list] = defaultdict(list)
        ids_by_mod: dict[str, list] = defaultdict(list)
        ts_by_mod: dict[str, list] = defaultdict(list)
        lens_by_mod: dict[str, list] = defaultdict(list)
        names_by_mod: dict[str, list] = defaultdict(list)
        for k, v in self.sequence.items():
            if k not in kwargs:
                continue
            assert k + "_ids" in kwargs and k + "_timesteps" in kwargs, "missing stream metadata"
            x_by_mod[v].append(kwargs[k])
            ids_by_mod[v].append(kwargs[k + "_ids"])
            ts_by_mod[v].append(kwargs[k + "_timesteps"])
            lens_by_mod[v].append(kwargs[k].shape[1])
            names_by_mod[v].append(k)

        active = list(x_by_mod.keys())
        all_mods = sorted(self.in_channels.keys())
        inactive = [m for m in all_mods if m not in active]
        dummy_len = 2  # attention kernels need >=2
        packed = seqlens is not None
        mod_lens: dict[str, list[int]] = {}
        ctx_lens: list[int] | None = None
        if packed:
            if batch_size != 1:
                raise ValueError(f"packed windows form one sequence of batch size 1, got {batch_size}")
            ctx_lens = [int(n) for n in seqlens["ctx"]]
            if sum(ctx_lens) != ctx.shape[1]:
                raise ValueError(f"context token counts {ctx_lens} do not add up to {ctx.shape[1]}")
            for m in active:
                if len(names_by_mod[m]) != 1:
                    raise ValueError(
                        f"packed windows need one stream per modality, {m} has {names_by_mod[m]}"
                    )
                lens = [int(n) for n in seqlens[names_by_mod[m][0]]]
                if len(lens) != len(ctx_lens) or sum(lens) != lens_by_mod[m][0]:
                    raise ValueError(
                        f"{names_by_mod[m][0]}: window token counts {lens} do not match the stream"
                    )
                mod_lens[m] = lens

        device = x_by_mod[active[0]][0].device
        dtype = x_by_mod[active[0]][0].dtype
        for k in all_mods:
            if k in active:
                x_by_mod[k] = torch.cat(x_by_mod[k], 1)
                ids_by_mod[k] = torch.cat(ids_by_mod[k], 1)
                ts_by_mod[k] = torch.cat(ts_by_mod[k], 1)
            else:
                x_by_mod[k] = torch.zeros(
                    batch_size,
                    dummy_len,
                    self.in_channels[k],
                    device=device,
                    dtype=dtype,
                )
                ids_by_mod[k] = torch.zeros(batch_size, dummy_len, 4, device=device, dtype=torch.int64)
                ts_by_mod[k] = torch.zeros(batch_size, dummy_len, device=device, dtype=dtype)

        vec_by_mod: dict[str, Tensor] = {}
        for k in all_mods:
            ts = rearrange(ts_by_mod[k], "b t -> (b t)")
            vec_by_mod[k] = rearrange(self.embed_timesteps(ts, dtype), "(b t) d -> b t d", b=batch_size)
        ts_ctx = rearrange(timesteps_ctx, "b t -> (b t)")
        vec_ctx = rearrange(self.embed_timesteps(ts_ctx, dtype), "(b t) d -> b t d", b=batch_size)

        vector_emb = self.vector_in(vector)
        for k in all_mods:
            vec_by_mod[k] = vec_by_mod[k] + vector_emb[:, None, :]
        vec_ctx = vec_ctx + vector_emb[:, None, :]

        early_content_mods: dict[str, tuple] = {}
        for key in self.early_stream_modulations:
            if key == "txt":
                continue
            mod, _ = self.early_stream_modulations[key](vec_by_mod[key])
            early_content_mods[key] = mod
        early_txt_mod, _ = self.early_stream_modulations["txt"](vec_ctx)

        single_content_mods: dict[str, list] = {}
        for key in self.single_stream_modulations:
            if key == "txt":
                continue
            mod, _ = self.single_stream_modulations[key](vec_by_mod[key])
            single_content_mods[key] = list(mod)
        single_txt_mod, _ = self.single_stream_modulations["txt"](vec_ctx)

        if self.depth_late_blocks > 0:
            late_content_mods: dict[str, tuple] = {}
            for key in self.late_stream_modulations:
                if key == "txt":
                    continue
                mod, _ = self.late_stream_modulations[key](vec_by_mod[key])
                late_content_mods[key] = mod
            late_txt_mod, _ = self.late_stream_modulations["txt"](vec_ctx)

        content_by_mod = {k: self.emb_in[k](x_by_mod[k]) for k in all_mods}
        txt = self.txt_in(ctx)

        pe_by_mod = {k: self.pe_embedder(ids_by_mod[k]) for k in all_mods}
        pe_ctx = self.pe_embedder(ctx_ids)

        for d in range(self.depth):
            for m in all_mods:
                content_by_mod[m] = self.content_mode_blocks[m][d](
                    content_by_mod[m], pe_by_mod[m], early_content_mods[m], seqlens=mod_lens.get(m)
                )
            txt = self.txt_mode_blocks[d](txt, pe_ctx, early_txt_mod, seqlens=ctx_lens)

        img = torch.cat([content_by_mod[k] for k in active], dim=1)
        content_mod = tuple(torch.cat([single_content_mods[k][i] for k in active], dim=1) for i in range(3))
        pe_content = torch.cat([pe_by_mod[k] for k in active], dim=2)
        pe = torch.cat((pe_ctx, pe_content), dim=2)

        if not packed:
            for i, block in enumerate(self.single_blocks):
                layer_pag = pag_strength if (pag_single_layers and i in pag_single_layers) else 0.0
                txt, img = block(
                    img,
                    txt,
                    pe,
                    content_mod,
                    single_txt_mod,
                    pag_strength=layer_pag,
                )
        else:
            # The joint sequence is laid out context-then-content; gather it window-major once so each
            # window is one contiguous segment for attention, run the blocks on that single sequence with
            # its per-token modulation, and scatter back for the per-stream heads.
            order, inverse, window_lens = window_major_order(
                [ctx_lens, *(mod_lens[k] for k in active)], img.device
            )
            seq = torch.cat((txt, img), dim=1)[:, order]
            mod = tuple(torch.cat((single_txt_mod[i], content_mod[i]), dim=1)[:, order] for i in range(3))
            pe = pe[:, :, order]
            for i, block in enumerate(self.single_blocks):
                layer_pag = pag_strength if (pag_single_layers and i in pag_single_layers) else 0.0
                _, seq = block(seq, None, pe, mod, None, pag_strength=layer_pag, seqlens=window_lens)
            txt, img = torch.split(seq[:, inverse], [txt.shape[1], img.shape[1]], dim=1)

        img_by_mod: dict[str, Tensor] = {}
        pos = 0
        for k in active:
            length = sum(lens_by_mod[k])
            img_by_mod[k] = img[:, pos : pos + length]
            pos += length
        for k in inactive:
            img_by_mod[k] = content_by_mod[k]

        if self.depth_late_blocks > 0:
            for d in range(self.depth_late_blocks):
                for m in all_mods:
                    img_by_mod[m] = self.late_content_mode_blocks[m][d](
                        img_by_mod[m], pe_by_mod[m], late_content_mods[m], seqlens=mod_lens.get(m)
                    )
                txt = self.late_txt_mode_blocks[d](txt, pe_ctx, late_txt_mod, seqlens=ctx_lens)

        final = {k: self.final_layer[k](img_by_mod[k], vec_by_mod[k]) for k in all_mods}
        out: dict[str, Tensor] = {}
        for k in active:
            pos = 0
            for name, length in zip(names_by_mod[k], lens_by_mod[k], strict=True):
                out[name] = final[k][:, pos : pos + length]
                pos += length
        return out
