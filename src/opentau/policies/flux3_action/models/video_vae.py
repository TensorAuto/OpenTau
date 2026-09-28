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
"""Video VAE (ViTNormInference). Swin3D + neighborhood attention with built-in DistributedRunningStats normalization."""

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from natten.functional import na2d, na3d
from torch import Tensor

logger = logging.getLogger(__name__)

_norm_layer = partial(nn.LayerNorm, eps=1e-5)

# NATTEN neighborhood-attention backend, resolved at first use per tensor
# layout (2D/3D). Prebuilt NATTEN wheels only carry fused CUTLASS kernels for
# some GPU architectures, so the fastest supported backend is probed at
# runtime: blackwell (SM100) > hopper (SM90) > generic CUTLASS > flex-attention
# (pure torch, runs everywhere at a speed/memory cost). Set $F3_NATTEN_BACKEND
# (blackwell-fna | hopper-fna | cutlass-fna | flex-fna) to override.
_NATTEN_BACKEND_ENV = "F3_NATTEN_BACKEND"
_PERSISTENT_KERNEL_BACKENDS = ("blackwell", "hopper")
_natten_backends: dict[int, str] = {}


def _natten_attention_kwargs(
    q: Tensor, k: Tensor, v: Tensor, *, kernel_size: list[int], is_causal: list[bool] | None = None
) -> dict:
    backend = _natten_backends.get(q.ndim)
    if backend is None:
        backend = os.environ.get(_NATTEN_BACKEND_ENV)
        if not backend:
            from natten import backends as nb

            if nb.can_run_cutlass_blackwell_fna(q, k, v):
                backend = "blackwell-fna"
            elif nb.can_run_cutlass_hopper_fna(q, k, v):
                backend = "hopper-fna"
            elif nb.can_run_cutlass_fna(q, k, v):
                backend = "cutlass-fna"
            else:
                backend = "flex-fna"  # universal fallback, no compiled kernels needed
        logger.info("natten backend (%dD tokens): %s", q.ndim - 3, backend)
        _natten_backends[q.ndim] = backend
    # A window that covers the whole input along every non-causal axis is plain attention to NATTEN, which
    # then dispatches to its dense "*-fmha" backends instead of the neighborhood "*-fna" ones.
    na_dim = q.ndim - 3
    causal = is_causal or [False] * na_dim
    if all(kk == s and not c for kk, s, c in zip(kernel_size, q.shape[1 : 1 + na_dim], causal, strict=True)):
        backend = backend.replace("-fna", "-fmha")
    kwargs = {"backend": backend}
    if backend.split("-")[0] in _PERSISTENT_KERNEL_BACKENDS:
        kwargs["run_persistent_kernel"] = True
    return kwargs


@dataclass
class ViTNormInferenceParams:
    z_dim: int = 96
    embed_dim: int = 256
    patch_size: list[int] = field(default_factory=lambda: [1, 4, 4])
    window_size: list[int] = field(default_factory=lambda: [5, 5, 5])
    enc_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    dec_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    num_heads: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    temporal: list[bool] = field(default_factory=lambda: [False, False, True, True])
    enc_causal: bool = True
    dec_causal: bool = False
    qk_norm: bool = True
    patch_norm: bool = False
    dtype: str = "bfloat16"
    use_compile: bool = True
    compile_config: dict | None = field(default_factory=lambda: {"dynamic_compile": True})
    compile_decoder: bool = False
    chunked_encode: bool = False
    chunk_size_frames: int = 45
    chunked_decode: bool = False
    chunk_size_latent_frames: int = 8
    chunk_overlap_latent_frames: int = 4
    # Looped decode: every decoder Swin block runs its attention and MLP over temporal windows of at
    # most this many frames (plus the attention halo) instead of the whole sequence. Numerically the
    # full decode, at a fraction of the activation memory. None disables it.
    decoder_max_t: int | None = None


# Full-decode peak ~= ELEM * latent.numel() + FIXED, profiled with the
# compiled decoder (calibrated at 720p, T_lat 8..64). Linear in latent elements,
# so it holds across resolutions. Used to pick full vs chunked decode at runtime.
DECODE_BYTES_PER_LATENT_ELEM = 1.35e6
DECODE_FIXED_BYTES = 2 * 2**30
DECODE_SAFETY_FACTOR = 1.25


class DistributedRunningStats(nn.Module):
    """Latent mean and variance of the trained VAE (buffers of the checkpoint); normalizes ``(B, C, ...)``."""

    def __init__(self, num_channels: int, device: str | torch.device | None = None):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device))
        self.register_buffer("running_var", torch.ones(num_channels, device=device))
        self.register_buffer("initialized", torch.tensor(False, device=device))

    def _shape(self, x: Tensor) -> tuple:
        return (1, -1) + (1,) * (x.dim() - 2)

    def normalize(self, x: Tensor) -> Tensor:
        s = self._shape(x)
        return (x - self.running_mean.view(s)) / self.running_var.sqrt().view(s)

    def denormalize(self, x: Tensor) -> Tensor:
        s = self._shape(x)
        return x * self.running_var.sqrt().view(s) + self.running_mean.view(s)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        B, D, H, W, C = x.shape  # noqa: N806
        x = x.reshape(B, D, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).flatten(4)
        return self.reduction(self.norm(x))


class TemporalMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        if D % 2 == 1:
            x = torch.concat([x[:, :1], x], dim=1)
        B, D, H, W, C = x.shape  # noqa: N806
        x = x.reshape(B, D // 2, 2, H, W, C)
        skip = x.mean(2)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(B, D // 2, H, W, 2 * C)
        return self.reduction(self.norm(x)) + skip


class PatchExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 4 * out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        assert self.dim == C, "patch channel mismatch"
        x = self.expansion(self.norm(x))
        x = x.view(B, D, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(B, D, H * 2, W * 2, self.out_dim)


class TemporalExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 2 * out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        assert self.dim == C, "temporal channel mismatch"
        x = self.expansion(self.norm(x)) + torch.concat([x, x], -1)
        x = x.view(B, D, H, W, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(B, D * 2, H, W, self.out_dim)
        return x[:, 1:]


def _compute_pad_size_3d(
    size_dhw: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    pad = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad[0], pad[1], pad[2]


torch.fx.wrap("_compute_pad_size_3d")


def _temporal_core_halo(
    core_start: int, core_end: int, length: int, kernel: int, causal: bool
) -> tuple[int, int]:
    """Frame range ``[halo_start, halo_end)`` of a length-``length`` sequence that a looped Swin block must
    attend over to reproduce frames ``[core_start, core_end)`` of the full-sequence output exactly.

    Neighborhood attention gives frame ``i`` the keys ``[start(i), start(i) + kernel)`` with
    ``start(i) = clamp(i - kernel // 2, 0, length - kernel)`` (NATTEN shifts the window inward at the
    edges), or ``[max(0, i - kernel + 1), i]`` when causal. Running the attention on the halo alone applies
    the same clamping relative to the halo, and the halo is chosen so both agree for every core frame.
    """
    if causal:
        return max(0, core_start - kernel + 1), core_end

    def window_start(index: int) -> int:
        return min(max(index - kernel // 2, 0), length - kernel)

    return window_start(core_start), window_start(core_end - 1) + kernel


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, head_dim: int, base: float = 256.0):
        super().__init__()
        assert head_dim % 8 == 0, "head dimension must be divisible by 8"
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 4

        axis_inv_freq = 1.0 / (base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim))
        inv_freq = torch.stack(
            [axis_inv_freq, axis_inv_freq, axis_inv_freq, torch.zeros(self.chunk_dim // 2)]
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q: Tensor, k: Tensor, *, temporal_offset: int | Tensor = 0) -> tuple[Tensor, Tensor]:
        """Rotate ``q`` and ``k`` ``(B, T, H, W, heads, head_dim)``. ``temporal_offset`` is the absolute time
        of frame 0, so a window cut out of a longer sequence gets the positions it has in that sequence
        (a 0-d tensor keeps a compiled looped core from recompiling per window)."""
        _, t, h, w, _, _ = q.shape
        device = q.device
        dtype = q.dtype
        grids = torch.meshgrid(
            torch.arange(t, device=device, dtype=torch.float32) + temporal_offset,
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
        freqs = torch.einsum("...a,af->...af", pos, self.inv_freq.float())
        freqs = freqs.reshape(1, t, h, w, 1, -1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        # Rotate in bf16 rather than upcasting q/k to fp32, which avoids large
        # fp32 transients during decode. cos/sin keep full precision from fp32
        # freqs and cast down only before the products.
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class Natten3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: list[int],
        num_heads: int,
        causal: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert len(window_size) == 3, "expected 3D window"
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionEmbedding3D(self.head_dim)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, h, w, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward_region(
        self, x: Tensor, *, temporal_offset: Tensor, output_start: int, output_end: int
    ) -> Tensor:
        """Attention over the temporal halo ``x`` ``(B, T', H, W, C)`` of a longer sequence, returning only
        its frames ``[output_start, output_end)``. ``x`` must contain the whole neighborhood of every
        returned frame and be at least the temporal kernel long (see ``_temporal_core_halo``); the frames
        are rotated with their absolute times, ``temporal_offset`` being the time of halo frame 0.
        """
        b, t, h, w, c = x.shape
        if not 0 <= output_start < output_end <= t:
            raise ValueError(f"invalid output frames [{output_start}, {output_end}) for a halo of T={t}")
        if t < self.window_size[0]:
            raise ValueError(f"halo T={t} is shorter than the temporal kernel {self.window_size[0]}")
        q, k, v = self._qkv(x)
        q, k = self.rope(q, k, temporal_offset=temporal_offset)
        out = self._attend(q, k, v)[:, output_start:output_end]
        return self.proj(out.reshape(b, output_end - output_start, h, w, c))

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        q, k, v = self._qkv(x)
        q, k = self.rope(q, k)
        return self.proj(self._attend(q, k, v).reshape(b, t, h, w, c))

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Neighborhood attention over ``(B, T, H, W, heads, head_dim)``; a single frame uses the 2-D kernel."""
        if q.shape[1] == 1:
            q2, k2, v2 = q.squeeze(1), k.squeeze(1), v.squeeze(1)
            kernel = self.window_size[1:]
            kwargs = _natten_attention_kwargs(q2, k2, v2, kernel_size=kernel)
            return na2d(q2, k2, v2, kernel_size=kernel, attention_kwargs=kwargs).unsqueeze(1)
        causal = [self.causal, False, False]
        kwargs = _natten_attention_kwargs(q, k, v, kernel_size=self.window_size, is_causal=causal)
        return na3d(q, k, v, is_causal=causal, kernel_size=self.window_size, attention_kwargs=kwargs)


class GLU_MLP(nn.Module):  # noqa: N801 (vendored name)
    def __init__(self, dim: int, align_to: int = 64):
        super().__init__()
        hidden_dim = align_to * ((int(dim * 8 / 3) + align_to - 1) // align_to)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SwinTransformerBlock(nn.Module):
    """Pre-norm neighborhood-attention block.

    With ``max_t`` set, the block runs "looped": it produces its output in temporal windows of at most
    ``max_t`` frames, attending over each window's halo (``_temporal_core_halo``) and applying the MLP to
    the window only. The result equals the plain forward frame for frame, but the attention and MLP
    activations are materialized for ``max_t`` plus a few frames at a time instead of the whole sequence,
    which is what bounds the memory of decoding long videos. The output is written into the input tensor
    in place, so looped mode requires gradients to be disabled; it is meant for the frozen decoder.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        causal: bool = False,
        qk_norm: bool = False,
        max_t: int | None = None,
    ):
        super().__init__()
        if max_t is not None and max_t < window_size[0]:
            # A window shorter than the kernel would let a halo reach back past the previous window,
            # whose output is already written; see _forward_looped.
            raise ValueError(f"max_t={max_t} must be at least the temporal kernel {window_size[0]}")
        self.norm1 = _norm_layer(dim)
        self.attn = Natten3D(dim, window_size, num_heads, causal=causal, qk_norm=qk_norm)
        self.norm2 = _norm_layer(dim)
        self.mlp = GLU_MLP(dim)
        self.max_t = max_t
        self._compiled_looped_core: Callable[..., Tensor] | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.max_t is None or x.shape[1] <= self.max_t:
            return self._forward_full(x)  # one window covers the sequence (also single-frame latents)
        return self._forward_looped(x)

    def _forward_full(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _forward_looped_core(
        self,
        normalized_halo: Tensor,
        residual_core: Tensor,
        *,
        temporal_offset: Tensor,
        output_start: int,
        output_end: int,
    ) -> Tensor:
        x = residual_core + self.attn.forward_region(
            normalized_halo, temporal_offset=temporal_offset, output_start=output_start, output_end=output_end
        )
        return x + self.mlp(self.norm2(x))

    def compile_looped_core(self, **compile_kwargs) -> None:
        """Compile the per-window core (``torch.compile`` keywords); the window loop itself stays eager."""
        if self.max_t is None:
            raise RuntimeError("compile_looped_core needs max_t")
        self._compiled_looped_core = torch.compile(self._forward_looped_core, **compile_kwargs)

    @torch.compiler.disable(recursive=False)
    def _forward_looped(self, x: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "looped decoding writes its output in place; run it under no_grad/inference_mode"
            )
        length = x.shape[1]
        kernel = self.attn.window_size[0]
        core = self._compiled_looped_core or self._forward_looped_core
        # The halo of a window reaches at most kernel - 1 < max_t frames back into the previous window, so
        # that window's output is held back for one iteration and written only after this halo was read.
        pending: tuple[int, int, Tensor] | None = None
        for core_start in range(0, length, self.max_t):
            core_end = min(core_start + self.max_t, length)
            halo_start, halo_end = _temporal_core_halo(core_start, core_end, length, kernel, self.attn.causal)
            normalized_halo = self.norm1(x[:, halo_start:halo_end])
            if pending is not None:
                x[:, pending[0] : pending[1]].copy_(pending[2])
            output = core(
                normalized_halo,
                x[:, core_start:core_end],
                temporal_offset=torch.tensor(float(halo_start), device=x.device),
                output_start=core_start - halo_start,
                output_end=core_end - halo_start,
            )
            pending = (core_start, core_end, output)
        assert pending is not None, "missing loop output"
        x[:, pending[0] : pending[1]].copy_(pending[2])
        return x


class PatchEmbed3d(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm: bool = True,
    ):
        super().__init__()
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        self.norm = _norm_layer(embed_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, t, h, w = x.size()
        pad = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.norm(x)


class DecoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        temporal: list[bool],
        num_heads: list[int],
        window_size: list[int],
        causal: bool = False,
        qk_norm: bool = False,
        max_t: int | None = None,
    ):
        super().__init__()
        assert len(temporal) == len(depths), "decoder stage mismatch"
        self.ps = patch_size
        self.max_t = max_t  # looped decode window, see SwinTransformerBlock
        self.proj_in = nn.Linear(z_ch, embed_dim * 2 ** (len(depths) - 1))
        self.proj_out = nn.Linear(embed_dim, math.prod(patch_size) * 3)

        layers: list[nn.Module] = []
        for i_stage in reversed(range(len(depths))):
            dim = embed_dim * 2**i_stage
            stage = [
                SwinTransformerBlock(
                    dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm, max_t=max_t
                )
                for _ in range(depths[i_stage])
            ]
            layers.append(nn.Sequential(*stage))
            if temporal[i_stage]:
                layers.append(TemporalExpansion(dim, dim))
                layers.append(
                    SwinTransformerBlock(
                        dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm, max_t=max_t
                    )
                )
            if i_stage > 0:
                layers.append(PatchExpansion(dim, dim // 2))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.proj_in(x)
        x = self.features(x)
        x = self.proj_out(x)
        b, t, h, w, _ = x.shape
        x = x.view(b, t, h, w, self.ps[0], self.ps[1], self.ps[2], 3)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(b, t * self.ps[0], h * self.ps[1], w * self.ps[2], 3)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class EncoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        temporal: list[bool],
        num_heads: list[int],
        window_size: list[int],
        causal: bool = False,
        qk_norm: bool = False,
        patch_norm: bool = True,
    ):
        super().__init__()
        assert len(temporal) == len(depths), "encoder stage mismatch"
        self.proj = nn.Linear(embed_dim * 2 ** (len(depths) - 1), z_ch)
        self.patch_embed = PatchEmbed3d(patch_size=patch_size, embed_dim=embed_dim, norm=patch_norm)

        layers: list[nn.Module] = []
        for i_stage in range(len(depths)):
            dim = embed_dim * 2**i_stage
            stage = [
                SwinTransformerBlock(dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm)
                for _ in range(depths[i_stage])
            ]
            layers.append(nn.Sequential(*stage))
            downsampled = False
            if i_stage < (len(depths) - 1):
                layers.append(PatchMerging(dim, 2 * dim))
                downsampled = True
            if temporal[i_stage]:
                if downsampled:
                    i_stage += 1
                    dim = 2 * dim
                layers.append(
                    SwinTransformerBlock(dim, num_heads[i_stage], window_size, causal=causal, qk_norm=qk_norm)
                )
                layers.append(TemporalMerging(dim, dim))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        _, c, _, _, _ = x.shape
        assert c == 3, "expected RGB input"
        x = self.patch_embed(x)
        # Inductor tiling assertion under dynamic=True without this break.
        torch._dynamo.graph_break()
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class ViTNorm(nn.Module):
    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        patch_size: list[int],
        window_size: list[int],
        enc_depths: list[int],
        dec_depths: list[int],
        num_heads: list[int],
        temporal: list[bool],
        enc_causal: bool = False,
        dec_causal: bool = False,
        qk_norm: bool = False,
        patch_norm: bool = False,
        decoder_max_t: int | None = None,
    ):
        super().__init__()
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        self.z_dim = z_dim
        self.encoder = EncoderSwin3D(
            z_ch=2 * z_dim,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=enc_depths,
            num_heads=num_heads,
            temporal=temporal,
            causal=enc_causal,
            qk_norm=qk_norm,
            patch_norm=patch_norm,
        )
        self.decoder = DecoderSwin3D(
            z_ch=z_dim,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=dec_depths,
            num_heads=num_heads,
            temporal=temporal,
            causal=dec_causal,
            qk_norm=qk_norm,
            max_t=decoder_max_t,
        )
        self.z_normalizer = DistributedRunningStats(z_dim)

    def encode(self, x: Tensor) -> Tensor:
        mu, _ = self.encoder(x).chunk(2, dim=-4)
        return self.z_normalizer.normalize(mu)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.z_normalizer.denormalize(z))


class ViTNormInference(nn.Module):
    TEMPORAL_DOWNSAMPLE = 4
    SPATIAL_DOWNSAMPLE = 32

    def __init__(self, params: ViTNormInferenceParams = ViTNormInferenceParams()):
        super().__init__()
        self.params = params
        if list(params.temporal) != [False, False, True, True]:
            raise ValueError(
                f"TEMPORAL_DOWNSAMPLE={self.TEMPORAL_DOWNSAMPLE} depends on temporal, got {params.temporal}"
            )
        if list(params.patch_size) != [1, 4, 4] or len(params.enc_depths) != 4:
            raise ValueError(
                f"SPATIAL_DOWNSAMPLE={self.SPATIAL_DOWNSAMPLE} depends on "
                f"patch_size and enc_depths, got {params.patch_size}, "
                f"{params.enc_depths}"
            )
        self.chunked_encode = params.chunked_encode
        self.chunk_size_frames = params.chunk_size_frames
        self.chunked_decode = params.chunked_decode
        self.chunk_size_latent_frames = params.chunk_size_latent_frames
        self.chunk_overlap_latent_frames = params.chunk_overlap_latent_frames
        self.decoder_max_t = params.decoder_max_t
        self.model = ViTNorm(
            z_dim=params.z_dim,
            embed_dim=params.embed_dim,
            patch_size=params.patch_size,
            window_size=params.window_size,
            enc_depths=params.enc_depths,
            dec_depths=params.dec_depths,
            num_heads=params.num_heads,
            temporal=params.temporal,
            enc_causal=params.enc_causal,
            dec_causal=params.dec_causal,
            qk_norm=params.qk_norm,
            patch_norm=params.patch_norm,
            decoder_max_t=params.decoder_max_t,
        )
        if params.dtype == "bfloat16":
            self.model = self.model.bfloat16()
        elif params.dtype != "float32":
            raise ValueError(f"unsupported video VAE dtype {params.dtype!r}")
        self.model.eval()
        self._encoder_compiled = False

    def compile_encoder(self) -> None:
        """Compile the encoder blocks used by inference without compiling the decoder."""
        if self._encoder_compiled:
            return
        cfg = self.params.compile_config or {}
        for layer_id, layer in self.model.encoder.features.named_children():
            self.model.encoder.features.register_module(
                layer_id,
                torch.compile(
                    layer,
                    mode="reduce-overhead",
                    fullgraph=cfg.get("fullgraph", False),
                    dynamic=cfg.get("dynamic_compile", False),
                    backend=cfg.get("backend", "inductor"),
                ),
            )
        self._encoder_compiled = True

    def apply_compile(self):
        params = self.params
        if not (params.use_compile or params.compile_decoder):
            return
        cfg = params.compile_config or {}
        mode = cfg.get("compile_mode", "default")
        fullgraph = cfg.get("fullgraph", False)
        dynamic = cfg.get("dynamic_compile", False)
        backend = cfg.get("backend", "inductor")
        if params.use_compile:
            self.compile_encoder()
        if params.compile_decoder and params.decoder_max_t is not None:
            # Looped blocks loop eagerly over windows; only the per-window core is compiled.
            for module in self.model.decoder.features.modules():
                if isinstance(module, SwinTransformerBlock):
                    module.compile_looped_core(
                        mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend
                    )
        elif params.compile_decoder:
            for layer_id, layer in self.model.decoder.features.named_children():
                self.model.decoder.features.register_module(
                    layer_id,
                    torch.compile(
                        layer,
                        mode=mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                        backend=backend,
                    ),
                )

    def min_chunk_enc(self, video: Tensor, target_num_frames: int | None = None) -> Tensor:
        """Encode ``(B, 3, T, H, W)`` in ``chunk_size_frames``-frame chunks overlapping by one frame; the first
        latent of every later chunk repeats the previous chunk's last one and is dropped."""
        if video.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W), got {video.shape}")
        chunk_size_frames = self.chunk_size_frames
        stride = chunk_size_frames - 1
        if stride <= 0:
            raise ValueError(f"chunk_size_frames must be > 1, got {chunk_size_frames}")
        num_frames = video.shape[2]
        if target_num_frames is None:
            target_num_frames = num_frames
        if target_num_frames > num_frames:
            raise ValueError(
                f"target_num_frames ({target_num_frames}) cannot exceed encoded frames ({num_frames})"
            )
        if num_frames < chunk_size_frames:
            raise ValueError(
                f"Chunked encoding expects at least {chunk_size_frames} frames; got T={num_frames}"
            )
        if (num_frames - chunk_size_frames) % stride != 0:
            raise ValueError(
                f"Chunked encoding expects T = {chunk_size_frames} + n * {stride}; got T={num_frames}"
            )

        latent_pieces = []
        for start in range(0, num_frames - chunk_size_frames + 1, stride):
            z_chunk = self.model.encode(video[:, :, start : start + chunk_size_frames])
            latent_pieces.append(z_chunk if start == 0 else z_chunk[:, :, 1:])

        latent = torch.cat(latent_pieces, dim=2)
        target_latent_frames = 1 + (target_num_frames - 1) // self.TEMPORAL_DOWNSAMPLE
        if target_latent_frames > latent.shape[2]:
            raise ValueError(
                f"Chunked encode produced {latent.shape[2]} latent frames, cannot "
                f"trim to {target_latent_frames}"
            )
        return latent[:, :, :target_latent_frames]

    def encode(
        self,
        x: Tensor,
        *,
        chunked_encode: bool | None = None,
        target_num_frames: int | None = None,
    ) -> Tensor:
        if chunked_encode is None:
            chunked_encode = self.chunked_encode
        if chunked_encode:
            return self.min_chunk_enc(x, target_num_frames=target_num_frames)
        return self.model.encode(x)

    def min_chunk_dec(self, z: Tensor) -> Tensor:
        # Decode the (already-denormalized) latent in overlapping temporal
        # windows, keep each window's core pixel frames, concatenate. The
        # overlap is decoded purely as temporal context and discarded (no
        # blending), larger overlap brings seam frames closer to full decode.
        if z.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W), got {z.shape}")
        core = self.chunk_size_latent_frames
        overlap = self.chunk_overlap_latent_frames
        if core < 1 or overlap < 1:
            raise ValueError(
                f"chunk_size/overlap latent frames must be >= 1, got core={core}, overlap={overlap}"
            )
        # Shrink core if free VRAM can't hold a full window (core + 2*overlap).
        free, _ = torch.cuda.mem_get_info(z.device)
        budget = free / DECODE_SAFETY_FACTOR - DECODE_FIXED_BYTES
        window_fit = int(budget / (DECODE_BYTES_PER_LATENT_ELEM * z[0, 0, 0].numel()))
        core = max(1, min(core, window_fit - 2 * overlap))
        t_lat = z.shape[2]
        logger.info(
            f"tiled video decode: t_lat={t_lat} core={core} overlap={overlap} free={free / 2**30:.1f}GiB"
        )

        pieces = []
        for core_lo in range(0, t_lat, core):
            core_hi = min(core_lo + core, t_lat)
            lo = max(0, core_lo - overlap)
            hi = min(t_lat, core_hi + overlap)
            decoded = self.model.decoder(z[:, :, lo:hi])
            # Slice the core latent range to its pixel range within this window.
            # core_lo == 0 keeps the leading pixel frame, interior cores start
            # at 4*(core_lo - lo) - 3. overlap >= 1 keeps local_start >= 0.
            local_start = 0 if core_lo == 0 else 4 * (core_lo - lo) - 3
            local_end = 4 * (core_hi - lo) - 3
            pieces.append(decoded[:, :, local_start:local_end])

        out = torch.cat(pieces, dim=2)
        assert out.shape[2] == 4 * t_lat - 3, (
            f"tiling misaligned: {out.shape[2]} frames, expected {4 * t_lat - 3}"
        )
        return out

    def _full_decode_fits(self, z: Tensor) -> bool:
        if z.device.type != "cuda":
            return True
        free, _ = torch.cuda.mem_get_info(z.device)
        need = DECODE_BYTES_PER_LATENT_ELEM * z[0, 0].numel() + DECODE_FIXED_BYTES
        return need * DECODE_SAFETY_FACTOR < free

    def decode(self, z: Tensor, *, chunked_decode: bool | None = None) -> Tensor:
        """Latents ``(B, 96, T_lat, H, W)`` -> pixels ``(B, 3, 4 * T_lat - 3, 32 H, 32 W)`` in ``[-1, 1]``.

        The whole latent is decoded in one pass when possible. With ``decoder_max_t`` set (the loader
        default) that pass is the looped decode, whose memory is bounded by the window rather than the clip
        length, so it is always attempted first; without it, a memory estimate picks between the full and
        the latent-chunked decode. The latent-chunked decode (``chunked_decode=True``, or the fallback after
        an out-of-memory error) tiles the latent along time with discarded overlap and is not seam-free.
        """
        auto = chunked_decode is None
        if auto:
            fits = self.decoder_max_t is not None or self._full_decode_fits(z)
            chunked_decode = self.chunked_decode and not fits
        if not chunked_decode:
            try:
                return self.model.decode(z).clamp(-1, 1)
            except torch.OutOfMemoryError:
                # Estimate was too optimistic, fall through to the chunked path below.
                if not (auto and self.chunked_decode):
                    raise
                logger.warning(f"full video decode OOM (t_latent={z.shape[2]}), falling back to chunked")
                torch.cuda.empty_cache()
        z = self.model.z_normalizer.denormalize(z)
        return self.min_chunk_dec(z).clamp(-1, 1)


# ---------------------------------------------------------------------------------------------
# Policy-side wrapper and loader. The KinoVAE is frozen,
# loaded from a local file / directory or a Hub repo, and never part of the policy's own
# ``model.safetensors``.
# ---------------------------------------------------------------------------------------------
from ..processing.packing import padded_chunk_length  # noqa: E402

from .runtime import random_init_  # noqa: E402

VIDEO_VAE_WEIGHTS_FILENAME = "video_vae.safetensors"


class VideoVAE:
    """Frozen video VAE behind the encode/decode contract the packing code relies on.

    ``encode`` takes ``(B, 3, T, H, W)`` in ``[-1, 1]`` with ``T == 1 (mod 4)`` and returns normalized
    latents ``(B, 96, 1 + (T - 1) // 4, H // 32, W // 32)``. Encoding is chunked (45-frame chunks,
    1-frame overlap, first latent of each later chunk dropped); shorter clips are padded by repeating
    the last frame and trimmed back. ``packing.encode_video`` pads with black to the chunk length
    *before* calling this, so training and deployment see the same latent distribution.

    ``decode`` is the looped decode (``ViTNormInference.decode``): the full-sequence result with the
    decoder's activations bounded by an 8-frame window.
    """

    def __init__(self, model: "ViTNormInference"):
        self._model = model

    @property
    def module(self) -> "ViTNormInference":
        return self._model

    @torch.inference_mode()
    def encode(self, video: Tensor) -> Tensor:
        num_frames = video.shape[2]
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"video VAE encode expects T == 1 (mod 4) frames, got T={num_frames}; "
                "trim or pad the clip to the 4k+1 grid (e.g. 45, 49, 121)"
            )
        chunk = self._model.chunk_size_frames
        padded = padded_chunk_length(num_frames, chunk)
        if padded > num_frames:
            tail = video[:, :, -1:].expand(-1, -1, padded - num_frames, -1, -1)
            video = torch.cat([video, tail], dim=2)
        return self._model.min_chunk_enc(video, target_num_frames=num_frames)

    @torch.inference_mode()
    def encode_task(self, video: Tensor) -> Tensor:
        """Independent history snapshot or future clip, without padding or chunk overlap."""
        return self._model.encode(video, chunked_encode=False)

    @torch.inference_mode()
    def encode_frame(self, frame: Tensor) -> Tensor:
        """``(B, 3, H, W)`` -> ``(B, 96, 1, H // 32, W // 32)``: the frame alone, without the 45-frame padding
        of ``encode``. See ``PolicyConfig.single_frame_encode``."""
        return self._model.encode(frame[:, :, None], chunked_encode=False)

    @torch.inference_mode()
    def decode(self, latents: Tensor) -> Tensor:
        return self._model.decode(latents)


def _resolve_weights(spec: str) -> "str":
    """Local file / DCP directory as is; anything else is ``repo_id[:filename][@revision]`` on the Hub."""
    from .runtime import resolve_weights

    return resolve_weights(spec, VIDEO_VAE_WEIGHTS_FILENAME)


def load_video_vae(
    weights: str | None,
    device: str | torch.device = "cpu",
    *,
    compile_model: bool = False,
    decoder_max_t: int | None = 8,
) -> VideoVAE:
    """Build the frozen KinoVAE.

    ``weights``: ``None`` -> random init (shapes only, for wiring tests), a ``.safetensors`` file, a
    distributed-checkpoint directory (``.metadata`` + ``__*.distcp``), or a Hub ``repo_id[:filename]``
    (default filename ``video_vae.safetensors``). Requires NATTEN (``natten`` package) at import.

    ``decoder_max_t``: temporal window of the looped decode (8 is the reference setting; it must be at
    least the temporal attention kernel, 5). ``None`` restores the plain full decode with the
    latent-chunked fallback.
    """
    import os
    from dataclasses import replace
    from pathlib import Path

    params = ViTNormInferenceParams(
        use_compile=compile_model,
        compile_decoder=compile_model,
        chunked_decode=True,
        chunked_encode=True,
        decoder_max_t=decoder_max_t,
    )
    if weights is None:
        model = ViTNormInference(params)
        random_init_(model)
    else:
        path = _resolve_weights(weights)
        if os.path.isdir(path) or (Path(path) / ".metadata").is_file():
            import torch.distributed.checkpoint as dcp

            model = ViTNormInference(replace(params, dtype="float32"))
            state_dict = {"model": model.model.state_dict()}
            dcp.load(state_dict, checkpoint_id=str(path))
            model.model.load_state_dict(state_dict["model"])
            model.model.bfloat16()
        else:
            from safetensors.torch import load_file

            model = ViTNormInference(params)
            model.load_state_dict(load_file(path, device="cpu"))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if compile_model:
        model.apply_compile()
    return VideoVAE(model)
