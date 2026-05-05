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

"""SigLIP video encoder with space-time separable attention (MEM paper).

Implements the low-level memory video encoder from Torne, Pertsch, Walke et al.
"MEM: Multi-Scale Embodied Memory for Vision Language Action Models"
(Section III-C + Appendix C): a standard SigLIP ViT extended with
space-time separable attention at every ``spacetime_layer_stride``-th layer
and a fixed sinusoidal temporal position encoding whose current-frame row is
zero. Past-timestep tokens are dropped after the encoder so the output shape
matches a single-image VLA.

This module is the **single, canonical** implementation of the SigLIP
video encoder; all callers — pi05_mem, pi07/low_level (Gemma 3 backbone),
and pi07_paligemma/low_level_planner (legacy PaliGemma backbone) — import
from here.

Key properties:
  - Introduces no new learnable parameters on top of the pretrained SigLIP
    weights (temporal attention re-uses each layer's own Q/K/V/O projections).
    Any pi05/pi05_continuous_state checkpoint can be loaded directly — the
    space-time layers just wrap the existing SiglipEncoderLayer weights.
  - Single-frame invariance: with ``T=1`` the output is byte-identical to
    ``PaliGemmaModel.get_image_features`` (see the single-frame invariance
    tests).
  - Convention: the current frame lives at the **last** time index
    (``t = T-1``). This matches
    ``src/opentau/datasets/factory.py:136`` (delta_timestamps) and
    ``PI05MemPolicy._build_history_batch``. ``obs_history_is_pad[:, -1]`` is
    always ``False`` by construction.
  - **Variable number of frames per forward.** The encoder is constructed
    with a ``max_num_frames`` cap (used to size the cached temporal PE
    buffer); each forward accepts any ``T`` in ``[1, max_num_frames]``.
    Slicing the precomputed PE as ``pe[max_num_frames - T:]`` preserves the
    "row T-1 is zero" invariant — the PE values are functions of the
    relative offset from the current frame, so the last ``T`` rows of a
    PE built for ``M`` frames are byte-identical to a PE built for ``T``
    frames directly.

The encoder does NOT own its own copy of the SigLIP weights. The caller
constructs a ``PaliGemmaWithExpertModel`` (pi05_mem, pi07_paligemma) or
``Gemma3WithExpertModel`` (pi07/low_level) — which already owns
``vision_tower`` and ``multi_modal_projector`` — and passes them in by
reference. This avoids duplicating ~400M parameters in memory.
"""

import math
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers.models.siglip.modeling_siglip import (
    SiglipEncoderLayer,
    SiglipVisionModel,
)

# Import triggers the transformers patch (see opentau.utils.transformers_patch)
# which rewrites PaliGemmaModel.get_image_features to drop the
# `/ sqrt(hidden_size)` scaling that stock HuggingFace applies after the
# multi_modal_projector. Our forward must match that patched behavior for
# single-frame invariance to hold.
import opentau.utils.transformers_patch  # noqa: F401


def _build_temporal_sinusoidal_pe(
    num_frames: int,
    embed_dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 40.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Fixed sinusoidal temporal positional embedding, ``(T, embed_dim)``.

    Row ``T-1`` (the current frame) is all zeros; earlier rows encode the
    temporal offset into the past via sin/cos on a geometric period schedule
    (matching ``create_sinusoidal_pos_embedding`` in ``modeling_pi05.py``).

    The zero-current-row condition lets a ``T=1`` forward pass match an
    un-modified SigLIP ViT exactly, which is required for single-frame
    invariance against ``PaliGemmaModel.get_image_features``.

    The ``max_period`` floor governs the LONGEST sinusoidal period. To avoid
    temporal aliasing (different timesteps mapping to nearly-identical
    low-frequency rows), ``max_period`` should comfortably exceed the time
    range ``T-1``. The default ``40.0`` covers up to ``T=20`` with the
    longest sinusoid completing at most ``19 / 40 ≈ 0.48`` of a cycle —
    every timestep in ``{-19, ..., 0}`` therefore gets a unique
    low-frequency encoding. Callers needing ``T > 20`` should pass a larger
    ``max_period`` explicitly (rule of thumb: ``2 * (T - 1)``).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2")
    if num_frames < 1:
        raise ValueError(f"num_frames ({num_frames}) must be >= 1")
    if num_frames - 1 > max_period:
        raise ValueError(
            f"num_frames ({num_frames}) exceeds the no-alias range of max_period "
            f"({max_period}); the lowest-frequency sinusoid would complete more than "
            f"a full cycle over the time range, causing temporal aliasing. Pass "
            f"max_period >= {2 * (num_frames - 1)} explicitly."
        )

    # time[i] = i - (T-1) in {-(T-1), ..., -1, 0}; row T-1 has time = 0.
    time = torch.arange(num_frames, dtype=torch.float64, device=device) - (num_frames - 1)
    fraction = torch.linspace(0.0, 1.0, embed_dim // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi  # (embed_dim/2,)
    phase = time.unsqueeze(-1) * scaling.unsqueeze(0)  # (T, embed_dim/2)
    pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (T, embed_dim)
    # Shift so row T-1 is exactly zero (preserves relative sinusoidal structure,
    # enforces boundary condition e(current) = 0 from MEM Appendix C).
    pe = pe - pe[-1:]
    return pe.to(dtype=dtype)


class SpaceTimeEncoderLayerWrapper(nn.Module):
    """Replaces a ``SiglipEncoderLayer`` in-place; adds factorized space-time
    attention via a single composed attention sublayer (Reading B of MEM
    Eq. 3).

    The wrapper **adopts** the original layer's submodules by reference —
    ``self_attn``, ``layer_norm1``, ``layer_norm2``, ``mlp`` — so its
    ``state_dict`` keys are **identical** to a vanilla ``SiglipEncoderLayer``.
    That means any pi05 / pi05_continuous_state checkpoint can load directly
    into the wrapped layer without any key remapping. The only new state is
    a non-persistent ``_temporal_pe`` buffer (excluded from state_dict).

    The forward computes (single QKV projection per block; two SDPAs share
    those projections; one residual; one out_proj):

        h_pe  = h + e(t)                              # broadcast over (B, N)
        z     = LN1(h_pe)                             # ONE LN
        Q,K,V = W_Q·z, W_K·z, W_V·z                   # ONE projection
        V'    = SDPA(Q, K, V; causal mask over T)     # temporal pass per patch
        out   = SDPA(Q, K, V'; no mask, over N)       # spatial pass per timestep
        h     = h + W_O(out)                          # ONE residual on h (not h_pe)
        h     = h + MLP( LN2(h) )                     # standard MLP block

    Q and K are reused across both passes (same tensor, just permuted into
    each layout); only V is replaced by ``V'`` for the spatial pass. ``W_O``
    fires once at the end. This matches the paper's "no new learnable
    parameters" claim at the projection level: ``W_Q``, ``W_K``, ``W_V``,
    ``W_O`` are each applied exactly once per block, not twice.

    At ``T=1`` the temporal SDPA collapses to the identity (a single key,
    so the attention weight is 1 and ``V`` passes through unchanged) and
    ``e(t=0)=0`` makes ``h_pe = h``. The block is therefore mathematically
    identical to a vanilla ``SiglipEncoderLayer`` forward at ``T=1`` — no
    short-circuit is required for correctness. The wrapper still routes
    ``T=1`` inputs through ``_spatial_block_forward`` for compute savings
    and bit-exact match in low-precision dtypes (where the no-op temporal
    SDPA can drift by ~1 ULP).

    Variable ``T`` per forward: the cached PE is built once for
    ``max_num_frames`` rows and sliced as ``pe[max_num_frames - num_frames:]``
    each forward. Because PE values depend only on the relative offset from
    the current frame (not on the absolute index), this slice is byte-identical
    to a PE freshly built for the actual ``num_frames``.
    """

    # Match the SiglipEncoderLayer class attribute so transformers'
    # gradient-checkpointing plumbing sees a familiar interface.
    gradient_checkpointing: bool = False

    def __init__(
        self,
        base_layer: SiglipEncoderLayer,
        max_num_frames: int,
        num_tokens_per_frame: int,
    ):
        super().__init__()
        # Adopt the base layer's submodules as our own (same attribute names).
        # The state_dict therefore uses keys like
        # ``encoder.layers.{i}.self_attn.q_proj.weight`` — identical to a
        # vanilla SiglipEncoderLayer, so pi05 checkpoints load directly.
        self.self_attn = base_layer.self_attn
        self.layer_norm1 = base_layer.layer_norm1
        self.layer_norm2 = base_layer.layer_norm2
        self.mlp = base_layer.mlp
        self.embed_dim = base_layer.embed_dim

        self.max_num_frames = max_num_frames
        self.num_tokens_per_frame = num_tokens_per_frame

        # Caller-driven flag: when False, ``forward`` short-circuits to the
        # vanilla spatial-only block (`_spatial_block_forward`) regardless of
        # the input shape. This is used by ``Gemma3WithExpertModel.embed_image``
        # via the ``suppress_spacetime_temporal`` context manager so the same
        # wrapped vision_tower can be reused for non-video inputs (e.g. single
        # subgoal images) without firing temporal attention over data that has
        # no time axis.  Flag lives on the wrapper rather than on a kwarg
        # because ``SiglipEncoder.forward`` does not accept extra kwargs.
        self._temporal_active: bool = True

        # Build the PE on the base layer's current device / dtype. The parent
        # vision_tower is often moved to GPU BEFORE this wrapper is inserted
        # (the normal load flow does ``paligemma = ...from_pretrained(...).to(
        # 'cuda')`` and then wraps); with no parent ``.to(device)`` happening
        # after wrapping, a PE built on CPU would stay on CPU and trigger a
        # cross-device RuntimeError at forward time. Pinning to the base
        # layer's device sidesteps that.
        ref_param = base_layer.self_attn.q_proj.weight
        pe = _build_temporal_sinusoidal_pe(
            max_num_frames, self.embed_dim, dtype=ref_param.dtype, device=ref_param.device
        )
        # Non-persistent: not saved in state_dict but moves with .to(device).
        self.register_buffer("_temporal_pe", pe, persistent=False)

    def _spatial_block_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        output_attentions: bool,
    ) -> tuple[Tensor, ...]:
        """Inlined SiglipEncoderLayer.forward using the adopted submodules.

        Mirrors
        ``transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward``
        exactly — any upstream change to that forward would need to be
        reflected here.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple[Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        temporal_attn_mask: Tensor | None = None,
        num_frames: int | None = None,
    ) -> tuple[Tensor, ...]:
        """hidden_states: (B*T, N, D) -> tuple starting with (B*T, N, D).

        Signature extends ``SiglipEncoderLayer.forward`` with two extra
        kwargs: ``temporal_attn_mask`` and ``num_frames``. Vanilla
        ``SiglipEncoderLayer`` instances never receive them:
        ``SpaceTimeSiglipVideoEncoder.forward`` dispatches them only to
        ``SpaceTimeEncoderLayerWrapper`` instances via an ``isinstance``
        check, bypassing ``SiglipEncoder.forward`` entirely.

        Args:
            temporal_attn_mask: Optional ``(B*N, 1, T, T)`` additive float mask
                for temporal attention. ``0.0`` = attend, ``-inf`` = block.
                When ``None``, the standard causal mask is used.
            num_frames: Number of frames ``T`` for this forward. Required when
                ``_temporal_active`` is ``True`` and the input is multi-frame
                (the wrapper cannot infer ``T`` from the ``(B*T, N, D)`` shape
                alone). May be ``None`` when ``_temporal_active`` is ``False``
                (suppress path) or when the caller knows the input is a single
                frame.
        """
        bt, n, d = hidden_states.shape
        if n != self.num_tokens_per_frame:
            raise ValueError(
                f"hidden_states.shape[1] ({n}) != num_tokens_per_frame ({self.num_tokens_per_frame})."
            )

        # Short-circuit when the caller has suppressed temporal attention.
        # ``Gemma3WithExpertModel.embed_image`` shares the same wrapped vision
        # tower for non-video inputs (e.g. subgoal frames) and toggles this
        # flag via the ``suppress_spacetime_temporal`` context manager — those
        # calls are spatial-only; there is no time axis to attend over.
        if not self._temporal_active:
            return self._spatial_block_forward(hidden_states, attention_mask, output_attentions)

        # Short-circuit at T=1. Under Reading B the block IS naturally
        # identity at T=1 (the temporal SDPA over a single key returns V
        # unchanged, and ``e(t=0)=0`` makes ``h_pe = h``), so this is purely
        # an optimization: it avoids the no-op temporal SDPA and guarantees
        # bit-exact match with vanilla SigLIP in low-precision dtypes.
        if num_frames is None or num_frames == 1:
            return self._spatial_block_forward(hidden_states, attention_mask, output_attentions)

        if num_frames > self.max_num_frames:
            raise ValueError(
                f"num_frames ({num_frames}) > max_num_frames ({self.max_num_frames}); "
                "reinstantiate the encoder with a larger max_num_frames."
            )

        t = num_frames
        if bt % t != 0:
            raise ValueError(
                f"hidden_states.shape[0] ({bt}) must be divisible by num_frames ({t}); "
                "video encoder expects inputs flattened as (B*T, N, D). "
                "Use `suppress_spacetime_temporal(...)` for non-video forwards."
            )
        b = bt // t

        attn = self.self_attn
        num_heads = attn.num_heads

        # Reshape (B*T, N, D) -> (B, T, N, D) and add temporal PE.
        # Slice the precomputed PE: take the LAST t rows so the current-frame
        # row (t = T-1) remains exactly zero (PE entries depend only on the
        # relative offset from the current frame, so ``pe[M-t:M]`` is
        # byte-identical to a fresh PE built for t frames). Cast PE to match
        # the tensor's device/dtype each call; no-op if already aligned.
        x = rearrange(hidden_states, "(b t) n d -> b t n d", b=b, t=t)
        pe_full = self._temporal_pe.to(device=x.device, dtype=x.dtype)
        pe = pe_full[self.max_num_frames - t :].view(1, t, 1, d)
        x_pe = x + pe

        # ONE LayerNorm. Its output feeds BOTH attention passes.
        z = self.layer_norm1(x_pe)

        # ONE QKV projection. Q/K/V come from a single application of the
        # layer's W_Q/W_K/W_V to LN1(h_pe). They are reshaped (not
        # re-projected) into the temporal and spatial layouts below.
        q = rearrange(attn.q_proj(z), "b t n (h d) -> b t n h d", h=num_heads)
        k = rearrange(attn.k_proj(z), "b t n (h d) -> b t n h d", h=num_heads)
        v = rearrange(attn.v_proj(z), "b t n (h d) -> b t n h d", h=num_heads)

        # ===== Temporal SDPA: per-patch causal attention over T =====
        # Layout (B*N, H, T, Dh): each spatial patch is its own length-T
        # causal sequence; flatten (B, N) into the SDPA batch axis.
        q_t = rearrange(q, "b t n h d -> (b n) h t d")
        k_t = rearrange(k, "b t n h d -> (b n) h t d")
        v_t = rearrange(v, "b t n h d -> (b n) h t d")

        if temporal_attn_mask is not None:
            # Caller-supplied mask already encodes both the causal pattern AND
            # padded-history blocking; do NOT also set is_causal=True (SDPA
            # disallows combining the two).
            v_temp = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=temporal_attn_mask,
                is_causal=False,
                dropout_p=0.0,
                scale=attn.scale,
            )
        else:
            # is_causal=True -> lower-triangular mask. Position i attends to
            # j <= i; since t = T-1 is the current frame, the current frame
            # attends to all past frames.
            v_temp = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=None,
                is_causal=True,
                dropout_p=0.0,
                scale=attn.scale,
            )
        # v_temp: (B*N, H, T, Dh). Rearrange back into (B, T, N, H, Dh) layout
        # so it can feed the spatial pass as the V input. ``rearrange`` is
        # defensive against non-contiguous SDPA outputs (some flash-attn
        # backends return non-standard strides) — it is a no-op when the
        # tensor is already contiguous and a copy when it is not.
        v_temp = rearrange(v_temp, "(b n) h t d -> b t n h d", b=b)

        # ===== Spatial SDPA: per-timestep bidirectional attention over N =====
        # Q, K REUSE the same projections from z (NOT recomputed from v_temp).
        # V is v_temp (the temporally-mixed values). Layout (B*T, H, N, Dh).
        q_s = rearrange(q, "b t n h d -> (b t) h n d")
        k_s = rearrange(k, "b t n h d -> (b t) h n d")
        v_s = rearrange(v_temp, "b t n h d -> (b t) h n d")

        out = F.scaled_dot_product_attention(
            q_s,
            k_s,
            v_s,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0.0,
            scale=attn.scale,
        )
        # (B*T, H, N, Dh) -> (B*T, N, D)
        out = rearrange(out, "bt h n d -> bt n (h d)")

        # ONE output projection.
        out = attn.out_proj(out)

        # ONE residual on the original hidden_states (NOT on x_pe). PE is a
        # transient positional signal that informs the attention computation;
        # it is not a feature perturbation to carry forward in the residual.
        h_after_attn = hidden_states + out

        # Standard SigLIP MLP block on the post-attention residual.
        residual = h_after_attn
        h_norm = self.layer_norm2(h_after_attn)
        h_mlp = self.mlp(h_norm)
        h_out = residual + h_mlp

        outputs: tuple[Tensor, ...] = (h_out,)
        if output_attentions:
            # SDPA does not return attention weights, and Reading B never
            # materializes either α_temporal or α_spatial. Return None to
            # keep the tuple shape; callers requesting weights at T>1 should
            # switch to an explicit-attention build of SigLIP if they need
            # per-layer weights.
            outputs = outputs + (None,)
        return outputs


@contextmanager
def suppress_spacetime_temporal(module: nn.Module) -> Iterator[None]:
    """Context manager that flips ``_temporal_active=False`` on every
    :class:`SpaceTimeEncoderLayerWrapper` in ``module``'s subtree, and
    restores the previous value on exit.

    Used by ``Gemma3WithExpertModel.embed_image`` so that single-image
    forwards through a vision_tower that has been wrapped with space-time
    attention skip the temporal sublayer (which has no time axis to attend
    over for non-video inputs).  When ``module`` contains no wrappers (e.g.
    no video encoder has been constructed yet), this is a no-op.
    """
    wrappers: list[SpaceTimeEncoderLayerWrapper] = [
        m for m in module.modules() if isinstance(m, SpaceTimeEncoderLayerWrapper)
    ]
    previous = [w._temporal_active for w in wrappers]
    for w in wrappers:
        w._temporal_active = False
    try:
        yield
    finally:
        for w, prev in zip(wrappers, previous, strict=True):
            w._temporal_active = prev


class SpaceTimeSiglipVideoEncoder(nn.Module):
    """SigLIP-based video encoder with space-time separable attention.

    Takes video tensors of shape ``(B, T, 3, H, W)`` in the ``[0, 1]`` range
    and produces ``(B, num_video_tokens, vlm_hidden_size)``. Rescales pixels
    to ``[-1, 1]`` internally (SigLIP's expected range).

    ``T`` may vary per forward in ``[1, max_num_frames]``; the encoder is
    constructed with a ``max_num_frames`` cap that sizes the cached temporal
    PE buffer.

    Past-timestep tokens are dropped after the encoder; only the current
    frame's ``num_video_tokens`` tokens are returned, so the output shape is
    identical to a single-frame VLA's vision-token prefix.

    The ``multi_modal_projector`` is applied to match the output space of
    ``PaliGemmaModel.get_image_features``. We intentionally **do not** apply
    the ``/ sqrt(text_hidden_size)`` scaling, matching
    ``opentau.utils.transformers_patch.patched_paligemma_model_get_image_features``
    which removes it from stock HuggingFace.

    The caller owns ``vision_tower`` and ``multi_modal_projector``. This
    module holds them by reference (via a list, so ``nn.Module`` does not
    re-register their parameters under this module's path) and mutates the
    vision_tower's encoder in place to wrap every ``spacetime_layer_stride``-th
    layer. Callers include ``PI07LowLevelFlowMatching`` (Gemma 3 backbone),
    ``PI05MemFlowMatching`` (PaliGemma backbone), and the legacy
    ``PI07LowLevelFlowMatching`` under ``pi07_paligemma`` (PaliGemma backbone).
    """

    def __init__(
        self,
        vision_tower: SiglipVisionModel,
        multi_modal_projector: nn.Module,
        max_num_frames: int,
        spacetime_layer_stride: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if max_num_frames < 1:
            raise ValueError(f"max_num_frames ({max_num_frames}) must be >= 1.")
        if spacetime_layer_stride < 1:
            raise ValueError(f"spacetime_layer_stride ({spacetime_layer_stride}) must be >= 1.")

        self.max_num_frames = max_num_frames
        self.spacetime_layer_stride = spacetime_layer_stride
        # Wrap each SigLIP encoder layer (vanilla or space-time) in
        # torch.utils.checkpoint.checkpoint during training. Mirrors the
        # explicit per-layer pattern used by pi05's PaliGemmaWithExpertModel
        # so we do not depend on transformers' SiglipEncoder internal
        # gradient-checkpointing plumbing. The strict distributed-backend
        # guard in src/opentau/scripts/train.py applies (DDP, single, or
        # DeepSpeed ZeRO-1/2 only).
        self.gradient_checkpointing = gradient_checkpointing

        # Hold references in lists so nn.Module.__setattr__ does not
        # re-register these modules under this encoder's path. They are owned
        # by the caller; double registration would duplicate ~400M params in
        # state_dict.
        self._vision_tower_ref: list[SiglipVisionModel] = [vision_tower]
        self._multi_modal_projector_ref: list[nn.Module] = [multi_modal_projector]

        # The number of output tokens is fixed by the SigLIP patch grid
        # (e.g. 224/14 = 16 -> 16*16 = 256 patches for the default config).
        vision_cfg = vision_tower.config
        num_patches = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        self.num_video_tokens = num_patches
        self.siglip_hidden_size = vision_cfg.hidden_size

        # Wrap every stride-th layer with space-time attention. The wrapper
        # adopts the base layer's submodules (self_attn / layer_norm{1,2} /
        # mlp) by reference, so wrapped-layer state-dict keys are byte-for-byte
        # identical to a vanilla ``SiglipEncoderLayer`` — no ``.base_layer.``
        # prefix appears.
        layers = vision_tower.vision_model.encoder.layers
        n_layers = len(layers)
        for i in range(spacetime_layer_stride - 1, n_layers, spacetime_layer_stride):
            layers[i] = SpaceTimeEncoderLayerWrapper(
                base_layer=layers[i],
                max_num_frames=max_num_frames,
                num_tokens_per_frame=num_patches,
            )

    @property
    def vision_tower(self) -> SiglipVisionModel:
        return self._vision_tower_ref[0]

    @property
    def multi_modal_projector(self) -> nn.Module:
        return self._multi_modal_projector_ref[0]

    @staticmethod
    def _build_temporal_attn_mask(
        obs_history_is_pad: Tensor,
        num_patches: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build a causal temporal attention mask that blocks padded frames.

        Pure pixel-level zeroing of padded frames is not enough — the SigLIP
        patch embedding has a learned bias and the temporal positional
        embedding ``e(t)`` is non-zero for ``t < T-1``, so padded "zero" frames
        still produce non-zero hidden states that the current frame would
        attend to. This mask blocks attention to padded keys at the SDPA call.

        Args:
            obs_history_is_pad: ``(B, T)`` bool — ``True`` for padded steps.
            num_patches: ``N``, number of spatial patches per frame (the
                video encoder runs one temporal sequence per patch position,
                so each patch row of the (B*N) batch reuses the same mask).
            dtype: Float dtype matching the hidden states (additive mask gets
                added to attention scores; mismatched dtypes force upcasts).

        Returns:
            ``(B*N, 1, T, T)`` additive float mask where ``0.0`` = attend and
            ``-inf`` = block. Row ``i`` can attend to column ``j`` iff
            ``j <= i`` (causal) **and** ``obs_history_is_pad[:, j]`` is
            ``False``. The current frame (``j = T-1``) is always attendable
            even if the caller set ``obs_history_is_pad[:, -1] = True`` —
            losing the current frame would defeat the encoder.
        """
        b, t = obs_history_is_pad.shape
        device = obs_history_is_pad.device

        # Causal: position i attends to j <= i.
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))  # (T, T)

        # Key-side visibility: True where frame j is real (not padded).
        # Force the last frame always attendable as a defensive fallback —
        # callers (e.g. the dataset's history_state_drop_prob augmentation)
        # set obs_history_is_pad to all-True; without this override, the
        # current frame would have no key to attend to and produce NaNs.
        # `~obs_history_is_pad` allocates a fresh tensor, so the in-place
        # write below does not reach the caller's `obs_history_is_pad`.
        key_valid = ~obs_history_is_pad  # (B, T)
        key_valid[:, -1] = True

        # Combined: (B, T_query, T_key)
        mask_bool = causal.unsqueeze(0) & key_valid.unsqueeze(1)  # (B, T, T)

        # Bool → additive float: True → 0.0, False → -inf.
        float_mask = torch.zeros(b, t, t, dtype=dtype, device=device)
        float_mask.masked_fill_(~mask_bool, float("-inf"))

        # Expand for the (B*N) flattened-patch batch dimension:
        #   (B, T, T) → (B, 1, T, T) → repeat_interleave(N) → (B*N, 1, T, T)
        float_mask = float_mask.unsqueeze(1)  # (B, 1, T, T)
        float_mask = float_mask.repeat_interleave(num_patches, dim=0)  # (B*N, 1, T, T)

        return float_mask

    def forward(self, video: Tensor, obs_history_is_pad: Tensor | None = None) -> Tensor:
        """Encode a video clip and return the current-frame tokens.

        Args:
            video: ``(B, T, C, H, W)`` pixel values in ``[0, 1]``, with
                ``1 <= T <= max_num_frames``, ``C == 3``, and spatial size
                matching the SigLIP config (224x224 by default).
            obs_history_is_pad: Optional ``(B, T)`` bool mask where ``True``
                marks padded history frames. Padded frames are blocked in
                the temporal attention so the current frame cannot read
                contaminated hidden states from them. When ``None`` and
                ``T > 1``, falls back to "only the current frame is real"
                (matches inference-time semantics where
                ``_build_history_batch`` does not populate this mask).

        Returns:
            ``(B, num_video_tokens, vlm_hidden_size)`` current-frame tokens,
            ready to concatenate into the VLA prefix.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W); got {tuple(video.shape)}.")
        b, t, c, h, w = video.shape
        if t < 1:
            raise ValueError(f"Expected T >= 1; got {t}.")
        if t > self.max_num_frames:
            raise ValueError(
                f"Expected T <= max_num_frames ({self.max_num_frames}); got {t}. "
                "Reinstantiate the encoder with a larger max_num_frames."
            )

        # SigLIP expects pixel values in [-1, 1]. The dataset loader yields
        # [0, 1]; rescale here (keeps prepare_videos producer-agnostic).
        video = video * 2.0 - 1.0

        # Flatten time into batch for the SigLIP pipeline.
        flat = rearrange(video, "b t c h w -> (b t) c h w")

        # Patch embedding + learned spatial position embedding.
        hidden = self.vision_tower.vision_model.embeddings(flat)

        # Build temporal attention mask. Skipped at T=1 because the wrapper's
        # T=1 short-circuit bypasses temporal attention entirely.
        temporal_attn_mask: Tensor | None = None
        if t > 1:
            if obs_history_is_pad is not None:
                temporal_attn_mask = self._build_temporal_attn_mask(
                    obs_history_is_pad, self.num_video_tokens, hidden.dtype
                )
            else:
                # Inference fallback: select_action -> _build_history_batch
                # zero-pads missing slots but does NOT emit obs_history_is_pad.
                # Treat all history as padded so the current frame's
                # representation is uncontaminated by the zero-pixel
                # placeholders.
                fallback_pad = torch.ones(b, t, dtype=torch.bool, device=hidden.device)
                fallback_pad[:, -1] = False
                temporal_attn_mask = self._build_temporal_attn_mask(
                    fallback_pad, self.num_video_tokens, hidden.dtype
                )

        # Encoder stack: standard spatial layers + wrapped every-Nth layer
        # with temporal attention. SpaceTimeEncoderLayerWrapper matches the
        # SiglipEncoderLayer signature, so we drive the loop manually here
        # (instead of calling SiglipEncoder.forward) so we can wrap each
        # layer in torch.utils.checkpoint.checkpoint when the flag is set —
        # the same explicit pattern PaliGemmaWithExpertModel uses.
        # ``temporal_attn_mask`` and ``num_frames`` are passed only to
        # spacetime-wrapped layers (vanilla layers don't accept them). Under
        # gradient checkpointing they MUST go in as positional args —
        # torch.utils.checkpoint.checkpoint with use_reentrant=False does not
        # forward kwargs to the wrapped function.
        use_ckpt = self.gradient_checkpointing and self.training
        for layer in self.vision_tower.vision_model.encoder.layers:
            is_spacetime = isinstance(layer, SpaceTimeEncoderLayerWrapper)
            if use_ckpt:
                if is_spacetime:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        layer, hidden, None, False, temporal_attn_mask, t, use_reentrant=False
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        layer, hidden, None, False, use_reentrant=False
                    )
            else:
                if is_spacetime:
                    layer_outputs = layer(
                        hidden, None, False, temporal_attn_mask=temporal_attn_mask, num_frames=t
                    )
                else:
                    layer_outputs = layer(hidden, None, False)
            hidden = layer_outputs[0]

        hidden = self.vision_tower.vision_model.post_layernorm(hidden)

        # Drop past-timestep tokens: keep only the current frame (t = T-1).
        # This matches the MEM paper's "we only pass the representation
        # computed for the current timestep onwards" and makes the encoder
        # a drop-in replacement for a single-frame vision tower.
        hidden = rearrange(hidden, "(b t) n d -> b t n d", b=b, t=t)
        current = hidden[:, -1]

        # multi_modal_projector: SigLIP hidden (1152) -> VLA hidden (2048).
        # We deliberately omit the `/ sqrt(hidden_size)` division to match
        # the patched ``PaliGemmaModel.get_image_features`` (see
        # ``opentau.utils.transformers_patch``).
        return self.multi_modal_projector(current)
