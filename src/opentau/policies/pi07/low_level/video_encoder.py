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

The encoder does NOT own its own copy of the SigLIP weights. The caller
(``PI07LowLevelFlowMatching``) constructs a ``Gemma3WithExpertModel``
— which already owns ``vision_tower`` and ``multi_modal_projector`` (under
``gemma3.model``) — and passes them in by reference. This avoids duplicating
~400M parameters in memory.
"""

import math
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
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
    max_period: float = 4.0,
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
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2")
    if num_frames < 1:
        raise ValueError(f"num_frames ({num_frames}) must be >= 1")

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


class _TemporalSelfAttention(nn.Module):
    """Parameter-free causal temporal self-attention.

    Re-uses an existing ``SiglipAttention`` instance's
    ``q_proj``/``k_proj``/``v_proj``/``out_proj`` linear layers, but applies
    them over the ``T`` axis (for each fixed patch position) with a
    standard lower-triangular causal mask (position ``i`` attends to
    positions ``j <= i``; since ``t = T-1`` is the current frame, the current
    frame attends to all past frames).

    The referenced ``SiglipAttention`` is held in a list to keep ``nn.Module``
    from re-registering its parameters under this module's path (which would
    duplicate them in ``state_dict`` under both
    ``base_layer.self_attn.*`` and ``_temporal_attn.attn.*``).
    """

    def __init__(self, attn: SiglipAttention):
        super().__init__()
        # Wrap in a list so nn.Module.__setattr__ does not treat ``attn``
        # as a child submodule; the base layer already owns these params.
        self._attn_ref: list[SiglipAttention] = [attn]

    @property
    def attn(self) -> SiglipAttention:
        return self._attn_ref[0]

    def forward(self, hidden_states: Tensor, temporal_attn_mask: Tensor | None = None) -> Tensor:
        """hidden_states: (B*N, T, D) -> (B*N, T, D).

        Args:
            hidden_states: ``(B*N, T, D)`` hidden states reshaped for temporal
                attention (one sequence per spatial patch position).
            temporal_attn_mask: Optional ``(B*N, 1, T, T)`` additive float mask.
                ``0.0`` = attend, ``-inf`` = block. When ``None``, falls back to
                the standard causal lower-triangular mask via ``is_causal=True``.
        """
        attn = self.attn
        bn, t, d = hidden_states.shape
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        q = attn.q_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)

        if temporal_attn_mask is not None:
            # Caller-supplied mask already encodes both causal AND padded-history
            # blocking; do NOT also set is_causal=True (SDPA disallows combining
            # the two).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=temporal_attn_mask, is_causal=False, dropout_p=0.0, scale=attn.scale
            )
        else:
            # is_causal=True -> lower-triangular mask, each position attends to
            # itself and earlier positions (our convention: t=T-1 is current).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0, scale=attn.scale
            )
        out = out.transpose(1, 2).reshape(bn, t, d)
        return attn.out_proj(out)


class SpaceTimeEncoderLayerWrapper(nn.Module):
    """Replaces a ``SiglipEncoderLayer`` in-place; adds a temporal sublayer.

    The wrapper **adopts** the original layer's submodules by reference —
    ``self_attn``, ``layer_norm1``, ``layer_norm2``, ``mlp`` — so its
    ``state_dict`` keys are **identical** to a vanilla ``SiglipEncoderLayer``.
    That means any pi05 / pi05_continuous_state checkpoint can load directly
    into the wrapped layer without any key remapping. The only new state is
    a non-persistent ``_temporal_pe`` buffer (excluded from state_dict) and
    an internal ``_temporal_attn`` wrapper that holds a by-reference pointer
    to ``self_attn`` (also excluded because it's kept in a list).

    The forward computes:

        h_pe  = h + e(t)                                   # broadcast over (B, N)
        h     = h + temporal_attn( LN1(h_pe) )             # new; causal over T
        # then the standard SigLIP block:
        h     = h + spatial_attn( LN1(h) )
        h     = h + MLP( LN2(h) )

    At ``T=1`` the temporal sublayer is skipped entirely so that the block
    degenerates to the unmodified SigLIP forward, satisfying the MEM paper's
    single-frame invariance claim. (With ``T=1`` causal attention over a
    single timestep is not an identity — it returns ``out_proj(v_proj(
    LN1(x)))`` — so ``e(0)=0`` alone is insufficient; the block itself must
    also be bypassed.)

    Reusing ``layer_norm1`` for both the temporal and spatial sublayers keeps
    the paper's "no new learnable parameters" guarantee. It is an intentional
    design choice: the two attentions operate on different axes and the
    LayerNorm is applied to different input tensors each time.
    """

    # Match the SiglipEncoderLayer class attribute so transformers'
    # gradient-checkpointing plumbing sees a familiar interface.
    gradient_checkpointing: bool = False

    def __init__(
        self,
        base_layer: SiglipEncoderLayer,
        num_frames: int,
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

        self.num_frames = num_frames
        self.num_tokens_per_frame = num_tokens_per_frame
        # The temporal attention re-uses self_attn's Q/K/V/O projections; it
        # holds its reference in a list (see _TemporalSelfAttention) so the
        # params don't show up twice in state_dict.
        self._temporal_attn = _TemporalSelfAttention(self.self_attn)

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
        # (the normal load flow for pi05_mem does
        # ``paligemma = ...from_pretrained(...).to('cuda')`` and then wraps);
        # with no parent ``.to(device)`` happening after wrapping, a PE built
        # on CPU would stay on CPU and trigger a cross-device RuntimeError at
        # forward time. Pinning to the base layer's device sidesteps that.
        ref_param = base_layer.self_attn.q_proj.weight
        pe = _build_temporal_sinusoidal_pe(
            num_frames, self.embed_dim, dtype=ref_param.dtype, device=ref_param.device
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
    ) -> tuple[Tensor, ...]:
        """hidden_states: (B*T, N, D) -> tuple starting with (B*T, N, D).

        Signature extends ``SiglipEncoderLayer.forward`` with an extra
        ``temporal_attn_mask`` kwarg. Vanilla ``SiglipEncoderLayer`` instances
        never receive it: ``SpaceTimeSiglipVideoEncoder.forward`` dispatches
        it only to ``SpaceTimeEncoderLayerWrapper`` instances via an
        ``isinstance`` check, bypassing ``SiglipEncoder.forward`` entirely.

        Args:
            temporal_attn_mask: Optional ``(B*N, 1, T, T)`` additive float mask
                for temporal attention. ``0.0`` = attend, ``-inf`` = block.
                When ``None``, the standard causal mask is used.
        """
        t = self.num_frames
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

        # Short-circuit at T=1: temporal self-attention over a single timestep
        # collapses to ``out_proj(v_proj(LN1(x)))``, which is NOT an identity
        # and would break single-frame invariance (the MEM paper's guarantee
        # that a T=1 pass matches the unmodified SigLIP ViT). e(t=0)=0 alone
        # is insufficient; the block must also be skipped.
        if t == 1:
            return self._spatial_block_forward(hidden_states, attention_mask, output_attentions)

        if bt % t != 0:
            raise ValueError(
                f"hidden_states.shape[0] ({bt}) must be divisible by num_frames ({t}); "
                "video encoder expects inputs flattened as (B*T, N, D). "
                "Use `suppress_spacetime_temporal(...)` for non-video forwards."
            )
        b = bt // t

        # Temporal sublayer.
        x = rearrange(hidden_states, "(b t) n d -> b t n d", b=b, t=t)
        # Cast PE to match tensor device/dtype each call. Both are no-ops if
        # already aligned (the common case — the buffer is constructed on
        # the base layer's device). The cast only allocates when something
        # external has moved the inputs onto a different device without
        # propagating ``.to()`` through to this wrapper.
        pe = self._temporal_pe.to(device=x.device, dtype=x.dtype).view(1, t, 1, d)
        x_pe = x + pe

        t_in = rearrange(x_pe, "b t n d -> (b n) t d")
        t_norm = self.layer_norm1(t_in)
        t_out = self._temporal_attn(t_norm, temporal_attn_mask=temporal_attn_mask)
        # Residual on the pre-PE hidden (not on x_pe): PE is a transient
        # positional signal, not a feature perturbation to carry forward.
        t_res = rearrange(x, "b t n d -> (b n) t d") + t_out
        h_after_t = rearrange(t_res, "(b n) t d -> (b t) n d", n=n)

        # Spatial + MLP sublayers.
        return self._spatial_block_forward(h_after_t, attention_mask, output_attentions)


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
    layer. In practice the only caller is ``PI07LowLevelFlowMatching``,
    which passes in the ``Gemma3WithExpertModel``'s SigLIP vision components
    (resolved via ``_vision_tower()`` / ``_multi_modal_projector()``).
    """

    def __init__(
        self,
        vision_tower: SiglipVisionModel,
        multi_modal_projector: nn.Module,
        num_frames: int,
        spacetime_layer_stride: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames ({num_frames}) must be >= 1.")
        if spacetime_layer_stride < 1:
            raise ValueError(f"spacetime_layer_stride ({spacetime_layer_stride}) must be >= 1.")

        self.num_frames = num_frames
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
        # by the caller (Gemma3WithExpertModel); double registration would
        # duplicate ~400M params in state_dict.
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
        # prefix appears. Pinned by ``test_pi07_video_encoder_cpu.py::
        # test_state_dict_keys_unchanged_after_wrapping``.
        layers = vision_tower.vision_model.encoder.layers
        n_layers = len(layers)
        for i in range(spacetime_layer_stride - 1, n_layers, spacetime_layer_stride):
            layers[i] = SpaceTimeEncoderLayerWrapper(
                base_layer=layers[i],
                num_frames=num_frames,
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
                ``T == num_frames``, ``C == 3``, and spatial size matching
                the SigLIP config (224x224 by default).
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
        if t != self.num_frames:
            raise ValueError(
                f"Expected T={self.num_frames} frames; got {t}. "
                "Reinstantiate the encoder with a matching num_frames."
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
        # ``temporal_attn_mask`` is passed only to spacetime-wrapped layers
        # (vanilla layers don't accept it). Under gradient checkpointing it
        # MUST go in as a positional arg — torch.utils.checkpoint.checkpoint
        # with use_reentrant=False does not forward kwargs to the wrapped
        # function.
        use_ckpt = self.gradient_checkpointing and self.training
        for layer in self.vision_tower.vision_model.encoder.layers:
            is_spacetime = isinstance(layer, SpaceTimeEncoderLayerWrapper)
            if use_ckpt:
                if is_spacetime:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        layer, hidden, None, False, temporal_attn_mask, use_reentrant=False
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        layer, hidden, None, False, use_reentrant=False
                    )
            else:
                if is_spacetime:
                    layer_outputs = layer(hidden, None, False, temporal_attn_mask=temporal_attn_mask)
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
