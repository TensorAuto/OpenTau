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

"""CPU-only tests for ``opentau.policies.pi07.video_encoder``.

This is the **canonical** test file for the unified ``SpaceTimeSiglipVideoEncoder``.
All three callers — pi05_mem (PaliGemma backbone), pi07/low_level (Gemma 3
backbone), and pi07_paligemma/low_level_planner (legacy PaliGemma backbone) —
share the same encoder implementation, so coverage lives here in one place.

Most tests are parametrized over both ``gemma3`` and ``paligemma`` projector
fixtures so the PaliGemma callers (which use ``PaliGemmaMultiModalProjector``)
get real coverage alongside the pi07/low_level Gemma 3 path.

The tests pin four guarantees the planners rely on:

  * Wrapping every ``stride``-th SigLIP layer with space-time attention
    leaves the wrapped vision_tower's state_dict keys identical to the
    vanilla SigLIP — a vanilla pi05 / pi07 / pi07_paligemma checkpoint loads
    in untouched.
  * At ``T=1`` the wrapper short-circuits to the vanilla spatial block, so a
    stride-999 (no-wrapping) encoder and a stride-4 (6 layers wrapped)
    encoder built on the same weights produce byte-identical outputs.
  * The ``suppress_spacetime_temporal`` context manager suppresses temporal
    attention for non-video forwards while leaving the spatial block
    untouched.
  * Variable ``T`` per forward: the encoder accepts any ``1 <= T <=
    max_num_frames``. PE values for the actual ``T`` rows of the cached
    buffer (the LAST ``T`` rows) are byte-identical to a fresh PE built for
    that ``T``, so single-frame invariance and the zero-current-row property
    hold at every ``T``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from opentau.policies.pi07.video_encoder import (
    SpaceTimeEncoderLayerWrapper,
    SpaceTimeSiglipVideoEncoder,
    _build_temporal_sinusoidal_pe,
    suppress_spacetime_temporal,
)

# ---------------------------------------------------------------------------
# Backbone fixtures: both Gemma 3 (pi07/low_level) and PaliGemma (pi05_mem,
# pi07_paligemma) projector variants are exercised.
# ---------------------------------------------------------------------------


def _build_gemma3_siglip_and_projector():
    """SigLIP + Gemma 3 multi-modal projector. Used by pi07/low_level."""
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

    vision_cfg_dict = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "image_size": 224,
        "projection_dim": 2560,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    }
    vision_tower = SiglipVisionModel(SiglipVisionConfig(**vision_cfg_dict))
    g3_cfg = Gemma3Config(
        vision_config=vision_cfg_dict,
        text_config={
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "vocab_size": 262_208,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
        mm_tokens_per_image=256,
        boi_token_index=255_999,
        eoi_token_index=256_000,
        image_token_index=262_144,
    )
    projector = Gemma3MultiModalProjector(g3_cfg)
    return vision_tower, projector, 2560


def _build_paligemma_siglip_and_projector():
    """SigLIP + PaliGemma multi-modal projector. Used by pi05_mem / pi07_paligemma."""
    from transformers import PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaMultiModalProjector,
    )

    vision_cfg_dict = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "num_image_tokens": 256,
        "patch_size": 14,
        "image_size": 224,
        "projection_dim": 2048,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    }
    vision_tower = SiglipVisionModel(SiglipVisionConfig(**vision_cfg_dict))
    pali_cfg = PaliGemmaConfig(
        vision_config=vision_cfg_dict,
        text_config={"hidden_size": 2048, "model_type": "gemma", "vocab_size": 257152},
    )
    projector = PaliGemmaMultiModalProjector(pali_cfg)
    return vision_tower, projector, 2048


_BACKBONES = {
    "gemma3": _build_gemma3_siglip_and_projector,
    "paligemma": _build_paligemma_siglip_and_projector,
}


@pytest.fixture(params=list(_BACKBONES.keys()))
def backbone(request):
    """Yield a fresh (vision_tower, projector, vlm_hidden_size) triple for each backbone."""
    return _BACKBONES[request.param]()


def _make_encoder_from_backbone(backbone_triple, *, max_num_frames: int = 2, spacetime_stride: int = 4):
    vision_tower, projector, vlm_hidden = backbone_triple
    encoder = SpaceTimeSiglipVideoEncoder(
        vision_tower=vision_tower,
        multi_modal_projector=projector,
        max_num_frames=max_num_frames,
        spacetime_layer_stride=spacetime_stride,
    )
    return encoder, vlm_hidden


def _make_gemma3_encoder(max_num_frames: int = 2, spacetime_stride: int = 4):
    """Backbone-agnostic helper for tests that don't need backbone parametrization."""
    return _make_encoder_from_backbone(
        _build_gemma3_siglip_and_projector(),
        max_num_frames=max_num_frames,
        spacetime_stride=spacetime_stride,
    )[0]


# ---------------------------------------------------------------------------
# Temporal sinusoidal PE
# ---------------------------------------------------------------------------


class TestTemporalSinusoidalPE:
    def test_current_frame_row_is_zero(self):
        pe = _build_temporal_sinusoidal_pe(num_frames=8, embed_dim=64)
        assert pe.shape == (8, 64)
        # Current frame lives at t=T-1; row must be all zeros so a T=1 pass
        # collapses to the unmodified SigLIP forward.
        assert torch.all(pe[-1] == 0)

    @pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("embed_dim", [16, 64, 1152])
    def test_pe_is_zero_at_t_eq_0(self, num_frames: int, embed_dim: int):
        """MEM Appendix C boundary condition: ``e(t=0) = 0``.

        The codebase convention places the current frame at array index
        ``T-1`` (so ``time[T-1] = 0``); this test asserts the boundary holds
        across every ``T`` the wrapper might encounter at runtime and across
        ``embed_dim`` values spanning small fixtures up to the real SigLIP
        hidden size (1152). Required for: (a) the K=1 natural identity in
        Reading B (the temporal SDPA over a single key returns V unchanged
        only if ``e(t=0)=0`` keeps ``h_pe = h``), and (b) the variable-T
        slicing trick — ``pe[max_num_frames - T:]`` is byte-identical to a
        fresh PE only because each PE's last row is zero by construction.
        """
        pe = _build_temporal_sinusoidal_pe(num_frames=num_frames, embed_dim=embed_dim)
        assert pe.shape == (num_frames, embed_dim)
        # The "t=0" row: array index T-1 (current frame).
        torch.testing.assert_close(
            pe[num_frames - 1],
            torch.zeros(embed_dim, dtype=pe.dtype),
            rtol=0,
            atol=0,
        )

    def test_earlier_rows_are_nonzero(self):
        pe = _build_temporal_sinusoidal_pe(num_frames=8, embed_dim=64)
        assert torch.any(pe[0] != 0)

    def test_single_frame_produces_zero_row(self):
        pe = _build_temporal_sinusoidal_pe(num_frames=1, embed_dim=64)
        assert pe.shape == (1, 64)
        assert torch.all(pe == 0)

    def test_odd_embed_dim_raises(self):
        with pytest.raises(ValueError, match="divisible by 2"):
            _build_temporal_sinusoidal_pe(num_frames=4, embed_dim=63)

    def test_zero_num_frames_raises(self):
        with pytest.raises(ValueError, match="num_frames"):
            _build_temporal_sinusoidal_pe(num_frames=0, embed_dim=64)

    def test_pe_supports_20_frames_without_aliasing(self):
        """The default ``max_period`` must comfortably cover ``T = 20``: every
        pair of distinct timesteps in ``{-19, ..., 0}`` should have visibly
        different sinusoidal rows. Pre-fix code (``max_period = 4.0``) failed
        this — the lowest-frequency component completed ~5 full cycles over
        the time range, so e.g. rows ``t = -16`` and ``t = -12`` collapsed
        to nearly-identical encodings in the long-period dims.
        """
        pe = _build_temporal_sinusoidal_pe(num_frames=20, embed_dim=64)
        assert pe.shape == (20, 64)
        # Current frame still zero (boundary preserved across max_period change).
        torch.testing.assert_close(pe[-1], torch.zeros(64, dtype=pe.dtype), rtol=0, atol=0)
        # Any two distinct rows must differ by at least a measurable margin.
        for i in range(20):
            for j in range(i + 1, 20):
                gap = (pe[i] - pe[j]).abs().max().item()
                assert gap > 1e-2, (
                    f"rows t={i - 19} and t={j - 19} differ by only {gap:.2e}; "
                    "max_period likely too small for T=20 (temporal aliasing)"
                )

    def test_max_period_too_small_raises(self):
        """Asking for more frames than ``max_period`` can encode without
        aliasing must raise a clear error rather than silently producing a
        degenerate PE.
        """
        with pytest.raises(ValueError, match="aliasing"):
            _build_temporal_sinusoidal_pe(num_frames=42, embed_dim=64)
        # Passing a larger max_period explicitly clears the guard.
        pe = _build_temporal_sinusoidal_pe(num_frames=42, embed_dim=64, max_period=128.0)
        assert pe.shape == (42, 64)

    def test_pe_slice_matches_fresh_build(self):
        """The MEM-paper PE values depend only on the relative offset from the
        current frame. Slicing ``pe[M-T:]`` of a PE built for ``M`` frames must
        therefore be byte-identical to a PE freshly built for ``T`` frames —
        this is what the wrapper relies on for variable-T support.
        """
        for max_t in (4, 8, 16):
            pe_max = _build_temporal_sinusoidal_pe(num_frames=max_t, embed_dim=64)
            for t in range(1, max_t + 1):
                pe_t = _build_temporal_sinusoidal_pe(num_frames=t, embed_dim=64)
                torch.testing.assert_close(pe_max[max_t - t :], pe_t, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Wrapper structure / state_dict invariance
# ---------------------------------------------------------------------------


class TestSpaceTimeWrapperStructure:
    def test_wrapped_layer_count(self):
        """Verify every ``stride``-th layer (1-indexed from the Nth) is wrapped."""
        encoder = _make_gemma3_encoder(max_num_frames=2, spacetime_stride=4)
        layers = encoder.vision_tower.vision_model.encoder.layers
        wrapped_indices = [
            i for i, layer in enumerate(layers) if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        ]
        # 27 SigLIP layers; every 4th starting at index 3 → [3, 7, 11, 15, 19, 23]
        assert wrapped_indices == [3, 7, 11, 15, 19, 23]

    def test_state_dict_keys_match_vanilla_siglip(self, backbone):
        """The wrapped vision_tower's state_dict must have identical keys to
        an unwrapped SigLIP — guarantees that any pi05/pi07/pi07_paligemma
        checkpoint loads without key remapping."""
        from transformers import SiglipVisionModel

        vision_tower, projector, _ = backbone
        reference_keys = set(SiglipVisionModel(vision_tower.config).state_dict().keys())

        SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=4,
            spacetime_layer_stride=4,
        )
        wrapped_keys = set(vision_tower.state_dict().keys())
        assert wrapped_keys == reference_keys, (
            f"state_dict keys diverged; "
            f"extra in wrapped: {wrapped_keys - reference_keys}, "
            f"missing from wrapped: {reference_keys - wrapped_keys}"
        )

    def test_temporal_pe_on_base_layer_device(self, backbone):
        """The temporal PE buffer must be constructed on the base layer's
        device/dtype, not default CPU.

        Regression for the pi05_mem GPU bug where the parent vision_tower had
        already been moved to a non-default dtype/device BEFORE wrapping —
        leaving the wrapper's PE on CPU/float32.
        """
        vision_tower, projector, _ = backbone
        vision_tower = vision_tower.to(dtype=torch.bfloat16)
        projector = copy.deepcopy(projector).to(dtype=torch.bfloat16)

        SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=4,
            spacetime_layer_stride=4,
        )

        ref_param = next(vision_tower.parameters())
        for layer in vision_tower.vision_model.encoder.layers:
            if isinstance(layer, SpaceTimeEncoderLayerWrapper):
                assert layer._temporal_pe.device == ref_param.device
                assert layer._temporal_pe.dtype == ref_param.dtype


# ---------------------------------------------------------------------------
# Forward shape / dtype / arg validation
# ---------------------------------------------------------------------------


class TestSpaceTimeForward:
    def test_forward_shape(self, backbone):
        encoder, vlm_hidden = _make_encoder_from_backbone(backbone, max_num_frames=2)
        encoder.eval()
        with torch.no_grad():
            video = torch.rand(1, 2, 3, 224, 224)
            out = encoder(video)
        # 16 patches/side at 224/14 → 256 tokens; the projector outputs the
        # backbone's text-hidden width (2560 for Gemma 3, 2048 for PaliGemma).
        assert out.shape == (1, 256, vlm_hidden)
        assert out.dtype == torch.float32

    def test_t_above_max_raises(self):
        """T > max_num_frames must be rejected (the cached PE buffer would be
        too short to slice; reinstantiating with a larger ``max_num_frames``
        is the explicit fix)."""
        encoder = _make_gemma3_encoder(max_num_frames=2)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="max_num_frames"):
            encoder(torch.rand(1, 4, 3, 224, 224))

    def test_wrong_ndim_raises(self):
        encoder = _make_gemma3_encoder(max_num_frames=2)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="5D"):
            encoder(torch.rand(2, 3, 224, 224))

    def test_zero_t_raises(self):
        encoder = _make_gemma3_encoder(max_num_frames=4)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="T >= 1"):
            encoder(torch.rand(1, 0, 3, 224, 224))

    def test_forward_after_external_dtype_cast(self):
        """End-to-end forward must succeed when vision_tower was moved to a
        different dtype BEFORE wrapping (mirrors the GPU load flow of
        ``gemma3.to(cuda/bf16)`` then construct encoder).
        """
        vision_tower, projector, vlm_hidden = _build_gemma3_siglip_and_projector()
        vision_tower = vision_tower.to(dtype=torch.bfloat16)
        projector = copy.deepcopy(projector).to(dtype=torch.bfloat16)

        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=4,
            spacetime_layer_stride=4,
        ).eval()
        with torch.no_grad():
            out = encoder(torch.rand(1, 4, 3, 224, 224, dtype=torch.bfloat16))
        assert out.shape == (1, 256, vlm_hidden)
        assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Variable-T support: a single encoder accepts any T in [1, max_num_frames].
# ---------------------------------------------------------------------------


class TestVariableNumFrames:
    """Pins the property that ``max_num_frames`` is a cap, not a fixed T —
    the encoder accepts any ``T`` in ``[1, max_num_frames]`` per forward and
    the result is identical to instantiating a separate encoder with
    ``max_num_frames=T``."""

    def test_runs_at_each_t_up_to_max(self):
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4).eval()
        for t in (1, 2, 3, 4):
            with torch.no_grad():
                out = encoder(torch.rand(1, t, 3, 224, 224))
            assert out.shape == (1, 256, 2560), f"failed at T={t}"

    def test_runs_at_t_below_max_with_pad_mask(self):
        """Variable T with an explicit ``obs_history_is_pad`` of matching
        shape — exercises both the temporal-mask construction and the PE
        slice for an off-max T."""
        encoder = _make_gemma3_encoder(max_num_frames=8, spacetime_stride=4).eval()
        with torch.no_grad():
            video = torch.rand(2, 3, 3, 224, 224)
            pad = torch.tensor([[True, False, False], [False, False, False]])
            out = encoder(video, obs_history_is_pad=pad)
        assert out.shape == (2, 256, 2560)

    def test_output_matches_max_equals_t_encoder(self):
        """A forward at ``T`` through an encoder built with ``max_num_frames=M``
        must be byte-identical to a forward through an encoder built with
        ``max_num_frames=T`` on the same weights — the cached PE for ``M``
        is sliced to its last ``T`` rows, which equals a fresh ``T``-row PE
        (verified at the unit level in ``test_pe_slice_matches_fresh_build``).
        """
        torch.manual_seed(0)
        vision_tower, projector, _ = _build_gemma3_siglip_and_projector()

        # Encoder built for max_num_frames=8 — will be invoked at T=3.
        vt_big = copy.deepcopy(vision_tower)
        proj_big = copy.deepcopy(projector)
        enc_big = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_big,
            multi_modal_projector=proj_big,
            max_num_frames=8,
            spacetime_layer_stride=4,
        ).eval()

        # Encoder built for max_num_frames=3 (matches the T we'll feed).
        enc_tight = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=3,
            spacetime_layer_stride=4,
        ).eval()

        video = torch.rand(2, 3, 3, 224, 224)
        pad = torch.tensor([[True, False, False], [False, False, False]])
        with torch.no_grad():
            out_big = enc_big(video, obs_history_is_pad=pad)
            out_tight = enc_tight(video, obs_history_is_pad=pad)
        torch.testing.assert_close(out_big, out_tight, rtol=1e-5, atol=1e-5)

    def test_single_frame_invariance_holds_for_any_max(self):
        """Single-frame invariance must hold regardless of how the encoder
        was sized — at T=1 the wrapper short-circuits, so even an encoder
        built with ``max_num_frames=16`` matches a no-wrapping encoder.
        """
        torch.manual_seed(0)
        vision_tower, projector, _ = _build_gemma3_siglip_and_projector()
        vt_no_st = copy.deepcopy(vision_tower)
        proj_no_st = copy.deepcopy(projector)

        enc_no_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_no_st,
            multi_modal_projector=proj_no_st,
            max_num_frames=1,
            spacetime_layer_stride=999,
        ).eval()
        enc_st_big = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=16,
            spacetime_layer_stride=4,
        ).eval()

        video = torch.rand(1, 1, 3, 224, 224)
        with torch.no_grad():
            out_no_st = enc_no_st(video)
            out_st = enc_st_big(video)
        torch.testing.assert_close(out_st, out_no_st, rtol=1e-5, atol=1e-5)

    def test_wrapper_rejects_t_above_max(self):
        """The wrapper itself must reject ``num_frames > max_num_frames`` —
        defensive guard since the encoder also checks at the outer boundary,
        but a direct caller of the wrapper (e.g. a future custom encoder
        loop) should still get a clean error.
        """
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)
        wrapper = next(
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        )
        # 256 = (224/14)**2 patches per frame; bt = 8*256 patches but
        # num_frames=8 exceeds max_num_frames=4.
        bad_input = torch.rand(8, wrapper.num_tokens_per_frame, wrapper.embed_dim)
        with pytest.raises(ValueError, match="max_num_frames"):
            wrapper.forward(bad_input, num_frames=8)


# ---------------------------------------------------------------------------
# Single-frame invariance
# ---------------------------------------------------------------------------


class TestSingleFrameInvariance:
    def test_single_frame_invariance_structural(self, backbone):
        """At T=1 the wrapper must short-circuit: byte-identical output to a
        stride-999 (no-wrapping) encoder sharing the exact same underlying
        weights. Holds for both Gemma 3 and PaliGemma projectors.
        """
        torch.manual_seed(0)
        vision_tower, projector, _ = backbone
        vt_no_st = copy.deepcopy(vision_tower)
        proj_no_st = copy.deepcopy(projector)

        enc_no_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_no_st,
            multi_modal_projector=proj_no_st,
            max_num_frames=1,
            spacetime_layer_stride=999,  # > 27 → no layers wrapped
        ).eval()
        enc_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=1,
            spacetime_layer_stride=4,  # 6 layers wrapped
        ).eval()

        # Adopt-submodules wrapping doesn't introduce ``.base_layer.`` keys,
        # so the two state_dicts have identical keys; weights are still
        # identical because we deep-copied.
        assert set(vt_no_st.state_dict().keys()) == set(vision_tower.state_dict().keys())

        video = torch.rand(1, 1, 3, 224, 224)
        with torch.no_grad():
            out_no_st = enc_no_st(video)
            out_st = enc_st(video)

        torch.testing.assert_close(out_st, out_no_st, rtol=1e-5, atol=1e-5)

    def test_t1_matches_vanilla_siglip_vision_model(self, backbone):
        """At T=1 the SpaceTime SigLIP video encoder must match the output of
        an unmodified ``SiglipVisionModel`` (the upstream HuggingFace vision
        encoder, never wrapped) on the same image, modulo the shared
        ``multi_modal_projector``.

        This is the direct restatement of the wrapper's "no new params, no
        behavior change at T=1" claim against the upstream SigLIP class —
        it does NOT rely on ``suppress_spacetime_temporal`` (the bypass flag)
        or on a sibling encoder built with stride=999. The wrapper's
        natural T=1 short-circuit (and equivalently, Reading B's natural
        identity at T=1: e(t=0)=0 and a single-key temporal SDPA) must
        produce the same hidden states as the un-instrumented SigLIP
        forward, so a deepcopy of the same vision_tower (built BEFORE
        wrapping) gives byte-identical output.
        """
        torch.manual_seed(0)
        vision_tower, projector, _ = backbone

        # Snapshot the un-instrumented SigLIP weights BEFORE constructing the
        # encoder (which mutates ``vision_tower`` in-place by replacing every
        # stride-th layer with a SpaceTimeEncoderLayerWrapper). The deepcopy
        # is a true vanilla SiglipVisionModel — no wrapper, no PE buffer.
        vt_vanilla = copy.deepcopy(vision_tower)
        proj_vanilla = copy.deepcopy(projector)
        vt_vanilla.eval()
        proj_vanilla.eval()

        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=4,
            spacetime_layer_stride=4,  # 6 layers wrapped
        ).eval()

        image_unit = torch.rand(2, 3, 224, 224)
        # Encoder rescales [0, 1] → [-1, 1] internally; the vanilla SigLIP
        # forward expects pre-rescaled pixels.
        image_siglip = image_unit * 2.0 - 1.0

        with torch.no_grad():
            out_video = encoder(image_unit.unsqueeze(1))  # (B, 1, C, H, W)
            last_hidden = vt_vanilla(pixel_values=image_siglip).last_hidden_state
            out_vanilla = proj_vanilla(last_hidden)

        torch.testing.assert_close(out_video, out_vanilla, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# `suppress_spacetime_temporal` context manager
# ---------------------------------------------------------------------------


class TestSuppressSpacetimeTemporal:
    def test_flag_toggles_and_restores(self):
        """Entering the context manager flips every wrapper to inactive;
        exiting restores the prior value (idempotent — works nested)."""
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)
        wrappers = [
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        ]
        assert wrappers, "expected the encoder to wrap at least one layer"
        assert all(w._temporal_active for w in wrappers)

        with suppress_spacetime_temporal(encoder.vision_tower):
            assert all(not w._temporal_active for w in wrappers)
            with suppress_spacetime_temporal(encoder.vision_tower):
                assert all(not w._temporal_active for w in wrappers)
            # Inner context restored to outer (False), not to True.
            assert all(not w._temporal_active for w in wrappers)
        assert all(w._temporal_active for w in wrappers)

    def test_no_op_when_no_wrappers_present(self):
        """When the module subtree contains no wrappers (e.g. the high-level
        planner only ever calls ``embed_image`` on a vanilla SigLIP), the
        context manager must be a silent no-op."""
        from transformers import SiglipVisionConfig, SiglipVisionModel

        plain = SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                patch_size=14,
                image_size=224,
            )
        )
        # Should not raise.
        with suppress_spacetime_temporal(plain):
            pass

    def test_non_divisible_batch_raises_when_active(self):
        """When ``temporal_active=True`` (the default), the wrapper now
        explicitly rejects ``bt % t != 0`` — replacing a previous silent
        short-circuit which could have wrongly fired temporal attention over
        spatial-only inputs that happened to be divisible by ``num_frames``.
        """
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)
        wrapper = next(
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        )
        # 256 = (224/14)**2 patches per frame (the expected per-frame token count).
        bad_input = torch.rand(3, 256, wrapper.embed_dim)
        with pytest.raises(ValueError, match="divisible by num_frames"):
            wrapper.forward(bad_input, num_frames=4)

    def test_non_divisible_batch_succeeds_when_suppressed(self):
        """Inside ``suppress_spacetime_temporal`` the wrapper accepts any
        batch size and dispatches to the vanilla spatial block."""
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)
        wrapper = next(
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        )
        bad_input = torch.rand(3, 256, wrapper.embed_dim)
        with suppress_spacetime_temporal(encoder.vision_tower):
            out = wrapper.forward(bad_input)[0]
        assert out.shape == bad_input.shape


# ---------------------------------------------------------------------------
# Padded-history attention masking.
#
# Pixel-zeroing of padded frames is not enough — the SigLIP patch embedding
# carries a learned bias and the temporal positional embedding e(t) is
# non-zero for t < T-1. So zero pixels still produce non-zero hidden states
# that the current frame would otherwise attend to. The encoder must build
# an attention mask blocking padded keys at the SDPA call.
# ---------------------------------------------------------------------------


class TestTemporalAttentionMaskBlocksPaddedHistory:
    def test_current_frame_invariant_to_padded_history(self):
        """Gold-standard: with the first three frames marked padded and
        identical current frame at ``[:, -1]``, the encoder output must NOT
        depend on the pixel values of those padded frames.

        Pre-fix code reads contaminated hidden states from padded frames
        through the temporal attention path and the two outputs diverge.
        """
        torch.manual_seed(0)
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4).eval()

        current = torch.rand(1, 1, 3, 224, 224)
        # vid_a: random padded history.
        vid_a = torch.cat([torch.rand(1, 3, 3, 224, 224), current], dim=1)
        # vid_b: zeroed padded history; same current frame.
        vid_b = torch.cat([torch.zeros(1, 3, 3, 224, 224), current], dim=1)

        obs_history_is_pad = torch.tensor([[True, True, True, False]])
        with torch.no_grad():
            out_a = encoder(vid_a, obs_history_is_pad=obs_history_is_pad)
            out_b = encoder(vid_b, obs_history_is_pad=obs_history_is_pad)

        torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)

    def test_inference_fallback_blocks_all_history(self):
        """``obs_history_is_pad=None`` triggers the inference fallback that
        treats every history slot as padded; pins the property that
        ``select_action`` (which never emits ``obs_history_is_pad``) does
        not let zero-pixel placeholders contaminate the current frame.
        """
        torch.manual_seed(1)
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4).eval()

        current = torch.rand(1, 1, 3, 224, 224)
        vid_a = torch.cat([torch.rand(1, 3, 3, 224, 224), current], dim=1)
        vid_b = torch.cat([torch.zeros(1, 3, 3, 224, 224), current], dim=1)

        with torch.no_grad():
            out_a = encoder(vid_a)  # obs_history_is_pad defaults to None
            out_b = encoder(vid_b)

        torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)

    def test_temporal_mask_shape_and_values(self):
        """Exercise ``_build_temporal_attn_mask`` directly. Documents the
        contract: causal lower-triangular AND key-side ``~obs_history_is_pad``
        with current-frame override; (B*N, 1, T, T) shape; additive 0/-inf.
        """
        obs_history_is_pad = torch.tensor([[True, False, False]])
        mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
            obs_history_is_pad, num_patches=4, dtype=torch.float32
        )

        # Shape: B=1, N=4, T=3 → (B*N, 1, T, T) = (4, 1, 3, 3).
        assert mask.shape == (4, 1, 3, 3)
        assert mask.dtype == torch.float32

        # All N rows for batch 0 share the same (T, T) submatrix.
        m = mask[0, 0]  # (3, 3)

        # Diagonal: each query attends to itself when key is real.
        # Row 0: query=frame0 (padded query, but causal allows j=0 only;
        # since frame 0 is padded its key is blocked → -inf). The current-frame
        # override only forces key j=T-1 real, not j=0.
        assert m[0, 0] == float("-inf"), "padded frame 0 key should be blocked"
        # Row 1: query=frame1 attends to {0 (padded → -inf), 1 (real → 0)}.
        assert m[1, 0] == float("-inf")
        assert m[1, 1] == 0.0
        # Row 2: query=current attends to {0 (padded → -inf), 1 (real → 0),
        # 2 (current, always real → 0)}.
        assert m[2, 0] == float("-inf")
        assert m[2, 1] == 0.0
        assert m[2, 2] == 0.0

        # Upper triangular off-diagonals are blocked (causal).
        assert m[0, 1] == float("-inf")
        assert m[0, 2] == float("-inf")
        assert m[1, 2] == float("-inf")

        # Patch rows for the same batch are identical (mask is broadcast over
        # spatial patch positions).
        for n in range(1, 4):
            torch.testing.assert_close(mask[n, 0], mask[0, 0])

    def test_temporal_mask_all_valid_is_causal(self):
        """All frames valid (no padding): mask must be a standard
        lower-triangular causal mask where row i attends to columns j <= i.
        """
        bsz, t = 2, 4
        num_patches = 4
        dtype = torch.float32
        neginf = float("-inf")

        no_pad = torch.zeros(bsz, t, dtype=torch.bool)
        mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(no_pad, num_patches, dtype)

        assert mask.shape == (bsz * num_patches, 1, t, t)

        expected_bool = torch.tril(torch.ones(t, t, dtype=torch.bool))
        expected = torch.zeros(t, t, dtype=dtype)
        expected.masked_fill_(~expected_bool, neginf)

        for bi in range(bsz):
            sample_mask = mask[bi * num_patches, 0]
            torch.testing.assert_close(
                sample_mask,
                expected,
                msg=f"sample {bi}: all-valid mask should be standard causal",
            )

        attn_2d = mask[:, 0, :, :]
        num_attendable = (attn_2d > neginf).sum(dim=-1)  # (B*N, T)
        for row_i in range(1, t):
            assert torch.all(num_attendable[:, row_i] == row_i + 1), (
                f"row {row_i} should attend to {row_i + 1} positions"
            )

    def test_temporal_mask_mixed_batch_per_sample_independence(self):
        """Heterogeneous pad masks in a batch: each sample's (T, T) slice must
        match the mask built from that sample alone — no cross-sample leakage.
        """
        t = 4
        num_patches = 4
        dtype = torch.float32

        pad_mask = torch.tensor(
            [
                [True, True, True, False],
                [False, False, False, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        )
        bsz = pad_mask.shape[0]

        mask_batched = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(pad_mask, num_patches, dtype)
        assert mask_batched.shape == (bsz * num_patches, 1, t, t)

        for bi in range(bsz):
            mask_solo = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
                pad_mask[bi : bi + 1], num_patches, dtype
            )
            batched_slice = mask_batched[bi * num_patches : (bi + 1) * num_patches]
            torch.testing.assert_close(
                batched_slice,
                mask_solo,
                msg=f"sample {bi}: batched mask slice differs from solo build",
            )

        # All-padded vs all-valid samples should produce different masks.
        s0 = mask_batched[0, 0]
        s1 = mask_batched[1 * num_patches, 0]
        assert not torch.equal(s0, s1)

    def test_current_frame_override_when_pad_includes_current(self):
        """Defensive: even if a caller passes ``obs_history_is_pad`` with
        the current step (T-1) marked padded — as the dataset's
        ``history_state_drop_prob`` augmentation does — the mask must keep
        the current frame attendable, otherwise the current query has no
        valid key and softmax produces NaNs.
        """
        all_padded = torch.tensor([[True, True, True, True]])
        mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
            all_padded, num_patches=1, dtype=torch.float32
        )
        # Current row (T-1=3) attends to the current key (T-1=3) at minimum.
        m = mask[0, 0]
        assert m[3, 3] == 0.0, "current frame must remain self-attendable"

    def test_pad_tensor_not_mutated(self):
        """The mask helper must not mutate the caller's ``obs_history_is_pad``
        tensor (it is also consumed downstream as the state mask).
        """
        all_padded = torch.tensor([[True, True, True, True]])
        snapshot = all_padded.clone()
        SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(all_padded, num_patches=1, dtype=torch.float32)
        torch.testing.assert_close(all_padded, snapshot)

    def test_t_above_1_with_full_pad_matches_t1(self):
        """T>1 with every history frame masked must produce the same encoder
        output as a T=1 forward on the current frame alone.

        Reading B at the current-frame slot, when the temporal mask blocks
        every past key, the temporal SDPA reduces to ``V_temp[current] =
        V[current]`` (only the current key is attendable, attention weight is
        1.0). The spatial pass then sees the same Q, K, V the T=1 path would,
        so the current-frame output is mathematically identical to the T=1
        forward — modulo low-precision accumulation noise from the no-op
        temporal SDPA. This pins the "fully-masked-history is exactly T=1"
        identity that downstream callers rely on at inference time when
        ``_build_history_batch`` zero-pads missing history slots.
        """
        torch.manual_seed(7)
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4).eval()

        current = torch.rand(1, 1, 3, 224, 224)
        # Arbitrary past frames; their pixels must not influence the output
        # because every history slot is marked padded.
        history = torch.rand(1, 3, 3, 224, 224)
        video_t4 = torch.cat([history, current], dim=1)
        pad_full = torch.tensor([[True, True, True, False]])

        with torch.no_grad():
            out_t1 = encoder(current)
            out_t4 = encoder(video_t4, obs_history_is_pad=pad_full)

        torch.testing.assert_close(out_t1, out_t4, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Mask-structure invariants: Reading B's temporal pass uses a causal mask;
# the spatial pass is bidirectional (no mask). These tests intercept the
# SDPA calls inside a single wrapper forward and verify the call shape +
# mask kwargs match the paper's factorization.
# ---------------------------------------------------------------------------


class TestReadingBMaskStructure:
    """Verify the two SDPAs inside ``SpaceTimeEncoderLayerWrapper.forward``
    use the right mask types: temporal = causal, spatial = bidirectional.
    """

    @staticmethod
    def _grab_wrapper() -> SpaceTimeEncoderLayerWrapper:
        encoder = _make_gemma3_encoder(max_num_frames=2, spacetime_stride=4)
        wrappers = [
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        ]
        assert wrappers, "expected at least one SpaceTimeEncoderLayerWrapper"
        return wrappers[0]

    @staticmethod
    def _record_sdpa_calls():
        """Returns ``(install, restore, calls)`` — install the recorder around
        ``torch.nn.functional.scaled_dot_product_attention`` then restore
        the original. Records ``q_shape``, ``is_causal``, and the attn_mask
        identity / shape for each call.
        """
        import torch.nn.functional as F  # noqa: N812

        original = F.scaled_dot_product_attention
        calls: list[dict] = []

        def recorder(q, k, v, attn_mask=None, is_causal=False, **kwargs):
            calls.append(
                {
                    "q_shape": tuple(q.shape),
                    "is_causal": bool(is_causal),
                    "attn_mask_is_none": attn_mask is None,
                    "attn_mask_shape": tuple(attn_mask.shape) if attn_mask is not None else None,
                }
            )
            return original(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kwargs)

        def install():
            F.scaled_dot_product_attention = recorder

        def restore():
            F.scaled_dot_product_attention = original

        return install, restore, calls

    def test_temporal_is_causal_spatial_is_bidirectional_no_mask(self):
        """Without ``temporal_attn_mask``: the temporal SDPA must use
        ``is_causal=True`` (lower-triangular causal mask) and the spatial
        SDPA must use ``is_causal=False`` with ``attn_mask=None``
        (bidirectional, every patch attends to every other patch within
        the same timestep).
        """
        wrapper = self._grab_wrapper()
        t = 2
        n = wrapper.num_tokens_per_frame
        d = wrapper.embed_dim
        hidden = torch.randn(t, n, d)

        install, restore, calls = self._record_sdpa_calls()
        install()
        try:
            with torch.no_grad():
                wrapper(hidden, attention_mask=None, output_attentions=False, num_frames=t)
        finally:
            restore()

        # Reading B fires exactly two SDPAs per wrapper forward at T>1.
        assert len(calls) == 2, f"expected 2 SDPA calls (temporal + spatial), got {len(calls)}"

        temporal, spatial = calls
        # Temporal: layout (B*N, H, T, Dh); seq axis (dim 2) must equal T.
        assert temporal["q_shape"][2] == t, f"temporal seq axis expected T={t}, got {temporal['q_shape']}"
        assert temporal["is_causal"] is True, "temporal SDPA must be causal"
        assert temporal["attn_mask_is_none"], (
            "with no caller-supplied temporal_attn_mask, the temporal pass must rely on "
            "is_causal=True (SDPA disallows combining is_causal with attn_mask)"
        )

        # Spatial: layout (B*T, H, N, Dh); seq axis (dim 2) must equal N.
        assert spatial["q_shape"][2] == n, f"spatial seq axis expected N={n}, got {spatial['q_shape']}"
        assert spatial["is_causal"] is False, "spatial SDPA must be bidirectional (not causal)"
        assert spatial["attn_mask_is_none"], (
            "with no caller-supplied attention_mask, the spatial pass must use no mask "
            "(bidirectional attention over patches at the same timestep)"
        )

    def test_temporal_uses_provided_mask_spatial_still_bidirectional(self):
        """When ``temporal_attn_mask`` is supplied: the temporal SDPA must
        receive that mask (and ``is_causal=False`` so SDPA does not reject
        the combination), while the spatial SDPA must still be bidirectional
        with no mask.
        """
        wrapper = self._grab_wrapper()
        t = 2
        n = wrapper.num_tokens_per_frame
        d = wrapper.embed_dim
        hidden = torch.randn(t, n, d)

        # Build a real temporal mask via the encoder helper. Shape (B*N, 1, T, T).
        pad = torch.tensor([[True, False]])
        temporal_attn_mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
            pad, num_patches=n, dtype=hidden.dtype
        )

        install, restore, calls = self._record_sdpa_calls()
        install()
        try:
            with torch.no_grad():
                wrapper(
                    hidden,
                    attention_mask=None,
                    output_attentions=False,
                    temporal_attn_mask=temporal_attn_mask,
                    num_frames=t,
                )
        finally:
            restore()

        assert len(calls) == 2
        temporal, spatial = calls

        # Temporal: receives the explicit mask; is_causal must be False
        # (SDPA disallows is_causal=True together with attn_mask=...).
        assert temporal["is_causal"] is False, (
            "with caller-supplied temporal_attn_mask, the temporal SDPA must NOT also "
            "set is_causal=True (SDPA rejects the combination)"
        )
        assert not temporal["attn_mask_is_none"], "temporal SDPA must receive the supplied mask"
        assert temporal["attn_mask_shape"] == tuple(temporal_attn_mask.shape)

        # Spatial: still bidirectional, no mask.
        assert spatial["is_causal"] is False
        assert spatial["attn_mask_is_none"], "spatial pass remains bidirectional regardless of temporal mask"

    def test_temporal_default_mask_is_lower_triangular(self):
        """Direct check that the wrapper's no-temporal-mask path is causal:
        the temporal pass at the past-frame slot (t=0) must NOT depend on
        the current-frame slot (t=T-1).

        Modify only the current frame's input and verify the wrapper's
        output at the past-frame slot is unchanged. This pins the causal
        mask invariant from the outside, without monkey-patching SDPA.
        """
        torch.manual_seed(11)
        wrapper = self._grab_wrapper()
        t = 2
        n = wrapper.num_tokens_per_frame
        d = wrapper.embed_dim

        past = torch.randn(1, n, d)
        current_a = torch.randn(1, n, d)
        current_b = torch.randn(1, n, d)
        hidden_a = torch.cat([past, current_a], dim=0)  # (T=2, N, D)
        hidden_b = torch.cat([past, current_b], dim=0)

        with torch.no_grad():
            out_a = wrapper(hidden_a, num_frames=t)[0]
            out_b = wrapper(hidden_b, num_frames=t)[0]

        # Past frame at index 0: causal mask must prevent it from seeing
        # the current frame's data, so out_a[0] must equal out_b[0].
        torch.testing.assert_close(out_a[0], out_b[0], rtol=1e-5, atol=1e-5)
        # Sanity: the current-frame output WAS allowed to differ between
        # the two runs (otherwise the test is vacuous).
        assert not torch.allclose(out_a[1], out_b[1], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Subgoal images share the video encoder weights — the only set of SigLIP
# weights that exists. ``embed_image`` (subgoal path) and the encoder's
# T=1 short-circuit go through the same modules; outputs must match
# byte-for-byte.
# ---------------------------------------------------------------------------


class TestSubgoalSharesVideoEncoderWeights:
    def test_embed_image_via_suppress_matches_encoder_t1(self):
        """A single-frame forward through ``vision_tower`` under
        ``suppress_spacetime_temporal`` (the path ``Gemma3WithExpertModel.
        embed_image`` takes for subgoal images) must produce byte-identical
        output to ``SpaceTimeSiglipVideoEncoder.forward`` at T=1 on the same
        image, **before** the [0, 1] → [-1, 1] rescale (since ``embed_image``
        expects callers to have already rescaled, while the encoder rescales
        internally).
        """
        torch.manual_seed(2)
        vision_tower, projector, _ = _build_gemma3_siglip_and_projector()
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=1,
            spacetime_layer_stride=4,
        ).eval()

        # Encoder takes [0, 1] and rescales internally.
        image_unit = torch.rand(2, 3, 224, 224)

        # embed_image equivalent: the caller pre-rescales to [-1, 1] and
        # invokes the same vision_tower under suppress_spacetime_temporal.
        image_siglip = image_unit * 2.0 - 1.0
        with torch.no_grad():
            with suppress_spacetime_temporal(vision_tower):
                last_hidden = vision_tower(pixel_values=image_siglip).last_hidden_state
            out_image = projector(last_hidden)
            out_video = encoder(image_unit.unsqueeze(1))  # (B, 1, C, H, W)

        torch.testing.assert_close(out_video, out_image, rtol=1e-5, atol=1e-5)

    def test_encoder_owns_no_siglip_parameters(self):
        """``SpaceTimeSiglipVideoEncoder`` holds ``vision_tower`` and
        ``multi_modal_projector`` by reference in lists. Its OWN parameters
        and state_dict must therefore be empty — proving there is no second
        copy of the SigLIP weights anywhere in the encoder's tree.
        """
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)

        own_param_count = sum(1 for _ in encoder.parameters())
        assert own_param_count == 0, (
            f"encoder owns {own_param_count} params; expected 0 — vision_tower "
            "and multi_modal_projector should be held by reference, not registered"
        )

        # state_dict contains only registered buffers/params. _temporal_pe is
        # non-persistent (excluded from state_dict). Adopted submodule keys
        # are accessed through the wrapped vision_tower, NOT under the
        # encoder's path. So the encoder's own state_dict must be empty.
        own_state = encoder.state_dict()
        assert len(own_state) == 0, f"encoder.state_dict() = {list(own_state.keys())}; expected empty"

    def test_only_one_copy_of_siglip_q_proj(self):
        """Across the wrapped vision_tower's full state_dict, there is
        exactly ONE copy of each SigLIP attention weight. Wrapping with
        space-time attention must not duplicate any q/k/v/o projection.
        """
        encoder = _make_gemma3_encoder(max_num_frames=4, spacetime_stride=4)
        keys = list(encoder.vision_tower.state_dict().keys())
        q_proj_keys = [k for k in keys if k.endswith("self_attn.q_proj.weight")]

        # 27 SigLIP layers → exactly 27 q_proj.weight entries.
        assert len(q_proj_keys) == 27, (
            f"expected 27 q_proj entries (one per layer); got {len(q_proj_keys)}: {q_proj_keys}"
        )
