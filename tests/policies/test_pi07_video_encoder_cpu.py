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

"""CPU-only tests for ``SpaceTimeSiglipVideoEncoder`` (pi07 low-level).

Mirrors PR #171's pi05_mem video-encoder tests, ported to pi07's namespace
and using the Gemma 3 SigLIP config.  Builds randomly-initialised SigLIP +
``Gemma3MultiModalProjector`` pairs from scratch so no HF download is needed.

The tests pin three guarantees the planner relies on:

  * Wrapping every ``stride``-th SigLIP layer with space-time attention
    leaves the wrapped vision_tower's state_dict keys identical to the
    vanilla SigLIP — a vanilla pi07 / pi07_paligemma checkpoint loads in
    untouched.
  * At ``T=1`` the wrapper short-circuits to the vanilla spatial block, so
    a stride-999 (no-wrapping) encoder and a stride-4 (6 layers wrapped)
    encoder built on the same weights produce byte-identical outputs.
  * The new ``suppress_spacetime_temporal`` context manager (replaces the
    silent ``bt % t != 0`` numerology in earlier drafts) suppresses temporal
    attention for non-video forwards while leaving the spatial block
    untouched.
"""

from __future__ import annotations

import copy

import pytest
import torch

from opentau.policies.pi07.low_level_planner.video_encoder import (
    SpaceTimeEncoderLayerWrapper,
    SpaceTimeSiglipVideoEncoder,
    _build_temporal_sinusoidal_pe,
    suppress_spacetime_temporal,
)

# Helpers


def _build_siglip_and_projector():
    """Construct a fresh, randomly-initialised SigLIP + Gemma 3 projector."""
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
    return vision_tower, projector


def _make_encoder(num_frames: int = 2, spacetime_stride: int = 4):
    vision_tower, projector = _build_siglip_and_projector()
    return SpaceTimeSiglipVideoEncoder(
        vision_tower=vision_tower,
        multi_modal_projector=projector,
        num_frames=num_frames,
        spacetime_layer_stride=spacetime_stride,
    )


# Temporal sinusoidal PE


class TestTemporalSinusoidalPE:
    def test_current_frame_row_is_zero(self):
        pe = _build_temporal_sinusoidal_pe(num_frames=8, embed_dim=64)
        assert pe.shape == (8, 64)
        # Current frame lives at t=T-1; row must be all zeros so a T=1 pass
        # collapses to the unmodified SigLIP forward.
        assert torch.all(pe[-1] == 0)

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


# Wrapper structure / state_dict invariance


class TestSpaceTimeWrapperStructure:
    def test_wrapped_layer_count(self):
        """Verify every ``stride``-th layer (1-indexed from the Nth) is wrapped."""
        encoder = _make_encoder(num_frames=2, spacetime_stride=4)
        layers = encoder.vision_tower.vision_model.encoder.layers
        wrapped_indices = [
            i for i, layer in enumerate(layers) if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        ]
        # 27 SigLIP layers; every 4th starting at index 3 → [3, 7, 11, 15, 19, 23]
        assert wrapped_indices == [3, 7, 11, 15, 19, 23]

    def test_state_dict_keys_match_vanilla_siglip(self):
        """The wrapped vision_tower's state_dict must have identical keys to
        an unwrapped SigLIP — guarantees that a pi07_paligemma checkpoint
        (or any vanilla SigLIP) loads without key remapping."""
        from transformers import SiglipVisionModel

        vision_tower, projector = _build_siglip_and_projector()
        reference_keys = set(SiglipVisionModel(vision_tower.config).state_dict().keys())

        SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=4,
            spacetime_layer_stride=4,
        )
        wrapped_keys = set(vision_tower.state_dict().keys())
        assert wrapped_keys == reference_keys, (
            f"state_dict keys diverged; "
            f"extra in wrapped: {wrapped_keys - reference_keys}, "
            f"missing from wrapped: {reference_keys - wrapped_keys}"
        )

    def test_temporal_pe_on_base_layer_device(self):
        """The temporal PE buffer must be constructed on the base layer's
        device/dtype, not default CPU.

        Regression for the pi05_mem GPU bug where the parent vision_tower had
        already been moved to a non-default dtype/device BEFORE wrapping —
        leaving the wrapper's PE on CPU/float32.
        """
        vision_tower, projector = _build_siglip_and_projector()
        vision_tower = vision_tower.to(dtype=torch.bfloat16)
        projector = copy.deepcopy(projector).to(dtype=torch.bfloat16)

        SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=4,
            spacetime_layer_stride=4,
        )

        ref_param = next(vision_tower.parameters())
        for layer in vision_tower.vision_model.encoder.layers:
            if isinstance(layer, SpaceTimeEncoderLayerWrapper):
                assert layer._temporal_pe.device == ref_param.device
                assert layer._temporal_pe.dtype == ref_param.dtype


# Forward shape / dtype / arg validation


class TestSpaceTimeForward:
    def test_forward_shape(self):
        encoder = _make_encoder(num_frames=2)
        encoder.eval()
        with torch.no_grad():
            video = torch.rand(1, 2, 3, 224, 224)
            out = encoder(video)
        # 16 patches/side at 224/14 → 256 tokens; Gemma 3 projector outputs
        # 256 mm tokens; per-token width is the Gemma 3 text hidden size.
        assert out.shape == (1, 256, 2560)
        assert out.dtype == torch.float32

    def test_wrong_num_frames_raises(self):
        encoder = _make_encoder(num_frames=4)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="frames"):
            encoder(torch.rand(1, 2, 3, 224, 224))

    def test_wrong_ndim_raises(self):
        encoder = _make_encoder(num_frames=2)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="5D"):
            encoder(torch.rand(2, 3, 224, 224))

    def test_forward_after_external_dtype_cast(self):
        """End-to-end forward must succeed when vision_tower was moved to a
        different dtype BEFORE wrapping (mirrors the GPU load flow of
        ``gemma3.to(cuda/bf16)`` then construct encoder).
        """
        vision_tower, projector = _build_siglip_and_projector()
        vision_tower = vision_tower.to(dtype=torch.bfloat16)
        projector = copy.deepcopy(projector).to(dtype=torch.bfloat16)

        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=4,
            spacetime_layer_stride=4,
        ).eval()
        with torch.no_grad():
            out = encoder(torch.rand(1, 4, 3, 224, 224, dtype=torch.bfloat16))
        assert out.shape == (1, 256, 2560)
        assert out.dtype == torch.bfloat16


# Single-frame invariance


class TestSingleFrameInvariance:
    def test_single_frame_invariance_structural(self):
        """At T=1 the wrapper must short-circuit: byte-identical output to a
        stride-999 (no-wrapping) encoder sharing the exact same underlying
        weights.
        """
        torch.manual_seed(0)
        vision_tower, projector = _build_siglip_and_projector()
        vt_no_st = copy.deepcopy(vision_tower)
        proj_no_st = copy.deepcopy(projector)

        enc_no_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_no_st,
            multi_modal_projector=proj_no_st,
            num_frames=1,
            spacetime_layer_stride=999,  # > 27 → no layers wrapped
        ).eval()
        enc_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=1,
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


# `suppress_spacetime_temporal` context manager (replaces the silent
# `bt % t != 0` numerology bypass).


class TestSuppressSpacetimeTemporal:
    def test_flag_toggles_and_restores(self):
        """Entering the context manager flips every wrapper to inactive;
        exiting restores the prior value (idempotent — works nested)."""
        encoder = _make_encoder(num_frames=4, spacetime_stride=4)
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
        explicitly rejects ``bt % t != 0`` — replacing the previous silent
        short-circuit which could have wrongly fired temporal attention over
        spatial-only inputs that happened to be divisible by ``num_frames``.
        """
        encoder = _make_encoder(num_frames=4, spacetime_stride=4)
        wrapper = next(
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        )
        # 256 = (224/14)**2 patches per frame (the expected per-frame token count).
        bad_input = torch.rand(3, 256, wrapper.embed_dim)
        with pytest.raises(ValueError, match="divisible by num_frames"):
            wrapper.forward(bad_input)

    def test_non_divisible_batch_succeeds_when_suppressed(self):
        """Inside ``suppress_spacetime_temporal`` the wrapper accepts any
        batch size and dispatches to the vanilla spatial block."""
        encoder = _make_encoder(num_frames=4, spacetime_stride=4)
        wrapper = next(
            layer
            for layer in encoder.vision_tower.vision_model.encoder.layers
            if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        )
        bad_input = torch.rand(3, 256, wrapper.embed_dim)
        with suppress_spacetime_temporal(encoder.vision_tower):
            out = wrapper.forward(bad_input)[0]
        assert out.shape == bad_input.shape
