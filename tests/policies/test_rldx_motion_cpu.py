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

"""CPU tests for the RLDX-1 STSS motion module and ``RLDXVideoEncoder``.

Covers three layers of the feature:

  * :class:`opentau.policies.pi05_mem.motion_module.MotionModule` in isolation
    (no SigLIP needed): output shape, zero-init no-op warm start, the L=1 vs
    L>1 temporal-window paths, norm/corr_func/layerscale variants, gradient
    flow, and the ragged (non-uniform grid) batch path.
  * :class:`opentau.policies.pi05_mem.rldx_video_encoder.RLDXVideoEncoder` on a tiny
    randomly-initialized SigLIP: it has NO space-time attention layers,
    zero-init is a no-op vs. a plain SigLIP forward, motion fires once un-gated,
    T=1 short-circuits, and ``obs_history_is_pad`` neutralizes padded frames so
    they cannot leak spurious motion into the current-frame residual.
  * ``PI05MemConfig`` motion-field validation.
"""

from __future__ import annotations

import pytest
import torch

from opentau.policies.pi05_mem.motion_module import MotionModule
from opentau.policies.pi05_mem.rldx_video_encoder import RLDXVideoEncoder
from opentau.policies.pi07.video_encoder import SpaceTimeEncoderLayerWrapper

# ---------------------------------------------------------------------------
# Tiny SigLIP fixture: a 6-layer / hidden-64 random tower with a 224/14 -> 16x16
# patch grid (so the default (5, 9, 9) motion window fits). Built once per
# module; RLDXVideoEncoder never mutates the tower, so it is safe to share.
# ---------------------------------------------------------------------------

_SIGLIP_HIDDEN = 64
_VLM_HIDDEN = 32
_N_LAYERS = 6
_GRID = 16
_N_PATCHES = _GRID * _GRID


def _build_tiny_siglip_and_projector():
    from transformers import PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

    vision_cfg = {
        "hidden_size": _SIGLIP_HIDDEN,
        "intermediate_size": 128,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 4,
        "num_hidden_layers": _N_LAYERS,
        "num_image_tokens": _N_PATCHES,
        "patch_size": 14,
        "image_size": 224,
        "projection_dim": _VLM_HIDDEN,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    }
    vision_tower = SiglipVisionModel(SiglipVisionConfig(**vision_cfg))
    # PaliGemmaMultiModalProjector maps vision hidden -> the TOP-LEVEL
    # PaliGemmaConfig.projection_dim, so set it (and the text hidden) explicitly.
    projector = PaliGemmaMultiModalProjector(
        PaliGemmaConfig(
            vision_config=vision_cfg,
            text_config={"hidden_size": _VLM_HIDDEN, "model_type": "gemma", "vocab_size": 257152},
            projection_dim=_VLM_HIDDEN,
        )
    )
    return vision_tower, projector


@pytest.fixture(scope="module")
def siglip_backbone():
    torch.manual_seed(0)
    return _build_tiny_siglip_and_projector()


def _make_encoder(backbone, *, max_num_frames=4, ungate=False, seed=0, **motion_kwargs):
    """Fresh RLDXVideoEncoder on the shared tower. ``ungate`` un-zeros the motion
    residual so its contribution is observable (zero-init makes it a no-op)."""
    torch.manual_seed(seed)
    vt, proj = backbone
    motion_kwargs.setdefault("motion_norm", "groupnorm")
    enc = RLDXVideoEncoder(vt, proj, max_num_frames=max_num_frames, **motion_kwargs).eval()
    if ungate:
        with torch.no_grad():
            enc.motion_module.out_proj.weight.normal_(0.0, 0.02)
    return enc


# ===========================================================================
# MotionModule (no SigLIP)
# ===========================================================================


class TestMotionModule:
    B, T, GH = 2, 4, 10  # 10x10 patch grid fits the default (5, 9, 9) window

    def _inputs(self, *, d_in, requires_grad=False, seed=0):
        torch.manual_seed(seed)
        n = self.GH * self.GH
        x = torch.randn(self.B * self.T * n, d_in, requires_grad=requires_grad)
        grid = torch.tensor([[self.T, self.GH, self.GH]] * self.B, dtype=torch.long)
        return x, grid

    def test_output_shape_matches_input_tokens(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, norm="groupnorm").eval()
        out = m(x, grid)
        assert out.shape == x.shape

    def test_zero_init_residual_is_noop(self):
        """Default zero_init_residual=True => output is exactly zero at init, so
        adding it as a residual leaves a checkpoint unchanged at step 0."""
        x, grid = self._inputs(d_in=48)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, norm="groupnorm").eval()
        out = m(x, grid)
        assert torch.count_nonzero(out) == 0

    def test_no_zero_init_contributes_immediately(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, norm="groupnorm", zero_init_residual=False).eval()
        out = m(x, grid)
        assert torch.count_nonzero(out) > 0
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("length", [1, 3, 5])
    def test_temporal_window_lengths(self, length):
        """L=1 (prev-frame correlation) and L>1 (unfolded window) paths both
        produce correctly-shaped, finite output with flowing gradients.

        Regression for the L=1 bug where x_src kept the un-rearranged 2D input
        and crashed _correlation's einsum.
        """
        x, grid = self._inputs(d_in=48, requires_grad=True)
        m = MotionModule(
            d_in=48, d_hid=32, d_out=48, window=(length, 9, 9), norm="groupnorm", zero_init_residual=False
        ).train()
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert m.out_proj.weight.grad is not None
        assert m.out_proj.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("norm", ["batchnorm", "groupnorm", "syncbn"])
    def test_norm_variants(self, norm):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, norm=norm).train()
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_unknown_norm_raises(self):
        with pytest.raises(ValueError, match="Unknown norm"):
            MotionModule(d_in=48, d_hid=32, d_out=48, norm="badnorm")

    @pytest.mark.parametrize("window", [(4, 9, 9), (2, 9, 9), (5, 8, 8), (5, 9, 9, 9)])
    def test_even_or_malformed_window_raises(self, window):
        """Even temporal or spatial dims silently change the STSS entry count and
        crash at forward; reject them at construction instead."""
        with pytest.raises(ValueError, match="odd|positive odd"):
            MotionModule(d_in=48, d_hid=32, d_out=48, window=window)

    def test_non_square_window_raises_in_module(self):
        with pytest.raises(ValueError, match="square"):
            MotionModule(d_in=48, d_hid=32, d_out=48, window=(5, 9, 7))

    @pytest.mark.parametrize("corr_func", ["cosine", "dotproduct", "dotproduct_softmax"])
    def test_corr_func_variants(self, corr_func):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(
            d_in=48, d_hid=32, d_out=48, corr_func=corr_func, norm="groupnorm", zero_init_residual=False
        ).eval()
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_multiple_encoders_summed(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(
            d_in=48, d_hid=32, d_out=48, n_encoders=3, norm="groupnorm", zero_init_residual=False
        ).eval()
        assert len(m.stss_encoders) == 3
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_full_integration_mode(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(
            d_in=48, d_hid=32, d_out=48, int_mode="full", norm="groupnorm", zero_init_residual=False
        ).eval()
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_layerscale_zero_init_is_noop(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, use_layerscale=True, norm="groupnorm").eval()
        assert torch.count_nonzero(m(x, grid)) == 0

    def test_layerscale_nonzero_contributes(self):
        x, grid = self._inputs(d_in=48)
        m = MotionModule(
            d_in=48, d_hid=32, d_out=48, use_layerscale=True, norm="groupnorm", zero_init_residual=False
        ).eval()
        assert torch.count_nonzero(m(x, grid)) > 0

    def test_nonuniform_grid_batch(self):
        """Videos with different T per batch element hit the per-video split path
        (all_same_grid is False)."""
        torch.manual_seed(0)
        gh = self.GH
        n = gh * gh
        t1, t2 = 2, 3
        x = torch.randn((t1 + t2) * n, 48)
        grid = torch.tensor([[t1, gh, gh], [t2, gh, gh]], dtype=torch.long)
        m = MotionModule(d_in=48, d_hid=32, d_out=48, norm="groupnorm", zero_init_residual=False).eval()
        out = m(x, grid)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ===========================================================================
# RLDXVideoEncoder
# ===========================================================================


class TestRLDXVideoEncoder:
    B, T = 2, 4

    def _video(self, t=None, seed=0):
        torch.manual_seed(seed)
        return torch.rand(self.B, t or self.T, 3, 224, 224)

    def test_has_no_spacetime_attention_layers(self, siglip_backbone):
        """The defining property: RLDX uses plain SigLIP + motion, never the
        space-time separable-attention wrappers."""
        enc = _make_encoder(siglip_backbone)
        n_wrappers = sum(isinstance(m, SpaceTimeEncoderLayerWrapper) for m in enc.modules())
        assert n_wrappers == 0

    def test_output_shape(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone)
        out = enc(self._video())
        assert out.shape == (self.B, _N_PATCHES, _VLM_HIDDEN)

    def test_zero_init_is_noop_vs_plain_siglip(self, siglip_backbone):
        """With the default zero-init gate, the encoder output equals a plain
        SigLIP forward (motion injects nothing at step 0)."""
        enc = _make_encoder(siglip_backbone)
        video = self._video()
        out_motion = enc(video)
        # Disable motion by moving the insert layer out of range -> bare SigLIP.
        saved = enc.motion_insert_layer
        enc.motion_insert_layer = 10_000
        out_plain = enc(video)
        enc.motion_insert_layer = saved
        assert torch.allclose(out_motion, out_plain, atol=1e-6)

    def test_motion_changes_output_when_ungated(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, ungate=True)
        video = self._video()
        out_motion = enc(video)
        saved = enc.motion_insert_layer
        enc.motion_insert_layer = 10_000
        out_plain = enc(video)
        enc.motion_insert_layer = saved
        assert (out_motion - out_plain).abs().max() > 1e-3

    def test_single_frame_skips_motion(self, siglip_backbone):
        """T=1 has no time axis: a plain single-frame SigLIP forward, no crash."""
        enc = _make_encoder(siglip_backbone, ungate=True)
        out = enc(self._video(t=1))
        assert out.shape == (self.B, _N_PATCHES, _VLM_HIDDEN)

    def test_pad_mask_neutralizes_padded_frames(self, siglip_backbone):
        """The current-frame output must be invariant to whatever junk sits in
        slots flagged padded by obs_history_is_pad."""
        enc = _make_encoder(siglip_backbone, ungate=True)
        torch.manual_seed(1)
        real = torch.rand(self.B, 2, 3, 224, 224)
        vid_a = torch.cat([torch.zeros(self.B, 2, 3, 224, 224), real], dim=1)
        vid_b = torch.cat([torch.rand(self.B, 2, 3, 224, 224), real], dim=1)
        pad = torch.zeros(self.B, self.T, dtype=torch.bool)
        pad[:, :2] = True
        out_a = enc(vid_a, obs_history_is_pad=pad)
        out_b = enc(vid_b, obs_history_is_pad=pad)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_no_mask_lets_padded_content_leak(self, siglip_backbone):
        """Without the mask, differing padded-slot content does change the
        output (confirms the mask is what neutralizes it)."""
        enc = _make_encoder(siglip_backbone, ungate=True)
        torch.manual_seed(1)
        real = torch.rand(self.B, 2, 3, 224, 224)
        vid_a = torch.cat([torch.zeros(self.B, 2, 3, 224, 224), real], dim=1)
        vid_b = torch.cat([torch.rand(self.B, 2, 3, 224, 224), real], dim=1)
        assert (enc(vid_a) - enc(vid_b)).abs().max() > 1e-3

    def test_all_but_current_padded_is_finite(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, ungate=True)
        pad = torch.ones(self.B, self.T, dtype=torch.bool)
        pad[:, -1] = False
        out = enc(self._video(), obs_history_is_pad=pad)
        assert torch.isfinite(out).all()

    def test_default_insert_layer_is_midstack(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone)
        assert enc.motion_insert_layer == _N_LAYERS // 3

    def test_custom_insert_layer(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, motion_insert_layer=4)
        assert enc.motion_insert_layer == 4

    def test_motion_params_registered_and_trainable(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone)
        keys = dict(enc.named_parameters())
        assert any(k.startswith("motion_module.") for k in keys)
        # Vision tower held by-reference is NOT registered under the encoder.
        assert not any("vision_tower" in k for k in keys)
        assert all(p.requires_grad for k, p in keys.items() if k.startswith("motion_module."))

    def test_gradients_flow_into_motion(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, ungate=True)
        enc.train()
        enc(self._video()).sum().backward()
        g = enc.motion_module.out_proj.weight.grad
        assert g is not None and g.abs().sum() > 0

    def test_gradient_checkpointing_forward(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, ungate=True, gradient_checkpointing=True)
        enc.train()
        out = enc(self._video())
        assert out.shape == (self.B, _N_PATCHES, _VLM_HIDDEN)
        out.sum().backward()
        assert enc.motion_module.out_proj.weight.grad is not None

    def test_window_spatial_larger_than_grid_raises(self, siglip_backbone):
        with pytest.raises(ValueError, match="exceeds patch grid"):
            _make_encoder(siglip_backbone, motion_window=(5, 21, 21))

    def test_insert_layer_out_of_range_raises(self, siglip_backbone):
        with pytest.raises(ValueError, match="out of range"):
            _make_encoder(siglip_backbone, motion_insert_layer=_N_LAYERS)

    def test_l1_window_end_to_end(self, siglip_backbone):
        enc = _make_encoder(siglip_backbone, ungate=True, motion_window=(1, 9, 9))
        out = enc(self._video())
        assert out.shape == (self.B, _N_PATCHES, _VLM_HIDDEN)
        assert torch.isfinite(out).all()


# ===========================================================================
# PI05MemConfig motion-field validation
# ===========================================================================


class TestRLDXMotionConfig:
    def _cfg(self, **kw):
        from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig

        return PI05MemConfig(**kw)

    def test_motion_off_by_default(self):
        c = self._cfg()
        assert c.use_motion is False

    def test_motion_field_defaults(self):
        c = self._cfg()
        assert c.motion_insert_layer is None
        assert c.motion_window == (5, 9, 9)
        assert c.motion_norm == "groupnorm"
        assert c.motion_int_mode == "lite"
        assert c.motion_zero_init is True

    def test_valid_motion_config(self):
        c = self._cfg(use_motion=True, motion_norm="groupnorm", n_obs_steps=8)
        assert c.use_motion is True

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="motion_norm"):
            self._cfg(use_motion=True, motion_norm="layernorm")

    def test_invalid_int_mode_raises(self):
        with pytest.raises(ValueError, match="motion_int_mode"):
            self._cfg(use_motion=True, motion_int_mode="heavy")

    def test_invalid_corr_func_raises(self):
        with pytest.raises(ValueError, match="motion_corr_func"):
            self._cfg(use_motion=True, motion_corr_func="l2")

    def test_non_square_window_raises(self):
        with pytest.raises(ValueError, match="square"):
            self._cfg(use_motion=True, motion_window=(5, 9, 7))

    @pytest.mark.parametrize("window", [(4, 9, 9), (5, 8, 8), (2, 9, 9)])
    def test_even_window_raises(self, window):
        with pytest.raises(ValueError, match="odd"):
            self._cfg(use_motion=True, motion_window=window)

    def test_bad_window_length_raises(self):
        with pytest.raises(ValueError, match="motion_window"):
            self._cfg(use_motion=True, motion_window=(5, 9))

    def test_single_obs_step_with_motion_raises(self):
        with pytest.raises(ValueError, match="n_obs_steps"):
            self._cfg(use_motion=True, n_obs_steps=1)

    def test_validation_skipped_when_motion_off(self):
        # Bad motion settings are ignored entirely when use_motion is False.
        c = self._cfg(use_motion=False, motion_norm="nonsense", n_obs_steps=1)
        assert c.use_motion is False
