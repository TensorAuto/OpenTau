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

"""Native-resolution vision input across the PaliGemma-family policies.

Covers the three guarantees introduced with native-resolution support:

1. **Config validation** — a `resize_imgs_with_padding` that differs from the
   resolution of the bound image features fails fast on the strict (training)
   path and warns loudly on the eval path, with
   `skip_input_resolution_check` as the legacy-checkpoint escape hatch.
2. **No silent crop** — at a resolution that does not divide the SigLIP patch
   size (e.g. 180x320 with patch 14), frames are padded up to the next patch
   multiple and position embeddings are interpolated, so the patch grid
   covers every pixel (ceil(180/14) x ceil(320/14) = 13 x 23 = 299 tokens
   instead of the floor-cropped 12 x 22).
3. **No-op passthrough** — when the input already matches the target, the
   resize helpers return the input tensor unchanged (same object, not a
   same-size bilinear round trip), and the 224x224 vision path stays
   bit-identical to the historical fixed-position-embedding path.

All models here are tiny random-init transformers configs — nothing is
downloaded from the Hub.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import resize_with_pad
from opentau.policies.utils import assert_gemma3_input_resolution
from opentau.scripts.eval import validate_eval_input_resolution
from opentau.utils.transformers_patch import patched_paligemma_model_get_image_features
from opentau.utils.vision_utils import pad_to_patch_multiple, patch_grid_hw

# 180x320 is the DROID camera resolution — the motivating native-resolution
# case: 180 % 14 == 320 % 14 == 12, so the stride-14 conv would floor-crop 12
# pixel rows and columns without the pad-to-patch-multiple step.
NATIVE_HW = (180, 320)
NATIVE_GRID_TOKENS = 13 * 23  # ceil(180/14) * ceil(320/14)
PATCH = 14


def _build_tiny_siglip_and_projector():
    """Tiny random SigLIP (patch 14, image_size 224) + PaliGemma projector.

    Mirrors the pattern of ``_build_paligemma_siglip_and_projector`` in
    ``test_pi07_video_encoder_cpu.py`` but at unit-test scale.
    """
    vision_cfg_dict = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_image_tokens": 256,
        "patch_size": PATCH,
        "image_size": 224,
        "projection_dim": 32,
        "vision_use_head": False,
    }
    vision_tower = SiglipVisionModel(SiglipVisionConfig(**vision_cfg_dict)).eval()
    pali_cfg = PaliGemmaConfig(
        vision_config=vision_cfg_dict,
        text_config={"hidden_size": 32, "model_type": "gemma", "vocab_size": 64},
        projection_dim=32,
    )
    projector = PaliGemmaMultiModalProjector(pali_cfg).eval()
    return vision_tower, projector, 32


@pytest.fixture
def siglip_backbone():
    torch.manual_seed(0)
    return _build_tiny_siglip_and_projector()


class TestPatchGridHelpers:
    def test_patch_grid_hw_ceils(self):
        assert patch_grid_hw(*NATIVE_HW, PATCH) == (13, 23)
        assert patch_grid_hw(224, 224, PATCH) == (16, 16)
        assert patch_grid_hw(14, 14, PATCH) == (1, 1)
        assert patch_grid_hw(15, 14, PATCH) == (2, 1)

    def test_pad_to_patch_multiple_is_noop_at_multiples(self):
        img = torch.rand(2, 3, 224, 224)
        assert pad_to_patch_multiple(img, PATCH) is img

    def test_pad_to_patch_multiple_pads_bottom_right(self):
        img = torch.rand(2, 3, *NATIVE_HW)
        padded = pad_to_patch_multiple(img, PATCH, pad_value=-1.0)
        assert padded.shape == (2, 3, 182, 322)
        # Original content untouched, origin pixel-aligned.
        assert torch.equal(padded[..., :180, :320], img)
        # Padding is bottom/right only, at the fill value.
        assert (padded[..., 180:, :] == -1.0).all()
        assert (padded[..., :, 320:] == -1.0).all()

    def test_padded_grid_covers_all_pixels(self):
        grid_h, grid_w = patch_grid_hw(*NATIVE_HW, PATCH)
        assert grid_h * PATCH >= NATIVE_HW[0]
        assert grid_w * PATCH >= NATIVE_HW[1]
        # And is minimal: one patch fewer would crop pixels.
        assert (grid_h - 1) * PATCH < NATIVE_HW[0]
        assert (grid_w - 1) * PATCH < NATIVE_HW[1]


class TestResizeWithPadNoOp:
    def test_same_size_returns_input_object(self):
        img = torch.rand(2, 3, *NATIVE_HW)
        out = resize_with_pad(img, width=NATIVE_HW[1], height=NATIVE_HW[0], pad_value=0)
        assert out is img

    def test_non_square_target_orientation(self):
        # A half-scale native frame letterboxed up to the native target must
        # come out (H, W) = (180, 320) — not transposed. This was latent
        # while every target was square: the function signature is
        # (width, height) but the config tuple is (height, width).
        img = torch.rand(2, 3, 90, 160)
        out = resize_with_pad(img, width=NATIVE_HW[1], height=NATIVE_HW[0], pad_value=0)
        assert out.shape == (2, 3, *NATIVE_HW)


class TestPatchedGetImageFeaturesNativeRes:
    def test_native_resolution_produces_full_grid(self, siglip_backbone):
        vision_tower, projector, vlm_hidden = siglip_backbone
        stub = SimpleNamespace(vision_tower=vision_tower, multi_modal_projector=projector)
        pixels = torch.rand(2, 3, *NATIVE_HW) * 2.0 - 1.0
        with torch.no_grad():
            out = patched_paligemma_model_get_image_features(stub, pixels)
        assert out.shape == (2, NATIVE_GRID_TOKENS, vlm_hidden)

    def test_config_resolution_is_bit_identical_to_vanilla_path(self, siglip_backbone):
        vision_tower, projector, _ = siglip_backbone
        stub = SimpleNamespace(vision_tower=vision_tower, multi_modal_projector=projector)
        pixels = torch.rand(1, 3, 224, 224) * 2.0 - 1.0
        with torch.no_grad():
            out = patched_paligemma_model_get_image_features(stub, pixels)
            reference = projector(vision_tower(pixels).last_hidden_state)
        assert torch.equal(out, reference)


class TestSpaceTimeVideoEncoderNativeRes:
    def test_native_forward_shape_and_token_count(self, siglip_backbone):
        vision_tower, projector, vlm_hidden = siglip_backbone
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=3,
            spacetime_layer_stride=2,
            expected_image_size=NATIVE_HW,
        )
        assert encoder.num_video_tokens == NATIVE_GRID_TOKENS
        # 2-D grid exposed for consumers needing the layout (e.g. pi05_mem
        # MRoPE, which must distinguish 16x4 from 8x8 despite both being 64).
        assert encoder.grid_hw == (13, 23)
        with torch.no_grad():
            out = encoder(torch.rand(2, 2, 3, *NATIVE_HW))
        assert out.shape == (2, NATIVE_GRID_TOKENS, vlm_hidden)

    def test_wrong_input_resolution_raises(self, siglip_backbone):
        vision_tower, projector, _ = siglip_backbone
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=2,
            spacetime_layer_stride=2,
            expected_image_size=NATIVE_HW,
        )
        with pytest.raises(ValueError, match="constructed for"):
            encoder(torch.rand(1, 1, 3, 224, 224))

    def test_default_construction_keeps_legacy_grid(self, siglip_backbone):
        vision_tower, projector, vlm_hidden = siglip_backbone
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=2,
            spacetime_layer_stride=2,
        )
        assert encoder.expected_image_size == (224, 224)
        assert encoder.num_video_tokens == 256
        assert not encoder._interpolate_pos_encoding
        with torch.no_grad():
            out = encoder(torch.rand(1, 1, 3, 224, 224))
        assert out.shape == (1, 256, vlm_hidden)

    def test_single_frame_invariance_at_native_res(self, siglip_backbone):
        # T=1 through the video encoder must stay byte-identical to
        # embed_image (the patched get_image_features) on the same frames —
        # this is what keeps the pi07_paligemma run/skip zero-fill paths and
        # the subgoal image path aligned at native resolution.
        vision_tower, projector, _ = siglip_backbone
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            max_num_frames=2,
            spacetime_layer_stride=2,
            expected_image_size=NATIVE_HW,
        )
        stub = SimpleNamespace(vision_tower=vision_tower, multi_modal_projector=projector)
        frames = torch.rand(2, 3, *NATIVE_HW)
        with torch.no_grad():
            video_out = encoder(frames.unsqueeze(1))  # video path takes [0, 1]
            image_out = patched_paligemma_model_get_image_features(stub, frames * 2.0 - 1.0)
        assert torch.equal(video_out, image_out)


class TestValidateInputResolution:
    def _config_with_camera(self, resize, camera_hw=NATIVE_HW):
        config = PI07PaligemmaLowLevelConfig(resize_imgs_with_padding=resize)
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *camera_hw)),
        }
        return config

    def test_strict_mismatch_raises(self):
        config = self._config_with_camera(resize=(224, 224))
        with pytest.raises(ValueError, match="resize_imgs_with_padding"):
            config.validate_input_resolution(strict=True)

    def test_eval_mismatch_warns_but_loads(self, caplog):
        config = self._config_with_camera(resize=(224, 224))
        with caplog.at_level("WARNING"):
            config.validate_input_resolution(strict=False)
        assert any("resize_imgs_with_padding" in record.message for record in caplog.records)

    def test_escape_hatch_downgrades_strict_to_warning(self, caplog):
        config = self._config_with_camera(resize=(224, 224))
        config.skip_input_resolution_check = True
        with caplog.at_level("WARNING"):
            config.validate_input_resolution(strict=True)
        assert any("resize_imgs_with_padding" in record.message for record in caplog.records)

    def test_matching_resolution_passes(self):
        self._config_with_camera(resize=NATIVE_HW).validate_input_resolution(strict=True)

    def test_none_resize_passes(self):
        self._config_with_camera(resize=None).validate_input_resolution(strict=True)

    def test_none_resize_mixed_camera_resolutions_rejected(self, caplog):
        # Native pass-through has no resize step to harmonize cameras, so a
        # mixed-resolution camera set must be flagged rather than silently
        # seeding the vision tower with whichever camera iterates first.
        config = self._config_with_camera(resize=None)
        config.input_features["camera1"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        with pytest.raises(ValueError, match="mixed resolutions"):
            config.validate_input_resolution(strict=True)
        with caplog.at_level("WARNING"):
            config.validate_input_resolution(strict=False)
        assert any("mixed resolutions" in record.message for record in caplog.records)

    def test_empty_camera_placeholders_ignored(self):
        config = self._config_with_camera(resize=NATIVE_HW)
        # validate_features()-style placeholder with its historical hard-coded
        # shape must not trip the check — it describes no real data.
        config.input_features["observation.images.empty_camera_0"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        )
        config.validate_input_resolution(strict=True)

    def test_input_image_size_prefers_resize_target(self):
        config = self._config_with_camera(resize=(224, 224))
        assert config.input_image_size == (224, 224)

    def test_input_image_size_falls_back_to_bound_features(self):
        config = self._config_with_camera(resize=None)
        assert config.input_image_size == NATIVE_HW

    def test_input_image_size_none_when_unbound(self):
        config = PI07PaligemmaLowLevelConfig(resize_imgs_with_padding=None)
        config.input_features = {}
        assert config.input_image_size is None

    def test_gemma3_guard_rejects_native_resolution(self):
        # pi06/pi07 constructors call this: Gemma3MultiModalProjector
        # hard-codes a square patch grid, so native input must fail with the
        # real diagnosis instead of a projector reshape crash.
        with pytest.raises(ValueError, match="Gemma3-family"):
            assert_gemma3_input_resolution(NATIVE_HW, 448)

    def test_gemma3_guard_passes_tower_resolution_and_unbound(self):
        assert_gemma3_input_resolution((448, 448), 448)
        assert_gemma3_input_resolution(None, 448)

    def test_eval_guard_native_passthrough_mismatch_raises(self):
        cfg = SimpleNamespace(
            policy=self._config_with_camera(resize=None),
            resolution=(224, 224),
        )
        with pytest.raises(ValueError, match="native pass-through"):
            validate_eval_input_resolution(cfg)

    def test_eval_guard_matching_and_resize_set_pass(self, caplog):
        # Matching native resolution passes.
        validate_eval_input_resolution(
            SimpleNamespace(policy=self._config_with_camera(resize=None), resolution=NATIVE_HW)
        )
        # With a resize target set, the in-policy letterbox reproduces the
        # checkpoint's training-time geometry; make_policy owns the warning.
        validate_eval_input_resolution(
            SimpleNamespace(policy=self._config_with_camera(resize=(224, 224)), resolution=(448, 448))
        )
        # Escape hatch downgrades the native-passthrough raise to a warning.
        skip_policy = self._config_with_camera(resize=None)
        skip_policy.skip_input_resolution_check = True
        with caplog.at_level("WARNING"):
            validate_eval_input_resolution(SimpleNamespace(policy=skip_policy, resolution=(224, 224)))
        assert any("different geometry" in record.message for record in caplog.records)

    def test_channels_last_shapes_recognized(self):
        # Bare LeRobotDatasetMetadata binds raw (H, W, C) shapes; the strict
        # check must not silently no-op on them.
        config = PI07PaligemmaLowLevelConfig(resize_imgs_with_padding=(224, 224))
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(*NATIVE_HW, 3)),
        }
        with pytest.raises(ValueError, match="resize_imgs_with_padding"):
            config.validate_input_resolution(strict=True)
        config.resize_imgs_with_padding = NATIVE_HW
        config.validate_input_resolution(strict=True)
