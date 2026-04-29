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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi05_mem.modeling_pi05 import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_discrete_tokens,
    resize_with_pad,
)


class TestPI05MemConfig:
    """Unit tests for PI05MemConfig validation and properties."""

    def test_default_config(self):
        config = PI05MemConfig()
        assert config.n_obs_steps == 8
        assert config.chunk_size == 50
        assert config.n_action_steps == 50
        assert config.history_interval == 1  # auto-set when n_obs_steps > 1
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.spacetime_layer_stride == 4

    def test_invalid_spacetime_layer_stride(self):
        with pytest.raises(ValueError, match="spacetime_layer_stride"):
            PI05MemConfig(spacetime_layer_stride=0)

    def test_obs_buffer_size_single_frame(self):
        config = PI05MemConfig(n_obs_steps=1)
        assert config.obs_buffer_size == 1

    def test_obs_buffer_size_with_history(self):
        config = PI05MemConfig(n_obs_steps=4, history_interval=2)
        assert config.obs_buffer_size == (4 - 1) * 2 + 1  # 7

    def test_obs_buffer_size_interval_defaults_to_1(self):
        config = PI05MemConfig(n_obs_steps=8)
        assert config.history_interval == 1
        assert config.obs_buffer_size == 8

    def test_invalid_n_obs_steps(self):
        with pytest.raises(ValueError, match="n_obs_steps"):
            PI05MemConfig(n_obs_steps=0)

    def test_invalid_history_interval(self):
        with pytest.raises(ValueError, match="history_interval"):
            PI05MemConfig(n_obs_steps=4, history_interval=-1)

    def test_n_action_steps_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="chunk size"):
            PI05MemConfig(chunk_size=10, n_action_steps=20)

    def test_max_delay_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="max delay"):
            PI05MemConfig(chunk_size=10, n_action_steps=10, max_delay=20)

    def test_validate_features_adds_empty_cameras(self):
        config = PI05MemConfig(empty_cameras=2)
        config.input_features = {}
        config.validate_features()
        assert "observation.images.empty_camera_0" in config.input_features
        assert "observation.images.empty_camera_1" in config.input_features

    def test_action_delta_indices(self):
        config = PI05MemConfig(chunk_size=10, n_action_steps=10)
        assert config.action_delta_indices == list(range(10))


class TestHelperFunctions:
    """Unit tests for standalone helper functions in modeling_pi05."""

    def test_sinusoidal_embedding_shape(self):
        time = torch.rand(2, 10)
        emb = create_sinusoidal_pos_embedding(time, dimension=64, min_period=4e-3, max_period=4.0)
        assert emb.shape == (2, 10, 64)

    def test_sinusoidal_embedding_odd_dimension_raises(self):
        time = torch.rand(2, 10)
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(time, dimension=63, min_period=4e-3, max_period=4.0)

    def test_sinusoidal_embedding_wrong_ndim_raises(self):
        time = torch.rand(10)
        with pytest.raises(ValueError, match="batch_size"):
            create_sinusoidal_pos_embedding(time, dimension=64, min_period=4e-3, max_period=4.0)

    def test_resize_with_pad_maintains_shape(self):
        img = torch.randn(2, 3, 480, 640)
        result = resize_with_pad(img, width=224, height=224)
        assert result.shape == (2, 3, 224, 224)

    def test_resize_with_pad_wrong_ndim(self):
        img = torch.randn(3, 480, 640)
        with pytest.raises(ValueError, match="expected"):
            resize_with_pad(img, width=224, height=224)

    def test_pad_discrete_tokens_padding(self):
        tokens = [[1, 2, 3], [4, 5]]
        result_tokens, result_masks = pad_discrete_tokens(tokens, max_length=5)
        assert result_tokens.shape == (2, 5)
        assert result_masks.shape == (2, 5)
        np.testing.assert_array_equal(result_tokens[0], [1, 2, 3, 0, 0])
        np.testing.assert_array_equal(result_masks[0], [True, True, True, False, False])
        np.testing.assert_array_equal(result_tokens[1], [4, 5, 0, 0, 0])
        np.testing.assert_array_equal(result_masks[1], [True, True, False, False, False])

    def test_pad_discrete_tokens_truncation(self):
        tokens = [[1, 2, 3, 4, 5]]
        result_tokens, result_masks = pad_discrete_tokens(tokens, max_length=3)
        assert result_tokens.shape == (1, 3)
        np.testing.assert_array_equal(result_tokens[0], [1, 2, 3])
        np.testing.assert_array_equal(result_masks[0], [True, True, True])

    def test_make_att_2d_masks_basic(self):
        pad_masks = torch.ones(1, 4, dtype=torch.bool)
        att_masks = torch.tensor([[1, 0, 0, 1]], dtype=torch.int32)
        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (1, 4, 4)
        assert result.dtype == torch.bool
        assert result[0, 0, 0].item() is True
        assert result[0, 0, 3].item() is False

    def test_make_att_2d_masks_with_padding(self):
        pad_masks = torch.tensor([[True, True, False]], dtype=torch.bool)
        att_masks = torch.tensor([[1, 0, 0]], dtype=torch.int32)
        result = make_att_2d_masks(pad_masks, att_masks)
        assert result[0, 2, 0].item() is False
        assert result[0, 0, 2].item() is False


class TestBuildHistoryBatch:
    """Test _build_history_batch logic using a lightweight mock policy."""

    @pytest.fixture
    def mock_policy(self):
        """Create a minimal mock that simulates PI05MemPolicy for history testing."""
        from opentau.configs.types import FeatureType, PolicyFeature

        policy = MagicMock()
        policy.config = PI05MemConfig(n_obs_steps=4, history_interval=1)
        policy.config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        policy._state_buffer = None
        policy._obs_buffers = {}

        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy._build_history_batch = PI05MemPolicy._build_history_batch.__get__(policy)
        return policy

    def test_first_call_zero_pads(self, mock_policy):
        batch = {
            "state": torch.ones(1, 8),
            "camera0": torch.ones(1, 3, 4, 4),
            "prompt": ["test"],
        }
        result = mock_policy._build_history_batch(batch)

        assert result["state"].shape == (1, 4, 8)
        assert result["camera0"].shape == (1, 4, 3, 4, 4)
        assert result["prompt"] == ["test"]

        # First 3 frames should be zero-padded, last should be the actual observation
        assert torch.all(result["state"][:, :3, :] == 0)
        assert torch.all(result["state"][:, 3, :] == 1)

    def test_history_fills_up(self, mock_policy):
        batch_template = {
            "state": torch.zeros(1, 8),
            "camera0": torch.zeros(1, 3, 4, 4),
            "prompt": ["test"],
        }

        for i in range(4):
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_template.items()}
            batch["state"] = torch.full((1, 8), float(i + 1))
            batch["camera0"] = torch.full((1, 3, 4, 4), float(i + 1))
            result = mock_policy._build_history_batch(batch)

        assert result["state"].shape == (1, 4, 8)
        for t in range(4):
            assert result["state"][0, t, 0].item() == float(t + 1)

    def test_metadata_keys_passed_through(self, mock_policy):
        batch = {
            "state": torch.ones(1, 8),
            "camera0": torch.ones(1, 3, 4, 4),
            "prompt": ["pick up the block"],
            "extra_metadata": "should_pass_through",
        }
        result = mock_policy._build_history_batch(batch)
        assert result["prompt"] == ["pick up the block"]
        assert result["extra_metadata"] == "should_pass_through"


class TestPrepareState:
    """Test prepare_state in isolation."""

    def test_2d_state_unsqueezes(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8, n_obs_steps=1)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 6)}
        result = policy.prepare_state(batch)
        assert result.shape == (2, 1, 8)

    def test_3d_state_pads(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=16)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 4, 10)}
        result = policy.prepare_state(batch)
        assert result.shape == (2, 4, 16)

    def test_state_exceeding_max_dim_raises(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 1, 16)}
        with pytest.raises(ValueError, match="exceeds max_state_dim"):
            policy.prepare_state(batch)

    def test_2d_state_with_history_raises(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8, n_obs_steps=4)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 6)}
        with pytest.raises(ValueError, match="Expected 3D"):
            policy.prepare_state(batch)


class TestPrepareVideos:
    """Test prepare_videos preprocessing logic."""

    @pytest.fixture
    def mock_policy(self):
        from opentau.configs.types import FeatureType, PolicyFeature
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        # Default fixture uses single-frame (n_obs_steps=1); individual tests
        # that exercise history behavior bump n_obs_steps themselves.
        policy.config = PI05MemConfig(
            n_obs_steps=1,
            resize_imgs_with_padding=(224, 224),
            empty_cameras=0,
        )
        policy.config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        policy.prepare_videos = PI05MemPolicy.prepare_videos.__get__(policy)
        return policy

    def test_4d_input_unsqueezes(self, mock_policy):
        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert len(videos) == 1
        assert videos[0].shape == (2, 1, 3, 224, 224)
        assert masks[0].shape == (2,)
        assert torch.all(masks[0])

    def test_5d_input_passes_through(self, mock_policy):
        batch = {"camera0": torch.randn(2, 4, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert videos[0].shape == (2, 4, 3, 224, 224)

    def test_all_images_missing_raises(self, mock_policy):
        batch = {"state": torch.randn(2, 8)}
        with pytest.raises(ValueError, match="All image features are missing"):
            mock_policy.prepare_videos(batch)

    def test_obs_history_is_pad_masks_frames(self, mock_policy):
        vid = torch.ones(1, 4, 3, 224, 224)
        obs_pad = torch.tensor([[True, False, False, False]])
        batch = {"camera0": vid}
        videos, masks = mock_policy.prepare_videos(batch, obs_history_is_pad=obs_pad)
        assert torch.all(videos[0][:, 0, :, :, :] == 0)
        assert torch.all(videos[0][:, 1, :, :, :] == 1)

    def test_empty_cameras_appended(self, mock_policy):
        from opentau.configs.types import FeatureType, PolicyFeature

        mock_policy.config.empty_cameras = 1
        mock_policy.config.input_features["empty_cam"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        )

        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert len(videos) == 2
        assert torch.all(videos[1] == 0)
        assert torch.all(masks[1] == 0)

    def test_4d_with_history_raises(self, mock_policy):
        mock_policy.config.n_obs_steps = 4
        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        with pytest.raises(ValueError, match="Expected 5D"):
            mock_policy.prepare_videos(batch)


class TestTemporalSinusoidalPE:
    """Unit tests for the standalone temporal-PE builder."""

    def test_current_frame_row_is_zero(self):
        from opentau.policies.pi05_mem.video_encoder import _build_temporal_sinusoidal_pe

        pe = _build_temporal_sinusoidal_pe(num_frames=8, embed_dim=64)
        assert pe.shape == (8, 64)
        # Current frame lives at t = T-1; row must be all zeros for T=1
        # invariance to hold.
        assert torch.all(pe[-1] == 0)

    def test_earlier_rows_are_nonzero(self):
        from opentau.policies.pi05_mem.video_encoder import _build_temporal_sinusoidal_pe

        pe = _build_temporal_sinusoidal_pe(num_frames=8, embed_dim=64)
        # At least one earlier row should be non-zero (sinusoidal content).
        assert torch.any(pe[0] != 0)

    def test_single_frame_produces_zero_row(self):
        from opentau.policies.pi05_mem.video_encoder import _build_temporal_sinusoidal_pe

        pe = _build_temporal_sinusoidal_pe(num_frames=1, embed_dim=64)
        assert pe.shape == (1, 64)
        assert torch.all(pe == 0)

    def test_odd_embed_dim_raises(self):
        from opentau.policies.pi05_mem.video_encoder import _build_temporal_sinusoidal_pe

        with pytest.raises(ValueError, match="divisible by 2"):
            _build_temporal_sinusoidal_pe(num_frames=4, embed_dim=63)

    def test_zero_num_frames_raises(self):
        from opentau.policies.pi05_mem.video_encoder import _build_temporal_sinusoidal_pe

        with pytest.raises(ValueError, match="num_frames"):
            _build_temporal_sinusoidal_pe(num_frames=0, embed_dim=64)


class TestSpaceTimeSiglipVideoEncoder:
    """CPU-only smoke tests for SpaceTimeSiglipVideoEncoder.

    The encoder takes ``vision_tower`` + ``multi_modal_projector`` by
    reference; the tests build small randomly-initialized SigLIP / projector
    pairs directly from ``PaliGemmaConfig`` so no HF download is needed.
    """

    @staticmethod
    def _build_siglip_and_projector():
        """Construct a fresh, randomly-initialized SigLIP + projector pair
        with the same config PaliGemma uses."""
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
        return vision_tower, projector

    def _make_encoder(self, num_frames: int = 2, spacetime_stride: int = 4):
        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        vision_tower, projector = self._build_siglip_and_projector()
        return SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=num_frames,
            spacetime_layer_stride=spacetime_stride,
        )

    def test_forward_shape(self):
        encoder = self._make_encoder(num_frames=2)
        encoder.eval()
        with torch.no_grad():
            video = torch.rand(1, 2, 3, 224, 224)
            out = encoder(video)
        assert out.shape == (1, 256, 2048)
        assert out.dtype == torch.float32

    def test_wrong_num_frames_raises(self):
        encoder = self._make_encoder(num_frames=4)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="frames"):
            encoder(torch.rand(1, 2, 3, 224, 224))

    def test_wrong_ndim_raises(self):
        encoder = self._make_encoder(num_frames=2)
        encoder.eval()
        with torch.no_grad(), pytest.raises(ValueError, match="5D"):
            encoder(torch.rand(2, 3, 224, 224))

    def test_wrapped_layer_count(self):
        """Verify every ``stride``-th layer (1-indexed from the Nth) is wrapped."""
        from opentau.policies.pi05_mem.video_encoder import SpaceTimeEncoderLayerWrapper

        encoder = self._make_encoder(num_frames=2, spacetime_stride=4)
        layers = encoder.vision_tower.vision_model.encoder.layers
        wrapped_indices = [
            i for i, layer in enumerate(layers) if isinstance(layer, SpaceTimeEncoderLayerWrapper)
        ]
        # 27 SigLIP layers; every 4th starting at index 3 -> [3, 7, 11, 15, 19, 23]
        assert wrapped_indices == [3, 7, 11, 15, 19, 23]

    def test_temporal_pe_on_base_layer_device(self):
        """The temporal PE buffer must be constructed on the base layer's
        device/dtype, not default CPU.

        Regression test for a bug where PaliGemma was moved to a non-default
        device BEFORE the encoder wrapped its layers, leaving the wrapper's
        PE on CPU. Uses a bfloat16 cast as a proxy for "parent was moved
        before wrapping" (dtype and device are moved by the same .to()
        mechanism, so fixing one catches the other)."""
        import copy

        from opentau.policies.pi05_mem.video_encoder import (
            SpaceTimeEncoderLayerWrapper,
            SpaceTimeSiglipVideoEncoder,
        )

        vision_tower, projector = self._build_siglip_and_projector()
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
                assert layer._temporal_pe.device == ref_param.device, (
                    f"PE device {layer._temporal_pe.device} != vision_tower {ref_param.device}"
                )
                assert layer._temporal_pe.dtype == ref_param.dtype, (
                    f"PE dtype {layer._temporal_pe.dtype} != vision_tower {ref_param.dtype}"
                )

    def test_forward_after_external_dtype_cast(self):
        """End-to-end forward must succeed when vision_tower was moved to a
        different dtype BEFORE wrapping (mirrors the GPU load flow of
        ``paligemma.to(cuda/bf16)`` then construct encoder)."""
        import copy

        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        vision_tower, projector = self._build_siglip_and_projector()
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
        assert out.shape == (1, 256, 2048)
        assert out.dtype == torch.bfloat16

    def test_state_dict_keys_match_vanilla_siglip(self):
        """The wrapped vision_tower's state_dict must have identical keys to
        an unwrapped SigLIP. This guarantees that a pi05 checkpoint loads
        without key remapping."""
        from transformers import SiglipVisionModel

        vision_tower, projector = self._build_siglip_and_projector()
        reference_keys = set(SiglipVisionModel(vision_tower.config).state_dict().keys())

        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        SpaceTimeSiglipVideoEncoder(
            vision_tower=vision_tower,
            multi_modal_projector=projector,
            num_frames=4,
            spacetime_layer_stride=4,
        )
        wrapped_keys = set(vision_tower.state_dict().keys())
        assert wrapped_keys == reference_keys, (
            f"state_dict keys diverged; extra in wrapped: {wrapped_keys - reference_keys}, "
            f"missing from wrapped: {reference_keys - wrapped_keys}"
        )

    def test_single_frame_invariance_structural(self):
        """At T=1 the wrapper must short-circuit: byte-identical output to a
        stride-999 (no-wrapping) encoder sharing the exact same underlying
        weights."""
        import copy

        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        torch.manual_seed(0)
        vision_tower, projector = self._build_siglip_and_projector()
        # Deep-copy so the two encoders have independent-but-identical weights;
        # otherwise wrapping mutates the shared vision_tower.
        vt_no_st = copy.deepcopy(vision_tower)
        proj_no_st = copy.deepcopy(projector)
        vt_st = vision_tower
        proj_st = projector

        enc_no_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_no_st,
            multi_modal_projector=proj_no_st,
            num_frames=1,
            spacetime_layer_stride=999,  # > 27 -> no layers wrapped
        ).eval()
        enc_st = SpaceTimeSiglipVideoEncoder(
            vision_tower=vt_st,
            multi_modal_projector=proj_st,
            num_frames=1,
            spacetime_layer_stride=4,  # 6 layers wrapped
        ).eval()

        # With the current design, state_dict keys are identical between the
        # two — adopt-submodules wrapping doesn't introduce .base_layer. keys.
        assert set(vt_no_st.state_dict().keys()) == set(vt_st.state_dict().keys())

        video = torch.rand(1, 1, 3, 224, 224)
        with torch.no_grad():
            out_no_st = enc_no_st(video)
            out_st = enc_st(video)

        torch.testing.assert_close(out_st, out_no_st, rtol=1e-5, atol=1e-5)
