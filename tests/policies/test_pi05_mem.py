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


# Padded-history attention masking. Pixel-zeroing of padded frames is not
# enough — the SigLIP patch embedding has a learned bias and the temporal
# positional embedding e(t) is non-zero for t < T-1, so padded "zero" frames
# still produce non-zero hidden states the current frame would otherwise
# attend to. The encoder must build an attention mask blocking padded keys.


class TestTemporalAttentionMaskBlocksPaddedHistory:
    @staticmethod
    def _make_encoder(num_frames: int = 4, spacetime_stride: int = 4):
        helper = TestSpaceTimeSiglipVideoEncoder()
        return helper._make_encoder(num_frames=num_frames, spacetime_stride=spacetime_stride)

    def test_current_frame_invariant_to_padded_history(self):
        """Gold-standard: with the first three frames marked padded and an
        identical current frame at ``[:, -1]``, the encoder output must NOT
        depend on the pixel values of those padded frames. Pre-fix code
        reads contaminated hidden states from padded frames through the
        temporal attention path and the two outputs diverge.
        """
        torch.manual_seed(0)
        encoder = self._make_encoder(num_frames=4, spacetime_stride=4).eval()

        current = torch.rand(1, 1, 3, 224, 224)
        vid_a = torch.cat([torch.rand(1, 3, 3, 224, 224), current], dim=1)
        vid_b = torch.cat([torch.zeros(1, 3, 3, 224, 224), current], dim=1)

        obs_history_is_pad = torch.tensor([[True, True, True, False]])
        with torch.no_grad():
            out_a = encoder(vid_a, obs_history_is_pad=obs_history_is_pad)
            out_b = encoder(vid_b, obs_history_is_pad=obs_history_is_pad)

        torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)

    def test_inference_fallback_blocks_all_history(self):
        """``obs_history_is_pad=None`` triggers the inference fallback that
        treats every history slot as padded; pins the property that
        ``select_action`` (which never emitted ``obs_history_is_pad`` before
        this fix) does not let zero-pixel placeholders contaminate the
        current frame.
        """
        torch.manual_seed(1)
        encoder = self._make_encoder(num_frames=4, spacetime_stride=4).eval()

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
        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        obs_history_is_pad = torch.tensor([[True, False, False]])
        mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
            obs_history_is_pad, num_patches=4, dtype=torch.float32
        )

        # Shape: B=1, N=4, T=3 -> (B*N, 1, T, T) = (4, 1, 3, 3).
        assert mask.shape == (4, 1, 3, 3)
        assert mask.dtype == torch.float32

        # All N rows for batch 0 share the same (T, T) submatrix.
        m = mask[0, 0]  # (3, 3)

        # Row 0: query=frame0 (padded query, but causal allows j=0 only;
        # since frame 0 is padded its key is blocked -> -inf). The
        # current-frame override only forces key j=T-1 real, not j=0.
        assert m[0, 0] == float("-inf"), "padded frame 0 key should be blocked"
        # Row 1: query=frame1 attends to {0 (padded -> -inf), 1 (real -> 0)}.
        assert m[1, 0] == float("-inf")
        assert m[1, 1] == 0.0
        # Row 2: query=current attends to {0 (padded -> -inf), 1 (real -> 0),
        # 2 (current, always real -> 0)}.
        assert m[2, 0] == float("-inf")
        assert m[2, 1] == 0.0
        assert m[2, 2] == 0.0

        # Upper triangular off-diagonals are blocked (causal).
        assert m[0, 1] == float("-inf")
        assert m[0, 2] == float("-inf")
        assert m[1, 2] == float("-inf")

        # Patch rows for the same batch are identical.
        for n in range(1, 4):
            torch.testing.assert_close(mask[n, 0], mask[0, 0])

    def test_current_frame_override_when_pad_includes_current(self):
        """Defensive: even if a caller passes ``obs_history_is_pad`` with the
        current step (T-1) marked padded — as the dataset's
        ``history_state_drop_prob`` augmentation does — the mask must keep
        the current frame attendable, otherwise the current query has no
        valid key and softmax produces NaNs.
        """
        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        all_padded = torch.tensor([[True, True, True, True]])
        mask = SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(
            all_padded, num_patches=1, dtype=torch.float32
        )
        m = mask[0, 0]
        assert m[3, 3] == 0.0, "current frame must remain self-attendable"

    def test_pad_tensor_not_mutated(self):
        """The mask helper must not mutate the caller's ``obs_history_is_pad``
        tensor (it is also consumed downstream as the state mask in
        ``embed_prefix``).
        """
        from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

        all_padded = torch.tensor([[True, True, True, True]])
        snapshot = all_padded.clone()
        SpaceTimeSiglipVideoEncoder._build_temporal_attn_mask(all_padded, num_patches=1, dtype=torch.float32)
        torch.testing.assert_close(all_padded, snapshot)


# State mask: current step (t = T-1) must always be marked real, even when
# obs_history_is_pad sets it to True (e.g. dataset's history_state_drop_prob
# augmentation flips the entire tensor to all-True). Without the override the
# policy is conditioned on no state at all when that augmentation fires.


def _make_embed_prefix_stub():
    """Construct a partial PI05MemFlowMatching that exposes only the attrs
    ``embed_prefix`` reads — no PaliGemma init, just enough for the layout
    paths exercised by the state-mask tests.
    """
    import types

    from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

    hidden = 4
    n_video_tokens = 3
    fm = PI05MemFlowMatching.__new__(PI05MemFlowMatching)

    class _FakePaligemma:
        def embed_language_tokens(self, tokens):
            return torch.zeros((*tokens.shape, hidden), dtype=torch.float32)

        def embed_discrete_actions(self, da):
            return torch.zeros((*da.shape, hidden), dtype=torch.float32)

    fm.paligemma_with_expert = _FakePaligemma()
    fm.state_proj = lambda state: torch.zeros(state.shape[0], state.shape[1], hidden, dtype=torch.float32)
    fm.embed_video = lambda video, obs_history_is_pad=None: torch.zeros(
        video.shape[0], n_video_tokens, hidden, dtype=torch.float32
    )
    fm.config = types.SimpleNamespace(discrete_action_max_length=2)
    return fm


def _embed_prefix_default_inputs(*, batch_size: int = 2, prompt_len: int = 3, t_state: int = 1):
    return {
        "videos": [torch.zeros(batch_size, t_state, 3, 4, 4)],
        "vid_masks": [torch.ones(batch_size, dtype=torch.bool)],
        "lang_tokens": torch.zeros(batch_size, prompt_len, dtype=torch.long),
        "lang_masks": torch.ones(batch_size, prompt_len, dtype=torch.bool),
        "state": torch.zeros(batch_size, t_state, 7),
    }


def _state_slice_indices(prompt_len: int, n_video_tokens: int, t_state: int) -> slice:
    """Layout: videos(n_video_tokens) + lang(prompt_len) + state(t_state)."""
    state_lo = n_video_tokens + prompt_len
    return slice(state_lo, state_lo + t_state)


class TestStateMaskCurrentStepAlwaysReal:
    """Pin the post-fix invariant: state_mask[:, -1] is True regardless of
    obs_history_is_pad. (Bug B from the audit; ported from pi07 / PR #205.)
    """

    def test_state_mask_current_step_real_when_all_history_padded(self):
        """``obs_history_is_pad = ones(B, T)`` (the
        ``history_state_drop_prob=1.0`` case) MUST still leave the current
        state token (index T-1) marked real, otherwise attention to it is
        masked out and the policy conditions on no state at all.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = torch.ones(bsize, t_state, dtype=torch.bool)

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        assert state_mask.shape == (bsize, t_state)

        for i in range(bsize):
            assert state_mask[i, -1].item() is True, (
                f"sample {i}: current state token (T-1) is masked out — the "
                f"history_state_drop_prob augmentation would condition on no "
                f"state at all. state_mask = {state_mask[i].tolist()}"
            )
        assert (~state_mask[:, :-1]).all().item() is True

    def test_state_mask_none_branch_assumes_history_padded_keeps_current_real(self):
        """``obs_history_is_pad = None`` means the caller didn't tell us
        which slots are real. Post-fix: assume all history is padded so the
        encoder cannot attend to garbage history slots — but the current
        step is still real.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = None

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        assert state_mask.shape == (bsize, t_state)

        for i in range(bsize):
            assert state_mask[i, -1].item() is True
            assert (~state_mask[i, :-1]).all().item() is True

    def test_state_mask_partial_history_pad_preserves_current(self):
        """Mixed pad pattern (typical of natural episode-boundary padding):
        some history slots padded, current step real -> state_mask matches
        ``~obs_history_is_pad`` exactly, with the override a no-op since the
        current bit was already True.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = torch.tensor([[True, True, False, False], [True, False, False, False]])

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        torch.testing.assert_close(
            state_mask,
            torch.tensor([[False, False, True, True], [False, True, True, True]]),
        )

    def test_state_mask_does_not_mutate_obs_history_is_pad(self):
        """The override path must not mutate the caller's
        ``obs_history_is_pad`` (it is also threaded into ``embed_video``
        for the temporal attention mask — mutating it there would cause
        cross-call corruption).
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 1
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        original_pad = torch.ones(bsize, t_state, dtype=torch.bool)
        kwargs["obs_history_is_pad"] = original_pad
        snapshot = original_pad.clone()

        PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        torch.testing.assert_close(original_pad, snapshot)


# `_build_history_batch` emits ``obs_history_is_pad`` so the encoder can use
# real mid-episode history while still masking start-of-episode zero-fill.
# Without this emit, the encoder's None-fallback masks ALL history at
# inference (mid-episode regression flagged in pi07's PR #253 review and
# inherited from pi05_mem's original design).


class TestBuildHistoryBatchEmitsObsHistoryIsPad:
    @staticmethod
    def _make_policy_stub(*, n_obs_steps: int, history_interval: int, image_keys: list[str]):
        """Construct a partial PI05MemPolicy that exposes only the attrs
        ``_build_history_batch`` reads.
        """
        import types

        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = PI05MemPolicy.__new__(PI05MemPolicy)
        buf_size = (n_obs_steps - 1) * history_interval + 1
        policy.config = types.SimpleNamespace(
            n_obs_steps=n_obs_steps,
            history_interval=history_interval,
            obs_buffer_size=buf_size,
            image_features=dict.fromkeys(image_keys),
        )
        policy._state_buffer = None
        policy._obs_buffers = None
        return policy

    def _make_batch(self, image_keys: list[str], state_dim: int = 4) -> dict:
        return {
            "state": torch.zeros(1, state_dim),
            **{k: torch.zeros(1, 3, 8, 8) for k in image_keys},
        }

    def test_first_step_marks_all_but_current_padded(self):
        """At episode start, only the very first observation is in the
        buffer; every other slot in the requested history was zero-filled.
        Mask should be ``[True, ..., True, False]`` — the canonical case
        the Bug A fix protects against contamination from.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=1, image_keys=["camera0"])
        out = policy._build_history_batch(self._make_batch(["camera0"]))

        assert "obs_history_is_pad" in out
        assert out["obs_history_is_pad"].shape == (1, 4)
        assert out["obs_history_is_pad"].dtype == torch.bool
        assert out["obs_history_is_pad"].tolist() == [[True, True, True, False]]

    def test_buffer_full_emits_all_false(self):
        """Once the buffer is full (after ``obs_buffer_size`` calls), every
        slot maps to a real observation — mask is all-False. This is the
        mid-episode case the previous behavior regressed: with no emit, the
        encoder masked these real frames out via the None-fallback.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=2, image_keys=["camera0"])
        # obs_buffer_size = (4-1)*2 + 1 = 7. Need 7 calls to fill.
        batch = self._make_batch(["camera0"])
        for _ in range(7):
            out = policy._build_history_batch(batch)
        assert out["obs_history_is_pad"].tolist() == [[False, False, False, False]]

    def test_partial_fill_marks_only_unfilled_slots(self):
        """After ``k < obs_buffer_size`` calls, the leading slots are still
        virtual past-steps. With ``n_obs_steps=4, history_interval=2``
        (buffer_size=7), after 4 calls the deque has 4 entries -> ``missing
        = 3`` -> slots with ``i*interval - 3 < 0`` are padded: i=0 -> -3 (T),
        i=1 -> -1 (T), i=2 -> 1 (F), i=3 -> 3 (F). Mask = [T, T, F, F].
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=2, image_keys=["camera0"])
        batch = self._make_batch(["camera0"])
        for _ in range(4):
            out = policy._build_history_batch(batch)
        assert out["obs_history_is_pad"].tolist() == [[True, True, False, False]]

    def test_mask_is_broadcast_over_batch(self):
        """The buffer is shared across batch elements, so the (B, T) mask is
        the same across the batch dim. Verify by emitting from a B=3 batch.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=1, image_keys=["camera0"])
        batch = {
            "state": torch.zeros(3, 4),
            "camera0": torch.zeros(3, 3, 8, 8),
        }
        out = policy._build_history_batch(batch)

        assert out["obs_history_is_pad"].shape == (3, 4)
        assert torch.all(out["obs_history_is_pad"] == out["obs_history_is_pad"][0:1])

    def test_state_and_camera_padding_match_emitted_mask(self):
        """The emitted mask must agree slot-for-slot with the actual
        zero-padding pattern of state and camera tensors.
        """
        policy = self._make_policy_stub(n_obs_steps=3, history_interval=1, image_keys=["camera0"])
        batch = {
            "state": torch.full((1, 4), 7.0),
            "camera0": torch.full((1, 3, 8, 8), 5.0),
        }
        out = policy._build_history_batch(batch)
        # After one call: missing = 2; mask = [True, True, False].
        is_pad = out["obs_history_is_pad"][0]  # (T,)
        state = out["state"][0]  # (T, D)
        cam = out["camera0"][0]  # (T, C, H, W)
        for t, padded in enumerate(is_pad.tolist()):
            if padded:
                assert torch.all(state[t] == 0.0), f"state[{t}] not zero-filled"
                assert torch.all(cam[t] == 0.0), f"camera[{t}] not zero-filled"
            else:
                assert torch.all(state[t] == 7.0), f"state[{t}] zero-filled but mask says real"
                assert torch.all(cam[t] == 5.0), f"camera[{t}] zero-filled but mask says real"
