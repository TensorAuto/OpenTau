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
        assert config.n_obs_history is None
        assert config.history_interval is None
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.vjepa2_dtype is None

    def test_obs_buffer_size_no_history(self):
        config = PI05MemConfig()
        assert config.obs_buffer_size == 1

    def test_obs_buffer_size_with_history(self):
        config = PI05MemConfig(n_obs_history=4, history_interval=2)
        assert config.obs_buffer_size == (4 - 1) * 2 + 1  # 7

    def test_obs_buffer_size_interval_defaults_to_1(self):
        config = PI05MemConfig(n_obs_history=8)
        assert config.history_interval == 1
        assert config.obs_buffer_size == 8

    def test_invalid_n_obs_history(self):
        with pytest.raises(ValueError, match="n_obs_history"):
            PI05MemConfig(n_obs_history=0)

    def test_invalid_history_interval(self):
        with pytest.raises(ValueError, match="history_interval"):
            PI05MemConfig(n_obs_history=4, history_interval=-1)

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
        policy.config = PI05MemConfig(n_obs_history=4, history_interval=1)
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
        policy.config = PI05MemConfig(max_state_dim=8)
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
        policy.config = PI05MemConfig(max_state_dim=8, n_obs_history=4)
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
        policy.config = PI05MemConfig(
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
        mock_policy.config.n_obs_history = 4
        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        with pytest.raises(ValueError, match="Expected 5D"):
            mock_policy.prepare_videos(batch)
