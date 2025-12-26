#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.common.envs.metaworld import Metaworld, register_lerobot_metaworld_envs
from lerobot.configs.train import TrainPipelineConfig


class TestMetaworld:
    """Test cases for Metaworld environment wrapper"""

    @pytest.fixture
    def mock_train_cfg(self):
        """Create a mock training configuration"""
        cfg = Mock(spec=TrainPipelineConfig)
        cfg.resolution = (84, 84)
        cfg.max_state_dim = 10
        cfg.num_cams = 1
        cfg.action_chunk = 4
        return cfg

    @pytest.fixture
    def mock_underlying_env(self):
        """Create a mock underlying gymnasium environment"""
        env = Mock()
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        env.spec = Mock()
        env.unwrapped = Mock()

        # Mock reset and step methods
        env.reset.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), {})
        env.step.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), 1.0, False, False, {})
        env.render.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        env.close.return_value = None
        env.seed.return_value = [42]

        return env

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_init_default_params(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test Metaworld initialization with default parameters"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        # Check that gym.make was called with correct parameters
        mock_gym_make.assert_called_once_with(
            "Meta-World/MT1",
            env_name="button-press-v3",
            render_mode="rgb_array",
            camera_name="corner",
            max_episode_steps=200,
        )

        # Check basic attributes
        assert env.task == "Meta-World/MT1"
        assert env.env_name == "button-press-v3"
        assert env.render_mode == "rgb_array"
        assert env.max_episode_steps == 200
        assert env.camera_name == "corner"
        assert env.train_cfg == mock_train_cfg
        assert env._episode_steps == 0
        assert env._episode_reward == 0.0

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_init_custom_params(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test Metaworld initialization with custom parameters"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(  # noqa: F841
            task="Meta-World/MT10",
            env_name="peg-insert-v3",
            render_mode="human",
            max_episode_steps=100,
            camera_name="topview",
            train_cfg=mock_train_cfg,
            custom_param="test",
        )

        # Check that gym.make was called with custom parameters
        mock_gym_make.assert_called_once_with(
            "Meta-World/MT1",
            env_name="peg-insert-v3",
            render_mode="human",
            camera_name="topview",
            max_episode_steps=100,
            custom_param="test",
        )

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_init_without_train_cfg(self, mock_gym_make, mock_underlying_env):
        """Test Metaworld initialization without train_cfg"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld()

        # Should use original observation space when no train_cfg
        assert env.observation_space == mock_underlying_env.observation_space

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_observation_space_with_train_cfg(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test observation space definition with train_cfg"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        # Check observation space structure
        assert hasattr(env.observation_space, "spaces")
        assert "camera0" in env.observation_space.spaces
        assert "state" in env.observation_space.spaces
        assert "prompt" in env.observation_space.spaces
        assert "img_is_pad" in env.observation_space.spaces
        assert "action_is_pad" in env.observation_space.spaces

        # Check camera0 space shape
        camera0_space = env.observation_space.spaces["camera0"]
        assert camera0_space.shape == (3, 84, 84)
        assert camera0_space.dtype == np.float32

        # Check state space
        state_space = env.observation_space.spaces["state"]
        assert state_space.shape == (10,)
        assert state_space.dtype == np.float32

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_to_standard_data_format(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test _to_standard_data_format method"""
        mock_gym_make.return_value = mock_underlying_env

        # Mock render to return a specific image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_underlying_env.render.return_value = test_image

        env = Metaworld(train_cfg=mock_train_cfg)

        # Mock the resize_with_pad function
        with patch("lerobot.common.envs.metaworld.resize_with_pad") as mock_resize:
            mock_resize.return_value = torch.randn(1, 3, 84, 84)

            observation = np.array([0.1, 0.2, 0.3, 0.4])
            result = env._to_standard_data_format(observation)

            # Check result structure
            assert "camera0" in result
            assert "state" in result
            assert "prompt" in result
            assert "img_is_pad" in result
            assert "action_is_pad" in result

            # Check camera0
            assert result["camera0"].shape == (3, 84, 84)
            assert result["camera0"].dtype == np.float32

            # Check state
            assert result["state"].shape == (10,)
            assert result["state"].dtype == np.float32
            assert np.array_equal(result["state"][:4], observation.astype(np.float32))
            assert np.all(result["state"][4:] == 0)  # Padded zeros

            # Check prompt
            assert result["prompt"] == "Press a button."

            # Check padding arrays
            assert result["img_is_pad"].shape == (1,)
            assert result["img_is_pad"].dtype == bool
            assert result["action_is_pad"].shape == (4,)
            assert result["action_is_pad"].dtype == bool

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_reset(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test reset method"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        # Mock _to_standard_data_format
        with patch.object(env, "_to_standard_data_format") as mock_format:
            mock_format.return_value = {"test": "observation"}

            observation, info = env.reset(seed=42)

            # Check that underlying env reset was called
            mock_underlying_env.reset.assert_called_once_with(seed=42, options=None)

            # Check that format was called
            mock_format.assert_called_once()

            # Check episode tracking reset
            assert env._episode_steps == 0
            assert env._episode_reward == 0.0

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_reset_with_options(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test reset method with options"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        with patch.object(env, "_to_standard_data_format") as mock_format:
            mock_format.return_value = {"test": "observation"}

            options = {"test_option": "value"}
            observation, info = env.reset(seed=42, options=options)

            mock_underlying_env.reset.assert_called_once_with(seed=42, options=options)

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_step(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test step method"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        # Mock _to_standard_data_format
        with patch.object(env, "_to_standard_data_format") as mock_format:
            mock_format.return_value = {"test": "observation"}

            action = np.array([0.1, 0.2, 0.3, 0.4])
            observation, reward, terminated, truncated, info = env.step(action)

            # Check that underlying env step was called
            mock_underlying_env.step.assert_called_once_with(action)

            # Check that format was called
            mock_format.assert_called_once()

            # Check episode tracking
            assert env._episode_steps == 1
            assert env._episode_reward == 1.0

            # Check custom info
            assert "episode_steps" in info
            assert "episode_reward" in info
            assert info["episode_steps"] == 1
            assert info["episode_reward"] == 1.0

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_render_corner_camera(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test render method with corner camera (should flip image)"""
        mock_gym_make.return_value = mock_underlying_env

        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_underlying_env.render.return_value = test_image

        env = Metaworld(train_cfg=mock_train_cfg, camera_name="corner")

        with patch("lerobot.common.envs.metaworld.cv2.flip") as mock_flip:
            flipped_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_flip.return_value = flipped_image

            result = env.render()

            # Check that cv2.flip was called with the correct arguments
            mock_flip.assert_called_once()
            call_args = mock_flip.call_args[0]
            assert call_args[1] == 0  # Check flip code
            # Don't check the exact array since it's randomly generated
            assert result is flipped_image

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_render_non_corner_camera(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test render method with non-corner camera (should not flip image)"""
        mock_gym_make.return_value = mock_underlying_env

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_underlying_env.render.return_value = test_image

        env = Metaworld(train_cfg=mock_train_cfg, camera_name="topview")

        result = env.render()

        # Should return original image without flipping
        assert result is test_image

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_render_none_image(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test render method when underlying env returns None"""
        mock_gym_make.return_value = mock_underlying_env
        mock_underlying_env.render.return_value = None

        env = Metaworld(train_cfg=mock_train_cfg)

        result = env.render()

        # Should return None
        assert result is None

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_close(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test close method"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)
        env.close()

        mock_underlying_env.close.assert_called_once()

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_close_no_close_method(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test close method when underlying env has no close method"""
        mock_gym_make.return_value = mock_underlying_env
        del mock_underlying_env.close

        env = Metaworld(train_cfg=mock_train_cfg)

        # Should not raise an error
        env.close()

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_seed(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test seed method"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)
        result = env.seed(42)

        mock_underlying_env.seed.assert_called_once_with(42)
        assert result == [42]

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_seed_no_seed_method(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test seed method when underlying env has no seed method"""
        mock_gym_make.return_value = mock_underlying_env
        del mock_underlying_env.seed

        env = Metaworld(train_cfg=mock_train_cfg)
        result = env.seed(42)

        assert result == [42]

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_unwrapped_property(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test unwrapped property"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)
        result = env.unwrapped

        assert result is mock_underlying_env.unwrapped

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_metadata(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test metadata property"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        assert env.metadata == {"render_modes": ["rgb_array", "human"], "render_fps": 80}

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_action_space(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test action_space property"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        assert env.action_space is mock_underlying_env.action_space

    @patch("lerobot.common.envs.metaworld.gym.make")
    def test_spec_property(self, mock_gym_make, mock_underlying_env, mock_train_cfg):
        """Test spec property"""
        mock_gym_make.return_value = mock_underlying_env

        env = Metaworld(train_cfg=mock_train_cfg)

        assert env.spec is mock_underlying_env.spec


class TestRegisterLeRobotMetaworldEnvs:
    """Test cases for register_lerobot_metaworld_envs function"""

    @patch("lerobot.common.envs.metaworld.gym.register")
    def test_register_lerobot_metaworld_envs(self, mock_register):
        """Test that environments are registered correctly"""
        register_lerobot_metaworld_envs()

        mock_register.assert_called_once_with(
            id="Metaworld",
            entry_point="lerobot.common.envs.metaworld:Metaworld",
        )

    @patch("lerobot.common.envs.metaworld.gym.register")
    def test_register_already_registered(self, mock_register):
        """Test that registration raises exception when already registered"""
        mock_register.side_effect = Exception("Already registered")

        # Should raise an exception since the function doesn't handle it internally
        with pytest.raises(Exception, match="Already registered"):
            register_lerobot_metaworld_envs()

        mock_register.assert_called_once()

    @patch("lerobot.common.envs.metaworld.gym.register")
    def test_register_module_not_available(self, mock_register):
        """Test that registration raises exception when module not available"""
        mock_register.side_effect = ModuleNotFoundError("No module named 'metaworld'")

        # Should raise an exception since the function doesn't handle it internally
        with pytest.raises(ModuleNotFoundError, match="No module named 'metaworld'"):
            register_lerobot_metaworld_envs()

        mock_register.assert_called_once()
