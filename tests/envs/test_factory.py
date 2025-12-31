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

import pytest

from src.opentau.envs.configs import LiberoEnv, MetaworldEnv
from src.opentau.envs.factory import make_env_config, make_envs
from src.opentau.configs.train import TrainPipelineConfig


class TestMakeEnvConfig:
    """Test cases for make_env_config function"""

    def test_make_env_config_metaworld(self):
        """Test making metaworld environment config"""
        config = make_env_config("metaworld", env_name="button-press-v3", episode_length=100)

        assert isinstance(config, MetaworldEnv)
        assert config.env_name == "button-press-v3"
        assert config.episode_length == 100
        assert config.type == "metaworld"

    def test_make_env_config_metaworld_defaults(self):
        """Test making metaworld environment config with defaults"""
        config = make_env_config("metaworld")

        assert isinstance(config, MetaworldEnv)
        assert config.env_name == "button-press-v3"  # Default value
        assert config.episode_length == 200  # Default value
        assert config.type == "metaworld"

    def test_make_env_config_invalid_type(self):
        """Test making environment config with invalid type"""
        with pytest.raises(ValueError, match="Env type 'invalid' is not available"):
            make_env_config("invalid")

    def test_make_env_config_metaworld_with_all_params(self):
        """Test making metaworld environment config with all parameters"""
        config = make_env_config(
            "metaworld", env_name="peg-insert-v3", render_mode="human", episode_length=200, fps=20
        )

        assert isinstance(config, MetaworldEnv)
        assert config.env_name == "peg-insert-v3"
        assert config.render_mode == "human"
        assert config.episode_length == 200
        assert config.fps == 20

    def test_make_env_config_libero(self):
        config = make_env_config("libero")
        assert isinstance(config, LiberoEnv)
        assert config.task == "libero_10"
        assert config.task_ids is None


class TestMakeEnv:
    """Test cases for make_env function"""

    @pytest.fixture
    def mock_train_cfg(self):
        """Create a mock training configuration"""
        cfg = Mock(spec=TrainPipelineConfig)
        return cfg

    @pytest.fixture
    def mock_env_config(self):
        """Create a mock environment configuration"""
        config = Mock(spec=MetaworldEnv)
        config.import_name = "src.opentau.envs.metaworld"
        config.make_id = "Metaworld"
        config.already_vectorized = False  # Default to False for most tests
        config.gym_kwargs = {
            "task": "Meta-World/MT1",
            "env_name": "button-press-v3",
            "render_mode": "rgb_array",
            "max_episode_steps": 50,
            "already_vectorized": False,
        }
        return config

    @pytest.fixture
    def libero_env_config(self):
        """Create a mock environment configuration"""
        return LiberoEnv(task_ids=[0])

    def test_make_env_invalid_n_envs(self, mock_env_config, mock_train_cfg):
        """Test make_env with invalid n_envs"""
        with pytest.raises(ValueError, match="`n_envs must be at least 1"):
            make_envs(mock_env_config, mock_train_cfg, n_envs=0)

    def test_make_env_negative_n_envs(self, mock_env_config, mock_train_cfg):
        """Test make_env with negative n_envs"""
        with pytest.raises(ValueError, match="`n_envs must be at least 1"):
            make_envs(mock_env_config, mock_train_cfg, n_envs=-1)

    @patch("src.opentau.envs.factory.importlib.import_module")
    def test_make_env_module_not_found(self, mock_import_module, mock_env_config, mock_train_cfg):
        """Test make_env when module is not found"""
        mock_import_module.side_effect = ModuleNotFoundError("No module named 'test_module'")

        with pytest.raises(ModuleNotFoundError, match="No module named 'test_module'"):
            make_envs(mock_env_config, mock_train_cfg, n_envs=1)

    @patch("src.opentau.envs.libero.LiberoEnv")
    def test_make_env_sync_vector_env(self, mock_libero_env_cls, libero_env_config, mock_train_cfg):
        mock_libero_env_inst = Mock()
        mock_libero_env_cls.return_value = mock_libero_env_inst

        # Mock SyncVectorEnv
        with patch("gymnasium.vector.SyncVectorEnv") as mock_sync_vector:
            mock_vector_env = Mock()
            mock_sync_vector.return_value = mock_vector_env

            result = make_envs(libero_env_config, mock_train_cfg, n_envs=3, use_async_envs=False)

            # Check that SyncVectorEnv was created
            mock_sync_vector.assert_called_once()

            # Check that SyncVectorEnv was called with a list of lambda functions
            call_args = mock_sync_vector.call_args[0][0]
            assert len(call_args) == 3
            assert all(callable(func) for func in call_args)

            # Test that lambda functions work correctly
            for func in call_args:
                func_result = func()
                assert func_result is mock_libero_env_inst

            assert isinstance(result, dict)
            assert isinstance(result.get("libero_10"), dict)
            assert result["libero_10"].get(0) is mock_vector_env

    @patch("src.opentau.envs.libero.LiberoEnv")
    def test_make_env_async_vector_env(self, mock_libero_env_cls, libero_env_config, mock_train_cfg):
        mock_libero_env_inst = Mock()
        mock_libero_env_cls.return_value = mock_libero_env_inst

        # Mock SyncVectorEnv
        with patch("gymnasium.vector.AsyncVectorEnv") as mock_async_vector:
            mock_vector_env = Mock()
            mock_async_vector.return_value = mock_vector_env

            result = make_envs(libero_env_config, mock_train_cfg, n_envs=2, use_async_envs=True)

            # Check that SyncVectorEnv was created
            mock_async_vector.assert_called_once()

            # Check that SyncVectorEnv was called with a list of lambda functions
            call_args = mock_async_vector.call_args[0][0]
            assert len(call_args) == 2
            assert all(callable(func) for func in call_args)

            # Test that lambda functions work correctly
            for func in call_args:
                func_result = func()
                assert func_result is mock_libero_env_inst

            assert isinstance(result, dict)
            assert isinstance(result.get("libero_10"), dict)
            assert result["libero_10"].get(0) is mock_vector_env

    def test_make_env_import_error_message(self, mock_env_config, mock_train_cfg):
        """Test that import error includes helpful installation message"""
        with patch("src.opentau.envs.factory.importlib.import_module") as mock_import_module:
            mock_import_module.side_effect = ModuleNotFoundError("No module named 'test_module'")

            with pytest.raises(ModuleNotFoundError) as exc_info:
                make_envs(mock_env_config, mock_train_cfg, n_envs=1)

            # The error should be re-raised as-is
            assert "No module named 'test_module'" in str(exc_info.value)


class TestMakeEnvIntegration:
    """Integration tests for make_env function"""

    @pytest.fixture
    def real_env_config(self):
        """Create a real MetaworldEnv config"""
        return MetaworldEnv(env_name="button-press-v3", render_mode="rgb_array", episode_length=50)

    @pytest.fixture
    def real_train_cfg(self):
        """Create a real TrainPipelineConfig"""
        cfg = Mock(spec=TrainPipelineConfig)
        return cfg

    @patch("src.opentau.envs.factory.importlib.import_module")
    @patch("src.opentau.envs.factory.gym.make")
    def test_make_env_gym_kwargs_from_real_config(
        self, mock_gym_make, mock_import_module, real_env_config, real_train_cfg
    ):
        """Test that gym_kwargs from real config are used correctly"""
        mock_import_module.return_value = Mock()

        # Mock gym.make to return a mock environment
        mock_env = Mock()
        mock_gym_make.return_value = mock_env

        # Mock SyncVectorEnv
        with patch("src.opentau.envs.factory.gym.vector.SyncVectorEnv") as mock_sync_vector:
            mock_vector_env = Mock()
            mock_sync_vector.return_value = mock_vector_env

            make_envs(real_env_config, real_train_cfg, n_envs=1)

            # Check that SyncVectorEnv was called with a lambda function
            call_args = mock_sync_vector.call_args[0][0]
            assert len(call_args) == 1
            assert callable(call_args[0])

            # Execute the lambda function to trigger gym.make
            call_args[0]()

            # Check that gym_kwargs from the config are used
            call_kwargs = mock_gym_make.call_args[1]
            assert call_kwargs["task"] == "Meta-World/MT1"
            assert call_kwargs["env_name"] == "button-press-v3"
            assert call_kwargs["render_mode"] == "rgb_array"
            assert call_kwargs["max_episode_steps"] == 50
