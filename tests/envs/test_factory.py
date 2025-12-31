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

from opentau.envs.configs import LiberoEnv, MetaworldEnv
from opentau.envs.factory import make_env_config, make_envs
from opentau.configs.train import TrainPipelineConfig


class TestMakeEnvConfig:
    """Test cases for make_env_config function"""

    def test_make_env_config_invalid_type(self):
        """Test making environment config with invalid type"""
        with pytest.raises(ValueError, match="Env type 'invalid' is not available"):
            make_env_config("invalid")

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
    def libero_env_config(self):
        """Create a mock environment configuration"""
        return LiberoEnv(task_ids=[0])

    @patch("opentau.envs.libero.LiberoEnv")
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

    @patch("opentau.envs.libero.LiberoEnv")
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

