#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from unittest.mock import Mock, patch

import pytest

from opentau.configs.train import TrainPipelineConfig
from opentau.envs.configs import LiberoEnv, RoboCasaEnv
from opentau.envs.factory import _pin_egl_render_device, make_env_config, make_envs


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

    def test_make_env_config_robocasa(self):
        config = make_env_config("robocasa")
        assert isinstance(config, RoboCasaEnv)
        assert config.task == "CloseFridge"


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

    @patch("opentau.envs.libero.LiberoEnv")
    def test_make_env_pins_egl_device_before_spawning_workers(
        self, mock_libero_env_cls, libero_env_config, mock_train_cfg
    ):
        """make_envs pins the per-rank EGL device *before* the vec env spawns workers.

        The ordering is the crux of the fix: the spawn workers inherit os.environ,
        so MUJOCO_EGL_DEVICE_ID must be set before the vec env is constructed.
        """
        mock_libero_env_cls.return_value = Mock()
        manager = Mock()
        with (
            patch("opentau.envs.factory._pin_egl_render_device", manager.pin_egl),
            patch("gymnasium.vector.SyncVectorEnv", manager.sync_vector),
        ):
            make_envs(libero_env_config, mock_train_cfg, n_envs=1, use_async_envs=False)
        ordered = [c[0] for c in manager.mock_calls]
        assert ordered.index("pin_egl") < ordered.index("sync_vector")


class TestPinEglRenderDevice:
    """`_pin_egl_render_device` pins each rank's MuJoCo EGL render to its own GPU.

    Without it, ``mujoco.egl`` defaults every rank to EGL device 0, so multi-GPU
    sim eval piles all ranks onto GPU 0 and OOMs it. These are pure env-var checks
    (no GPU, no sim import); ``monkeypatch`` restores the environment after each.
    """

    def test_falls_back_to_local_index_when_no_cuda_visible_devices(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "egl")
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch("opentau.envs.factory.get_proc_accelerator", return_value=Mock(local_process_index=3)):
            assert _pin_egl_render_device() == "3"
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"

    def test_maps_through_masked_cuda_visible_devices(self, monkeypatch):
        # Rank 1 of a job pinned to physical GPUs 4-7 must render on GPU 5, not 1 —
        # the raw local index would also trip robosuite's CUDA_VISIBLE_DEVICES assert.
        monkeypatch.setenv("MUJOCO_GL", "egl")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        with patch("opentau.envs.factory.get_proc_accelerator", return_value=Mock(local_process_index=1)):
            assert _pin_egl_render_device() == "5"
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"

    def test_noop_when_render_backend_is_not_egl(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "osmesa")
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        with patch("opentau.envs.factory.get_proc_accelerator", return_value=Mock(local_process_index=2)):
            assert _pin_egl_render_device() is None
        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ

    def test_respects_explicit_device_id(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "egl")
        monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "5")
        with patch("opentau.envs.factory.get_proc_accelerator", return_value=Mock(local_process_index=3)):
            assert _pin_egl_render_device() is None
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"

    def test_noop_without_accelerator(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "egl")
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        with patch("opentau.envs.factory.get_proc_accelerator", return_value=None):
            assert _pin_egl_render_device() is None
        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
