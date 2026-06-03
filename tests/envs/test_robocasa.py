#!/usr/bin/env python

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

r"""CPU-only tests for the RoboCasa env integration.

These never import the ``robocasa`` / ``robosuite`` sim packages: the wrapper
defers those imports into ``_ensure_env`` / ``_resolve_tasks``, so config
registration, the factory dispatch, per-rank task sharding, and the pure
action/camera helpers are all exercisable without a GPU or the sim installed.
Real sim rollouts are validated separately on a CUDA box.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from opentau.configs.train import TrainPipelineConfig
from opentau.configs.types import FeatureType
from opentau.constants import ACTION, OBS_IMAGES, OBS_STATE
from opentau.envs.configs import EnvConfig, RoboCasaEnv
from opentau.envs.factory import make_env_config, make_envs
from opentau.envs.robocasa import (
    ACTION_DIM,
    OBS_STATE_DIM,
    _default_camera_name_mapping,
    _parse_camera_names,
    _resolve_tasks,
    convert_action,
    create_robocasa_envs,
)


class TestRoboCasaConfig:
    """Config registration, defaults, and feature wiring."""

    def test_registered_in_choice_registry(self):
        assert "robocasa" in EnvConfig._choice_registry
        assert EnvConfig._choice_registry["robocasa"] is RoboCasaEnv
        assert RoboCasaEnv().type == "robocasa"

    def test_default_values(self):
        cfg = RoboCasaEnv()
        assert cfg.task == "CloseFridge"
        assert cfg.fps == 20
        assert cfg.episode_length == 1000
        assert cfg.obs_type == "pixels_agent_pos"
        assert cfg.observation_height == 256
        assert cfg.observation_width == 256
        assert cfg.split is None
        assert cfg.obj_registries == ["lightwheel"]
        assert len(_parse_camera_names(cfg.camera_name)) == 3

    def test_action_and_state_features(self):
        cfg = RoboCasaEnv()
        assert cfg.features["action"].type == FeatureType.ACTION
        assert cfg.features["action"].shape == (12,)
        # pixels_agent_pos adds a 16-D proprio state.
        assert cfg.features["agent_pos"].type == FeatureType.STATE
        assert cfg.features["agent_pos"].shape == (16,)
        assert cfg.features_map["action"] == ACTION
        assert cfg.features_map["agent_pos"] == OBS_STATE

    def test_camera_features_map_uses_image_convention(self):
        """The 3 raw cameras map to OpenTau's image/image2/image3 keys."""
        cfg = RoboCasaEnv()
        cams = _parse_camera_names(cfg.camera_name)
        assert cfg.features_map[f"pixels/{cams[0]}"] == f"{OBS_IMAGES}.image"
        assert cfg.features_map[f"pixels/{cams[1]}"] == f"{OBS_IMAGES}.image2"
        assert cfg.features_map[f"pixels/{cams[2]}"] == f"{OBS_IMAGES}.image3"
        for cam in cams:
            assert cfg.features[f"pixels/{cam}"].type == FeatureType.VISUAL
            assert cfg.features[f"pixels/{cam}"].shape == (256, 256, 3)

    def test_pixels_only_omits_agent_pos(self):
        cfg = RoboCasaEnv(obs_type="pixels")
        assert "agent_pos" not in cfg.features

    @pytest.mark.parametrize("bad_fps", [0, -5])
    def test_rejects_nonpositive_fps(self, bad_fps):
        with pytest.raises(ValueError, match="must be positive"):
            RoboCasaEnv(fps=bad_fps)

    def test_rejects_unsupported_obs_type(self):
        with pytest.raises(ValueError, match="Unsupported obs_type"):
            RoboCasaEnv(obs_type="state")

    def test_gym_kwargs_carries_obs_params_and_split(self):
        cfg = RoboCasaEnv(split="pretrain")
        kwargs = cfg.gym_kwargs
        assert kwargs["obs_type"] == "pixels_agent_pos"
        assert kwargs["observation_height"] == 256
        assert kwargs["observation_width"] == 256
        assert kwargs["visualization_height"] == 512
        assert kwargs["split"] == "pretrain"

    def test_gym_kwargs_omits_split_when_none(self):
        assert "split" not in RoboCasaEnv().gym_kwargs


class TestMakeEnvConfigRoboCasa:
    """``make_env_config`` dispatch."""

    def test_make_env_config_robocasa(self):
        cfg = make_env_config("robocasa")
        assert isinstance(cfg, RoboCasaEnv)
        assert cfg.task == "CloseFridge"

    def test_make_env_config_robocasa_with_overrides(self):
        cfg = make_env_config("robocasa", task="OpenDrawer", split="target")
        assert cfg.task == "OpenDrawer"
        assert cfg.split == "target"


class TestPureHelpers:
    """Helpers that never touch the sim."""

    def test_convert_action_layout(self):
        flat = np.arange(ACTION_DIM, dtype=np.float32)
        out = convert_action(flat)
        np.testing.assert_array_equal(out["action.base_motion"], flat[0:4])
        np.testing.assert_array_equal(out["action.control_mode"], flat[4:5])
        np.testing.assert_array_equal(out["action.end_effector_position"], flat[5:8])
        np.testing.assert_array_equal(out["action.end_effector_rotation"], flat[8:11])
        np.testing.assert_array_equal(out["action.gripper_close"], flat[11:12])

    def test_parse_camera_names(self):
        assert _parse_camera_names("a, b ,c") == ["a", "b", "c"]
        assert _parse_camera_names(["x", "y"]) == ["x", "y"]
        with pytest.raises(ValueError):
            _parse_camera_names(" , ")

    def test_default_camera_name_mapping_is_positional(self):
        mapping = _default_camera_name_mapping(["cam_a", "cam_b", "cam_c"])
        assert mapping == {"cam_a": ["camera0"], "cam_b": ["camera1"], "cam_c": ["camera2"]}

    def test_resolve_single_and_comma_tasks_no_split(self):
        # Concrete task names never import robocasa and leave split untouched.
        assert _resolve_tasks("CloseFridge") == (["CloseFridge"], None)
        names, split = _resolve_tasks("CloseFridge, PickPlaceCoffee")
        assert names == ["CloseFridge", "PickPlaceCoffee"]
        assert split is None

    def test_resolve_empty_task_raises(self):
        with pytest.raises(ValueError, match="at least one RoboCasa task"):
            _resolve_tasks("  ,  ")

    def test_constants(self):
        assert ACTION_DIM == 12
        assert OBS_STATE_DIM == 16


def _mock_accelerator(num_processes: int, process_index: int) -> Mock:
    acc = Mock()
    acc.num_processes = num_processes
    acc.process_index = process_index
    return acc


class TestCreateRoboCasaEnvs:
    """``create_robocasa_envs`` return shape and per-rank task sharding.

    ``env_cls`` is mocked so the env factories are never invoked — no
    ``RoboCasaEnv`` is constructed and the sim is never imported.
    """

    def test_returns_one_vec_env_per_task(self):
        sentinel = Mock(name="vec_env")
        env_cls = Mock(return_value=sentinel)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=None):
            out = create_robocasa_envs(task="A,B", n_envs=3, env_cls=env_cls)
        assert set(out.keys()) == {"A", "B"}
        assert out["A"][0] is sentinel and out["B"][0] is sentinel
        # Each task built one vec env from exactly n_envs factory callables.
        assert env_cls.call_count == 2
        fns = env_cls.call_args[0][0]
        assert len(fns) == 3
        assert all(callable(f) for f in fns)

    @pytest.mark.parametrize(
        ("process_index", "expected"),
        [(0, {"A", "C"}), (1, {"B", "D"})],
    )
    def test_round_robin_task_sharding(self, process_index, expected):
        env_cls = Mock(return_value=Mock())
        acc = _mock_accelerator(num_processes=2, process_index=process_index)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=acc):
            out = create_robocasa_envs(task="A,B,C,D", n_envs=1, env_cls=env_cls)
        assert set(out.keys()) == expected

    def test_more_ranks_than_tasks_returns_empty(self):
        env_cls = Mock(return_value=Mock())
        acc = _mock_accelerator(num_processes=4, process_index=2)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=acc):
            out = create_robocasa_envs(task="A", n_envs=1, env_cls=env_cls)
        assert out == {}
        env_cls.assert_not_called()

    def test_rejects_bad_n_envs(self):
        # n_envs is validated before any accelerator call, so no patch needed.
        with pytest.raises(ValueError, match="positive int"):
            create_robocasa_envs(task="A", n_envs=0, env_cls=Mock())

    def test_rejects_non_callable_env_cls(self):
        with pytest.raises(ValueError, match="env_cls must be a callable"):
            create_robocasa_envs(task="A", n_envs=1, env_cls=None)


class TestMakeEnvsDispatch:
    """``make_envs`` routes RoboCasa configs to ``create_robocasa_envs``."""

    @pytest.fixture
    def mock_train_cfg(self):
        return Mock(spec=TrainPipelineConfig)

    def test_make_envs_dispatches_to_create_robocasa_envs(self, mock_train_cfg):
        expected = {"CloseFridge": {0: Mock()}}
        with patch("opentau.envs.robocasa.create_robocasa_envs", return_value=expected) as mock_create:
            result = make_envs(RoboCasaEnv(), mock_train_cfg, n_envs=2, use_async_envs=False)

        assert result is expected
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["task"] == "CloseFridge"
        assert kwargs["n_envs"] == 2
        assert kwargs["episode_length"] == 1000
        assert kwargs["obj_registries"] == ("lightwheel",)
        # SyncVectorEnv path: env_cls is the class itself.
        import gymnasium as gym

        assert kwargs["env_cls"] is gym.vector.SyncVectorEnv
