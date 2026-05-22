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

r"""The LIBERO sim control frequency must be configurable via ``env.fps``.

Before this was wired, ``OffScreenRenderEnv`` was built without ``control_freq``,
so the sim always ran at robosuite's hardcoded 20 Hz regardless of the config.
These tests pin the wiring: config ``fps`` -> ``gym_kwargs["control_freq"]`` ->
``OffScreenRenderEnv(control_freq=...)``.
"""

from unittest.mock import Mock, patch

import pytest

from opentau.envs.configs import LiberoEnv as LiberoEnvConfig
from opentau.envs.libero import LiberoEnv as LiberoGymEnv


def _make_task_suite() -> Mock:
    """A task suite whose ``get_task`` returns string-valued fields os.path.join can consume."""
    task = Mock()
    task.name = "demo_task"
    task.language = "do something"
    task.problem_folder = "problem_folder"
    task.bddl_file = "task.bddl"
    suite = Mock()
    suite.get_task.return_value = task
    return suite


def test_libero_config_fps_defaults_to_20():
    # 20 Hz is robosuite's native LIBERO rate and the de-facto rate before fps was wired.
    assert LiberoEnvConfig().fps == 20


def test_libero_gym_kwargs_carries_control_freq():
    cfg = LiberoEnvConfig(fps=10, task_ids=[0])
    # Force the no-accelerator branch so the property doesn't need the LIBERO benchmark.
    with patch("opentau.envs.configs.get_proc_accelerator", return_value=None):
        gym_kwargs = cfg.gym_kwargs
    assert gym_kwargs["control_freq"] == 10


@pytest.mark.parametrize("bad_fps", [0, -5])
def test_libero_config_rejects_nonpositive_fps(bad_fps):
    # fps now drives the sim control loop, so a non-positive value must fail loudly
    # at config time rather than passing through to robosuite.
    with pytest.raises(ValueError, match="must be positive"):
        LiberoEnvConfig(fps=bad_fps)


@patch("opentau.envs.libero.get_libero_path", return_value="/tmp/libero")
@patch("opentau.envs.libero.OffScreenRenderEnv")
def test_control_freq_reaches_robosuite(mock_offscreen, _mock_path):
    # A value distinct from both the default (20) and the dataset label (10) proves it threads through.
    LiberoGymEnv(
        task_suite=_make_task_suite(),
        task_id=0,
        task_suite_name="libero_10",
        init_states=False,
        control_freq=13,
    )
    assert mock_offscreen.call_args.kwargs["control_freq"] == 13


@patch("opentau.envs.libero.get_libero_path", return_value="/tmp/libero")
@patch("opentau.envs.libero.OffScreenRenderEnv")
def test_control_freq_defaults_to_20(mock_offscreen, _mock_path):
    LiberoGymEnv(
        task_suite=_make_task_suite(),
        task_id=0,
        task_suite_name="libero_10",
        init_states=False,
    )
    assert mock_offscreen.call_args.kwargs["control_freq"] == 20


@pytest.mark.gpu
def test_control_freq_reaches_real_robosuite_sim():
    """On a real LIBERO sim (needs GL + assets), the configured control_freq must reach robosuite.

    The mocked tests above can't catch robosuite silently dropping the kwarg; this one builds the
    actual ``OffScreenRenderEnv`` and reads back ``control_freq`` from the underlying robosuite env.
    """
    import gymnasium as gym

    from opentau.envs.libero import create_libero_envs

    envs = create_libero_envs(
        task="libero_10",
        n_envs=1,
        gym_kwargs={"task_ids": {"libero_10": [0]}, "control_freq": 13},
        env_cls=gym.vector.SyncVectorEnv,
        init_states=False,
    )
    vec = envs["libero_10"][0]
    try:
        assert vec.envs[0]._env.env.control_freq == 13
    finally:
        vec.close()
