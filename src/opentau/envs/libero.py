# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

r"""This module provides an environment wrapper for LIBERO tasks."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

from opentau.utils.accelerate_utils import acc_print


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def _get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action() -> list[float | int]:
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


OBS_STATE_DIM = 8
ACTION_DIM = 7
AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    r"""Environment wrapper for LIBERO tasks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, list[str]] | None = None,
        num_steps_wait: int = 10,
        render_cam: str | None = None,
    ):
        r"""Initialize the LiberoEnv.
        Args:
            task_suite: The LIBERO task suite to use.
            task_id: The ID of the task within the suite.
            task_suite_name: The name of the task suite.
            camera_name: The name(s) of the camera(s) to use for observations. If a string, can be comma-separated.
            obs_type: The type of observation to return. Options are 'pixels' or 'pixels
            render_mode: The render mode for the environment.
            observation_width: The width of the observation images.
            observation_height: The height of the observation images.
            visualization_width: The width of the visualization window.
            visualization_height: The height of the visualization window.
            init_states: Whether to use predefined initial states for the tasks.
            episode_index: The index of the episode for selecting initial states.
            camera_name_mapping: Optional mapping from raw camera names to desired observation keys.
            num_steps_wait: Number of no-op steps to take after reset to stabilize the environment.
            render_cam: The camera name to use for rendering. If None, uses the first camera.
        """
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)
        self.render_cam = render_cam

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": ["camera0"],
                "robot0_eye_in_hand_image": ["camera1"],
            }
        self.camera_name_mapping = camera_name_mapping
        for cam in self.camera_name_mapping:
            assert not isinstance(self.camera_name_mapping[cam], str), (
                "camera_name_mapping values must be lists of strings; "
                f"got string {self.camera_name_mapping[cam]} for {cam} instead"
            )
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        # Load once and keep
        self._init_states = _get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

        images = {}
        for cam in self.camera_name:
            for mapped_cam in self.camera_name_mapping[cam]:
                images[mapped_cam] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=AGENT_POS_LOW,
                        high=AGENT_POS_HIGH,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self) -> np.ndarray:
        r"""Render the environment and return a numpy array representing the RGB camera.
        If `self.render_cam` is set, use that camera; otherwise, use the first camera."""
        raw_obs = self._env.env._get_observations()
        cams: dict[str, np.ndarray] = self._format_raw_obs(raw_obs)["pixels"]
        # if `self.render_cam` is not set, use the first camera
        render_cam = self.render_cam or next(iter(cams))
        return cams[render_cam]

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            image = image[::-1, ::-1]  # rotate 180 degrees
            for mapped_cam in self.camera_name_mapping[camera_name]:
                images[mapped_cam] = image.copy()
        state = np.concatenate(
            (
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            )
        )
        agent_pos = state
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}
        if self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images.copy(),
                "agent_pos": agent_pos,
            }
        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Reset the environment with the given seed.

        Args:
            seed: The seed to use for resetting the environment.
        Returns:
            observation: The initial observation after reset.
            info: Additional information about the reset.
        """
        super().reset(seed=seed)
        self._env.seed(seed)
        if self.init_states and self._init_states is not None:
            self._env.set_init_state(self._init_states[self._init_state_id])
        raw_obs = self._env.reset()

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        r"""Take a step in the environment with the given action.
        Args:
            action: The action to take.
        Returns:
            observation: The observation after taking the step.
            reward: The reward obtained from taking the step.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information about the step.
        """
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        if len(action) > ACTION_DIM:
            action = action[:ACTION_DIM]
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info["is_success"] = is_success

        observation = self._format_raw_obs(raw_obs)
        if done:
            self.reset()
            info.update(
                {
                    "task": self.task,
                    "task_id": self.task_id,
                    "done": done,
                    "is_success": is_success,
                }
            )
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        r"""Close the environment and release any resources."""
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_index=episode_index,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# main API entry point
def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: type[gym.vector.SyncVectorEnv] | type[gym.vector.AsyncVectorEnv] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (dict[str, list[int] | None]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter: dict[str, list[int] | None] | None = gym_kwargs.pop("task_ids", None)

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    acc_print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        # No tasks selected → return empty dict.
        # This happens when you have more accelerator processes than evaluation tasks.
        if len(task_ids_filter) == 0:
            acc_print("Empty task_ids specified, returning empty dict.")
            return {}

        acc_print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter and task_ids_filter[suite_name])

        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
            )
            out[suite_name][tid] = env_cls(fns)
            acc_print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}
