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

r"""Environment wrapper for RoboCasa kitchen tasks.

This module mirrors :mod:`opentau.envs.libero`: it adapts a single RoboCasa task into a
``gymnasium.Env`` so it can be vectorised by ``gym.vector.SyncVectorEnv`` /
``gym.vector.AsyncVectorEnv`` and consumed by the standard OpenTau eval loop.

Cameras are exposed under the ``pixels/{camera_name}`` namespace and the proprio state is
exposed under ``agent_pos`` — the same convention LIBERO uses, so
``preprocess_observation`` / ``features_map`` translate them into the standard OpenTau
observation dict.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from opentau.envs.configs import ROBOCASA_DEFAULT_CAMERA_NAMES
from opentau.utils.accelerate_utils import acc_print

# Proprio observation keys exposed by RoboCasa's PandaOmron-style envs. Concatenated in
# order to form the policy state vector. Kept here (rather than imported from
# ``robocasa.scripts.client``) so the wrapper works against an unmodified upstream
# robocasa install — that ``client.py`` is not part of the published package.
ROBOCASA_DEFAULT_PROPRIO_KEYS: tuple[str, ...] = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)


def _build_proprio_vector(obs: dict, keys: Sequence[str] = ROBOCASA_DEFAULT_PROPRIO_KEYS) -> np.ndarray:
    """Concatenate low-dim robot state for policy input.

    Matches the layout produced by ``robocasa.scripts.client.build_proprio_vector`` so
    checkpoints trained against client.py-recorded data see the same proprio convention.
    """
    parts: list[np.ndarray] = []
    for k in keys:
        if k not in obs:
            raise KeyError(
                f"RoboCasa observation missing proprio key {k!r}. "
                f"Available keys (sample): "
                f"{[x for x in obs if not str(x).endswith('_image')][:20]}..."
            )
        parts.append(np.asarray(obs[k], dtype=np.float64).ravel())
    return np.concatenate(parts, axis=0)


def _get_task_prompt(env: Any) -> str:
    """Return the natural-language instruction for the current RoboCasa episode."""
    try:
        meta = env.get_ep_meta()
    except Exception:
        return ""
    if not meta:
        return ""
    lang = meta.get("lang", "") if isinstance(meta, dict) else ""
    if lang is None:
        return ""
    if isinstance(lang, (list, tuple)):
        return " ".join(str(x) for x in lang)
    return str(lang)


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize ``camera_name`` into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
# Upper bound for the proprio vector length returned by RoboCasa's ``build_proprio_vector``.
# Concrete length is detected on first reset() and the box is reshaped; this constant only
# controls how big the declared observation_space is up-front.
DEFAULT_PROPRIO_DIM = 32


class RoboCasaEnv(gym.Env):
    r"""Wrap a single RoboCasa task class as a ``gymnasium.Env``.

    One sub-env corresponds to one rollout: it owns its own MuJoCo simulator instance and
    its own deterministic seed (``seed_base + episode_index``). Vectorisation across
    rollouts and across task names is handled by :func:`create_robocasa_envs`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        *,
        camera_names: Sequence[str] = ROBOCASA_DEFAULT_CAMERA_NAMES,
        camera_height: int = 256,
        camera_width: int = 256,
        num_steps_wait: int = 10,
        episode_length: int = 1500,
        seed_base: int = 0,
        split: str | None = "all",
        render_cam: str | None = None,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        # Import lazily so that machines without robocasa installed (e.g. CPU dev) can
        # still import opentau.envs.configs to inspect schemas without crashing.
        from robocasa.utils.env_utils import create_env

        self._create_env = create_env
        self._build_proprio_vector = _build_proprio_vector
        self._get_task_prompt = _get_task_prompt

        self.task_name = task_name
        # ``task`` mirrors ``task_name`` for compatibility with the LIBERO-style env
        # attribute contract that ``check_env_attributes_and_types`` and several policy
        # heads (which infer the task identifier per rollout) look for.
        self.task = task_name
        self.episode_index = episode_index
        self.camera_names = _parse_camera_names(camera_names)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.num_steps_wait = int(num_steps_wait)
        self.episode_length = int(episode_length)
        self.seed_base = int(seed_base)
        self.split = split
        # ``render_cam`` indexes the positional ``camera{i}`` keys (matching what
        # ``_format_obs`` emits), not the raw RoboCasa cam name. Accept either form:
        # if the caller passed a raw name, translate it to the positional key.
        if render_cam is None:
            self.render_cam = "camera0"
        elif render_cam in self.camera_names:
            self.render_cam = f"camera{self.camera_names.index(render_cam)}"
        else:
            self.render_cam = render_cam
        self.render_mode = render_mode

        # Built lazily on first reset() so that any AsyncVectorEnv worker process can
        # fork before MuJoCo is initialised (RoboCasa's renderer must live in the worker).
        self._env: Any | None = None
        self._task_description: str = ""
        self._action_dim: int | None = None
        self._proprio_dim: int = DEFAULT_PROPRIO_DIM

        self._max_episode_steps = self.episode_length

        # Match the positional ``camera{i}`` keys emitted by ``_format_obs`` so that
        # gymnasium's AsyncVectorEnv observation validation sees consistent keys.
        images: dict[str, spaces.Box] = {
            f"camera{idx}": spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            )
            for idx in range(len(self.camera_names))
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(images),
                "agent_pos": spaces.Box(
                    low=AGENT_POS_LOW,
                    high=AGENT_POS_HIGH,
                    shape=(self._proprio_dim,),
                    dtype=np.float64,
                ),
            }
        )
        # Action space is sized on first reset() once env.action_dim is known. Provide a
        # permissive placeholder so vector envs can inspect a shape before reset.
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(16,), dtype=np.float32)

    def _build_env(self, seed: int) -> Any:
        # ``create_env`` derives ``has_offscreen_renderer`` / ``use_camera_obs`` itself
        # from ``render_onscreen`` (see robocasa.utils.env_utils.create_env). Passing
        # them again duplicates the kwarg and raises
        # ``TypeError: dict() got multiple values for keyword argument 'has_offscreen_renderer'``.
        return self._create_env(
            self.task_name,
            split=self.split,
            seed=seed,
            render_onscreen=False,
            camera_names=list(self.camera_names),
            camera_widths=self.camera_width,
            camera_heights=self.camera_height,
        )

    def _flip(self, image: np.ndarray) -> np.ndarray:
        # RoboCasa returns vertically-flipped images from mujoco; flip top-down to match
        # the rendering convention used by client.py (``flip_image_obs``).
        return image[::-1, :, :].copy()

    def _format_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # Emit positional ``camera{idx}`` keys (matching LiberoEnv's convention) so
        # downstream ``preprocess_observation`` + ``features_map`` see the same keys the
        # policy was trained against, regardless of what robocasa internally calls the
        # cameras. ``self.camera_names[i]`` -> ``camera{i}``.
        images: dict[str, np.ndarray] = {}
        for idx, cam in enumerate(self.camera_names):
            key = f"{cam}_image"
            if key not in raw_obs:
                raise KeyError(
                    f"Camera image key {key!r} missing from RoboCasa observation; "
                    f"available keys: {sorted(raw_obs.keys())}"
                )
            images[f"camera{idx}"] = self._flip(raw_obs[key])
        proprio = np.asarray(self._build_proprio_vector(raw_obs), dtype=np.float64).ravel()
        if proprio.shape[0] != self._proprio_dim:
            # Re-fit the observation_space the first time we learn the real proprio dim.
            self._proprio_dim = int(proprio.shape[0])
            self.observation_space.spaces["agent_pos"] = spaces.Box(
                low=AGENT_POS_LOW,
                high=AGENT_POS_HIGH,
                shape=(self._proprio_dim,),
                dtype=np.float64,
            )
        return {"pixels": images, "agent_pos": proprio}

    def _refit_action_space(self) -> None:
        if self._env is None:
            return
        ad = getattr(self._env, "action_dim", None)
        if ad is None:
            return
        self._action_dim = int(ad)
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(self._action_dim,), dtype=np.float32
        )

    @property
    def task_description(self) -> str:
        return self._task_description

    def has_wrapper_attr(self, name: str) -> bool:
        """Gymnasium 1.0+ API shim used by ``check_env_attributes_and_types``.

        gymnasium ``0.29.x`` (what we pin) hasn't added this on ``gym.Env`` yet, so the
        async worker raises ``AttributeError`` when the eval loop asks each rollout env
        whether it carries a ``task`` / ``task_description``. Forwarding to plain
        ``hasattr`` gives the same answer the future API will once we upgrade.
        """
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gymnasium 1.0+ API shim — paired with :meth:`has_wrapper_attr`.

        Used by ``opentau.envs.utils.add_envs_task`` to look up ``task`` /
        ``task_description`` per rollout env. Pre-1.0 gymnasium doesn't expose this on
        ``gym.Env``; fall through to ``getattr`` here.
        """
        return getattr(self, name)

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        # Determine seed: prefer caller-provided seed; fall back to deterministic
        # seed_base + episode_index so that re-running the same vec env reproduces the
        # same RoboCasa layout each batch.
        if seed is None:
            seed = self.seed_base + self.episode_index

        if self._env is None:
            self._env = self._build_env(seed)
        else:
            # RoboSuite envs don't expose a re-seed API after construction, but they do
            # accept a fresh seed via reset() in newer versions; for portability, rebuild.
            with contextlib.suppress(Exception):
                self._env.close()
            self._env = self._build_env(seed)

        raw_obs = self._env.reset()
        self._refit_action_space()
        try:
            self._task_description = str(self._get_task_prompt(self._env) or "")
        except Exception:
            self._task_description = ""

        # Settle the simulator with a zero action (RoboCasa's dummy action) for a few
        # frames after reset so that objects come to rest before the policy first acts.
        if self._action_dim and self.num_steps_wait > 0:
            zero = np.zeros(self._action_dim, dtype=np.float64)
            for _ in range(self.num_steps_wait):
                raw_obs, _r, _d, _i = self._env.step(zero)

        obs = self._format_obs(raw_obs)
        info = {"is_success": False, "task": self.task_name, "task_description": self._task_description}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("RoboCasaEnv.step called before reset().")
        action = np.asarray(action, dtype=np.float64).ravel()
        if self._action_dim is not None and action.shape[0] != self._action_dim:
            if action.shape[0] > self._action_dim:
                action = action[: self._action_dim]
            else:
                padded = np.zeros(self._action_dim, dtype=np.float64)
                padded[: action.shape[0]] = action
                action = padded

        raw_obs, reward, done, info = self._env.step(action)
        is_success = bool(self._env._check_success())  # noqa: SLF001 — RoboCasa API
        terminated = bool(done) or is_success
        truncated = False
        info = dict(info or {})
        info.update(
            {
                "is_success": is_success,
                "task": self.task_name,
                "task_description": self._task_description,
            }
        )
        obs = self._format_obs(raw_obs)
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("RoboCasaEnv.render called before reset().")
        raw_obs = self._env._get_observations()  # noqa: SLF001 — RoboCasa API
        obs = self._format_obs(raw_obs)
        return obs["pixels"][self.render_cam]

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], RoboCasaEnv]]:
    """Build ``n_envs`` factory callables for a single RoboCasa task."""

    def _make_env(episode_index: int, **kwargs) -> RoboCasaEnv:
        local_kwargs = dict(kwargs)
        local_kwargs.pop("tasks", None)
        local_kwargs.pop("episode_length", None)
        return RoboCasaEnv(
            task_name=task_name,
            episode_index=episode_index,
            episode_length=int(gym_kwargs.get("episode_length", 1500)),
            **local_kwargs,
        )

    base_kwargs = {
        "camera_names": gym_kwargs.get("camera_names", ROBOCASA_DEFAULT_CAMERA_NAMES),
        "camera_height": gym_kwargs.get("camera_height", 256),
        "camera_width": gym_kwargs.get("camera_width", 256),
        "num_steps_wait": gym_kwargs.get("num_steps_wait", 10),
        "seed_base": gym_kwargs.get("seed_base", 0),
        "split": gym_kwargs.get("split", "all"),
        "render_cam": gym_kwargs.get("render_cam"),
    }

    fns: list[Callable[[], RoboCasaEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **base_kwargs))
    return fns


def create_robocasa_envs(
    task: str | Iterable[str],
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: type[gym.vector.SyncVectorEnv] | type[gym.vector.AsyncVectorEnv] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create vectorised RoboCasa environments shaped like LIBERO's output.

    Returns:
        ``dict[task_name][0] -> vec_env`` — one vec env per RoboCasa task class name.
        ``task_id=0`` is kept as a placeholder so the downstream eval loop (which keys
        videos and metrics on ``(task_group, task_id)``) works without modification.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    # The factory call passes us the per-rank task list under "tasks"; fall back to the
    # caller's `task` for direct invocations (e.g. from a standalone script).
    assigned_tasks = gym_kwargs.pop("tasks", None)
    if assigned_tasks is None:
        if isinstance(task, str):
            assigned_tasks = [s.strip() for s in task.split(",") if s.strip()]
        else:
            assigned_tasks = [str(t).strip() for t in task if str(t).strip()]
    if not assigned_tasks:
        acc_print("[RoboCasa] No tasks assigned to this rank; returning empty dict.")
        return {}

    acc_print(f"Creating RoboCasa envs | tasks={assigned_tasks} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_name in assigned_tasks:
        fns = _make_env_fns(task_name=task_name, n_envs=n_envs, gym_kwargs=gym_kwargs)
        out[task_name][0] = env_cls(fns)
        acc_print(f"Built vec env | task={task_name} | n_envs={n_envs}")

    return {t: dict(m) for t, m in out.items()}
