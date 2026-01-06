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
r"""This module contains factory methods to create environments based on their configuration."""

import importlib
from functools import partial

import gymnasium as gym

from opentau.configs.train import TrainPipelineConfig
from opentau.envs.configs import EnvConfig, LiberoEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    r"""Factory method to create an environment config based on the env_type.
    Right now, only 'libero' is supported.
    """
    if env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Env type '{env_type}' is not available.")


def make_envs(
    cfg: EnvConfig, train_cfg: TrainPipelineConfig, n_envs: int = 1, use_async_envs: bool = False
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Makes a nested collection of gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            A mapping from suite name to indexed vectorized environments.
            - For multi-task benchmarks (e.g., LIBERO): one entry per suite, and one vec env per task_id.
            - For single-task environments: a single suite entry (cfg.type) with task_id=0."""

    if n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    # "spawn" is more robust (and, for libero on oracle, the only option) than "fork".
    # Caveat is that the entry point must be protected by `if __name__ == "__main__":`.
    env_cls = (
        partial(gym.vector.AsyncVectorEnv, context="spawn") if use_async_envs else gym.vector.SyncVectorEnv
    )

    # Note: The official LeRobot repo makes a special case for Libero envs here.
    #   cf. https://github.com/huggingface/lerobot/commit/25384727812de60ff6e7a5e705cc016ec5def552
    if isinstance(cfg, LiberoEnv):
        from opentau.envs.libero import create_libero_envs

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    try:
        importlib.import_module(cfg.import_name)
    except ModuleNotFoundError as e:
        print(f"{cfg.import_name} is not installed. Please install it with `uv sync --all-extras'`")
        raise e

    def _make_one():
        return gym.make(
            cfg.make_id, disable_env_checker=cfg.disable_env_checker, **cfg.gym_kwargs, train_cfg=train_cfg
        )

    env = env_cls([_make_one] * n_envs)  # safe to repeat the same callable object

    return {
        cfg.type: {
            0: env,
        }
    }
