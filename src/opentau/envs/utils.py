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

r"""This module contains utility functions for environments."""

import warnings
from collections.abc import Mapping, Sequence
from functools import singledispatch
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.utils.accelerate_utils import get_proc_accelerator
from opentau.utils.utils import auto_torch_device


def preprocess_observation(np_observations: dict, cfg: TrainPipelineConfig) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to OpenTau format observation.
    Args:
        np_observations: Dictionary of observation batches from a Gym vector environment.
        cfg: Training configuration that contains max_state_dim, num_cams, resolution, etc.
    Returns:
        Dictionary of observation batches with keys renamed to OpenTau format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    img_transform = Compose([ToTensor(), Resize(cfg.resolution, antialias=True)])

    if "pixels" in np_observations:
        assert isinstance(np_observations["pixels"], dict)
        imgs: dict[str, np.ndarray] = np_observations["pixels"]

        for imgkey, img in imgs.items():
            return_observations[imgkey] = torch.stack([img_transform(img) for img in img])

    if "environment_state" in np_observations:
        env_state = torch.from_numpy(np_observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations["environment_state"] = env_state

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = torch.from_numpy(np_observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)

    # Preprocess so that agent_pos has the same dimension as max_state_dim
    agent_pos = BaseDataset.pad_vector(agent_pos, cfg.max_state_dim)
    return_observations["state"] = agent_pos

    batch_size = agent_pos.shape[0]
    # Mirror training-time ``BaseDataset._standardize_images``: when the env exposes
    # fewer cameras than the policy was trained with (e.g. LIBERO emits only
    # ``camera0``/``camera1`` but the policy was trained with ``num_cams=4``),
    # zero-fill the missing ``cameraN`` slots and flag them in ``img_is_pad``.
    # Without this the eval batch has fewer image keys than any training batch,
    # so the VLM sees a strictly shorter visual prefix than at training and
    # ``Normalize`` prints a missing-key warning for every absent slot every step.
    if cfg.num_cams > 0:
        img_is_pad = torch.zeros((batch_size, cfg.num_cams), dtype=torch.bool)
        for i in range(cfg.num_cams):
            key = f"camera{i}"
            if key not in return_observations:
                return_observations[key] = torch.zeros((batch_size, 3, *cfg.resolution), dtype=torch.float32)
                img_is_pad[:, i] = True
        return_observations["img_is_pad"] = img_is_pad

    # convert all floating point tensors to bfloat16 to save memory
    acc = get_proc_accelerator()
    device = auto_torch_device() if acc is None else acc.device

    for k, v in return_observations.items():
        if isinstance(v, Tensor):
            dtype = torch.bfloat16 if v.dtype.is_floating_point else v.dtype
            return_observations[k] = v.to(device=device, dtype=dtype)

    return return_observations


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    r"""Checks if all environments in a vectorized environment are of the same type.

    Args:
        env: A vectorized Gym environment (SyncVectorEnv or AsyncVectorEnv).
    Returns:
        True if all environments are of the same type, False otherwise.
    """
    if not isinstance(env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
        raise ValueError("Only gym.vector.SyncVectorEnv and gym.vector.AsyncVectorEnv are supported for now.")

    types = env.call("get_wrapper_attr", "__class__")
    first_type = types[0]
    return all(t == first_type for t in types)


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    r"""Checks if all environments in a vectorized environment have 'task_description' or 'task' attributes.
    A warning will be raised if any environment is missing these attributes.

    Args:
        env: A vectorized Gym environment (SyncVectorEnv or AsyncVectorEnv).
    Raises:
        ValueError: If the environment is not a SyncVectorEnv or AsyncVectorEnv.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # Apply filter only in this function

        if not isinstance(env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
            raise ValueError(
                "Only gym.vector.SyncVectorEnv and gym.vector.AsyncVectorEnv are supported for now."
            )

        task_desc_set = env.call("has_wrapper_attr", "task_description")
        task_set = env.call("has_wrapper_attr", "task")
        if not all(td or t for td, t in zip(task_desc_set, task_set, strict=True)):
            warnings.warn(
                "At least 1 environment does not have 'task_description' or 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "The environments have different types. Make sure you infer the right task from each environment.",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    r"""Adds task feature to the observation dict with respect to the first environment attribute.

    Args:
        env: A vectorized Gym environment (SyncVectorEnv or AsyncVectorEnv).
        observation: A dictionary of observations from the vectorized environment, which will be modified in place.
    Returns:
        The updated observation dictionary with the 'prompt' key added.
    """
    if not isinstance(env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
        raise ValueError("Only gym.vector.SyncVectorEnv and gym.vector.AsyncVectorEnv are supported for now.")

    task_result = [""] * env.num_envs
    for task_key in ["task_description", "task"]:
        tasks = env.call("get_wrapper_attr", task_key)
        if len(tasks) != env.num_envs:
            raise ValueError(f"Environment returned {len(tasks)} task(s); expected {env.num_envs}.")
        for i, t in enumerate(tasks):
            if task_result[i] == "" and isinstance(t, str) and t != "":
                task_result[i] = t

    observation["prompt"] = task_result
    return observation


def add_eval_metadata(observation: dict[str, Any], cfg: TrainPipelineConfig) -> dict[str, Any]:
    r"""Inject pi07 inference-time metadata from ``cfg.env.metadata``.

    Numeric fields (``speed`` / ``quality`` / ``mistake``) are broadcast as
    ``(num_envs,)`` tensors with a parallel ``{key}_is_pad=False`` flag,
    using the same dtypes the dataset emits at training time
    (``torch.long`` for ``speed`` / ``quality``, ``torch.bool`` for
    ``mistake``). String fields (``robot_type`` / ``control_mode``) are
    broadcast as ``list[str]`` of length ``num_envs``. Fields set to
    ``None`` on the config are skipped, so the corresponding batch key
    stays absent and the policy's ``prepare_metadata`` pad default kicks
    in.

    ``fps`` is the env's stepping frequency (:attr:`EnvConfig.fps`,
    e.g. ``20`` for LIBERO) and is broadcast as a ``torch.long`` tensor
    with ``fps_is_pad=False`` whenever ``meta.emit_fps`` is ``True``
    (the default). Set ``emit_fps=False`` on ``EnvMetadataConfig`` to
    skip it — useful when resuming a checkpoint trained without fps
    conditioning.

    ``cfg.env`` is guaranteed non-``None`` here — ``eval_policy``
    dereferences ``cfg.env.type`` before the rollout loop starts.

    Args:
        observation: Observation dict produced by ``preprocess_observation``
            (must contain ``state`` to source the batch size and device).
        cfg: Training/eval pipeline config; reads ``cfg.env.metadata`` and
            ``cfg.env.fps``.

    Returns:
        The observation dict, mutated in place.
    """
    meta = cfg.env.metadata
    state = observation["state"]
    batch_size = state.shape[0]
    device = state.device

    for key, dtype in (
        ("speed", torch.long),
        ("quality", torch.long),
        ("mistake", torch.bool),
    ):
        val = getattr(meta, key)
        if val is not None:
            observation[key] = torch.tensor([val] * batch_size, dtype=dtype, device=device)
            observation[f"{key}_is_pad"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for key in ("robot_type", "control_mode"):
        val = getattr(meta, key)
        if val is not None:
            observation[key] = [val] * batch_size

    if meta.emit_fps:
        observation["fps"] = torch.tensor([cfg.env.fps] * batch_size, dtype=torch.long, device=device)
        observation["fps_is_pad"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

    return observation


def _close_single_env(env: Any) -> None:
    try:
        env.close()
    except Exception as exc:
        print(f"Exception while closing env {env}: {exc}")


@singledispatch
def close_envs(obj: Any) -> None:
    """Close a single environment, a list of environments, or a dictionary of environments."""
    raise NotImplementedError(f"close_envs not implemented for type {type(obj).__name__}")


@close_envs.register
def _(env: Mapping) -> None:
    for v in env.values():
        if isinstance(v, Mapping):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(envs: Sequence) -> None:
    if isinstance(envs, (str, bytes)):
        return
    for v in envs:
        if isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(env: gym.Env) -> None:
    _close_single_env(env)


@close_envs.register
def _(env: None) -> None:
    pass
