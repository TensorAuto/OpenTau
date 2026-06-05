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

import glob
import importlib
import json
import os
import tempfile
from functools import partial

import gymnasium as gym

from opentau.configs.train import TrainPipelineConfig
from opentau.envs.configs import EnvConfig, LiberoEnv, RoboCasaEnv
from opentau.utils.accelerate_utils import get_proc_accelerator


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    r"""Factory method to create an environment config based on the env_type.
    Supports 'libero' and 'robocasa'.
    """
    if env_type == "libero":
        return LiberoEnv(**kwargs)
    elif env_type == "robocasa":
        return RoboCasaEnv(**kwargs)
    else:
        raise ValueError(f"Env type '{env_type}' is not available.")


def _ensure_nvidia_egl_icd() -> str | None:
    r"""Register the nvidia EGL ICD when a container ships only Mesa's, so GPU rendering works.

    Some container images (e.g. the training enroot image) inject
    ``libEGL_nvidia.so`` via ``NVIDIA_DRIVER_CAPABILITIES`` *graphics* but ship only
    Mesa's glvnd ICD descriptor (``egl_vendor.d/50_mesa.json``) and no nvidia one.
    glvnd then loads Mesa, whose GPU path needs ``/dev/dri`` render nodes the
    container typically does not expose — so ``eglQueryDevicesEXT`` enumerates
    unusable Mesa devices and robosuite's per-device EGL context (it indexes that
    list by ``MUJOCO_EGL_DEVICE_ID``) fails to initialize a headless display, while
    MuJoCo's own backend silently falls back to the llvmpipe *software* device.

    When ``MUJOCO_GL=egl`` and no nvidia ICD is registered, write a minimal glvnd
    descriptor pointing at the injected ``libEGL_nvidia.so`` and point
    ``__EGL_VENDOR_LIBRARY_FILENAMES`` at it, so glvnd uses only nvidia EGL and
    ``eglQueryDevicesEXT`` returns the GPUs. The vec env's ``AsyncVectorEnv`` spawn
    workers inherit ``os.environ``, so setting it here — before the env is built —
    reaches the worker that creates the EGL context (paired with
    ``_pin_egl_render_device``, which selects which GPU that worker uses).

    No-op unless ``MUJOCO_GL=egl``; if ``__EGL_VENDOR_LIBRARY_*`` is already set, an
    nvidia ICD descriptor already exists, or no ``libEGL_nvidia.so`` is present (so
    there is nothing to point at), the environment is left untouched.

    Returns the descriptor path it wrote, or ``None`` if it left the environment alone.
    """
    if os.environ.get("MUJOCO_GL") != "egl":
        return None
    if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") or os.environ.get("__EGL_VENDOR_LIBRARY_DIRS"):
        return None
    # An nvidia ICD already registered with glvnd -> let it be.
    for vendor_dir in ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"):
        if glob.glob(os.path.join(vendor_dir, "*nvidia*.json")):
            return None
    # Prefer the version-stable soname symlink (``libEGL_nvidia.so.0``) over a
    # specific versioned file, so the choice doesn't depend on lexicographic-vs-
    # numeric ordering of multiple injected libs.
    libs = (
        glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0")
        or glob.glob("/usr/lib/*/libEGL_nvidia.so.0")
        or sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.*"))
        or sorted(glob.glob("/usr/lib/*/libEGL_nvidia.so.*"))
    )
    if not libs:
        return None
    # Write the descriptor into a fresh private (0700) tmp dir: mkdtemp returns a
    # unique, unpredictable path owned by this process, so there is no cross-user
    # collision and no fixed path a co-tenant could pre-squat to break the write.
    # Each rank makes its own; the spawn workers inherit __EGL_VENDOR_LIBRARY_FILENAMES.
    icd_dir = tempfile.mkdtemp(prefix="opentau_egl_")
    icd_path = os.path.join(icd_dir, "10_nvidia.json")
    with open(icd_path, "w") as f:
        json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": libs[0]}}, f)
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = icd_path
    return icd_path


def _pin_egl_render_device() -> str | None:
    r"""Pin this rank's MuJoCo EGL renderer to its own GPU, for multi-GPU sim eval.

    MuJoCo's EGL backend (``mujoco.egl.create_initialized_egl_device_display``)
    selects the render GPU from ``MUJOCO_EGL_DEVICE_ID`` and falls back to EGL
    device 0 when it is unset — and robosuite forwards a ``device_id`` that
    ``mujoco.egl`` ignores. So under multi-rank eval *every* rank would render on
    GPU 0, overloading it (it also holds rank 0's resident training state, so it
    typically OOMs) while the other GPUs render nothing. Setting it to the rank's
    own GPU makes each rank render on its own device.

    The value is the rank's entry in ``CUDA_VISIBLE_DEVICES`` (so a masked or
    reordered subset like ``"4,5,6,7"`` still maps to the right physical GPU and
    satisfies robosuite's ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` assert),
    falling back to the local process index when ``CUDA_VISIBLE_DEVICES`` is unset.

    No-op unless ``MUJOCO_GL=egl``; an explicit ``MUJOCO_EGL_DEVICE_ID`` is left
    untouched, and so is the single-process / no-accelerator case (device 0 is
    correct there). The vec env's ``AsyncVectorEnv`` spawn workers inherit
    ``os.environ``, so setting it here — before the env is built — reaches the
    worker that actually creates the EGL context.

    Returns the value it set, or ``None`` if it left the environment untouched.
    """
    if os.environ.get("MUJOCO_GL") != "egl" or "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return None
    acc = get_proc_accelerator()
    if acc is None:
        return None
    local_index = acc.local_process_index
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    device_id = visible[local_index] if local_index < len(visible) else str(local_index)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device_id
    return device_id


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

    # Make GPU EGL rendering work before the vec env's spawn workers (which inherit
    # os.environ) create their render context. Both are no-ops unless MUJOCO_GL=egl:
    # (1) register the nvidia EGL ICD if the container ships only Mesa's (else robosuite
    # cannot init a headless GPU display); (2) pin each rank's renderer to its own GPU
    # (else every rank renders on GPU 0 and OOMs it).
    _ensure_nvidia_egl_icd()
    _pin_egl_render_device()

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

    # RoboCasa, like LIBERO, is multi-task: build one vec env per task so eval
    # reports per-task success and per-task grid videos, and so tasks shard
    # disjointly across accelerator ranks (handled inside create_robocasa_envs).
    if isinstance(cfg, RoboCasaEnv):
        from opentau.envs.robocasa import create_robocasa_envs

        return create_robocasa_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
            episode_length=cfg.episode_length,
            obj_registries=tuple(cfg.obj_registries),
            assets_root=cfg.assets_root,
            auto_download_assets=cfg.auto_download_assets,
        )

    try:
        importlib.import_module(cfg.import_name)
    except ModuleNotFoundError as e:
        print(f"{cfg.import_name} is not installed. Please install it with `uv sync --all-extras`")
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
