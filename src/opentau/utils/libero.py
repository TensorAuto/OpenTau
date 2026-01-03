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

"""Utilities for working with the LIBERO robotics environment.

This module provides functions for converting LIBERO observations to PyTorch tensors,
summarizing LIBERO evaluation results, and recording observations from LIBERO environments.
"""

import logging
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from robosuite.utils.transform_utils import quat2axisangle


def rotate_numpy_image(image: np.ndarray) -> np.ndarray:
    """Rotate and normalize a numpy image array.

    Args:
        image: Input image array in HWC format with values in [0, 255].

    Returns:
        Rotated and normalized image array in CHW format with values in [0, 1].
    """
    image = image.astype(float) / 255.0
    image = np.rot90(image, 2)
    return rearrange(image, "H W C -> C H W")


def _libero2np(obs: dict[str, np.ndarray], cfg) -> dict[str, str | np.ndarray]:
    """Convert LIBERO observation dictionary to numpy format.

    Args:
        obs: LIBERO observation dictionary containing robot state and images.
        cfg: Configuration object with task language, state dimensions, etc.

    Returns:
        Dictionary with converted observations in numpy format, including camera
        images, state, prompt, and padding flags.
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_angle = quat2axisangle(obs["robot0_eef_quat"])
    gripper_pos = obs["robot0_gripper_qpos"]

    state = np.hstack((eef_pos, eef_angle, gripper_pos))

    agent_view = rotate_numpy_image(obs["agentview_image"])
    wrist_view = rotate_numpy_image(obs["robot0_eye_in_hand_image"])

    return {
        "camera0": agent_view,
        "camera1": wrist_view,
        "prompt": cfg.libero.task.language,
        "state": np.pad(state, (0, cfg.max_state_dim - len(state))),
        "img_is_pad": np.zeros(cfg.num_cams, dtype=bool),
        "action_is_pad": np.zeros(cfg.action_chunk, dtype=bool),
    }


def _np2torch(
    np_input: dict[str, str | np.ndarray], device: str, dtype: torch.dtype
) -> dict[str, str | torch.Tensor]:
    """Convert numpy arrays in dictionary to PyTorch tensors.

    Args:
        np_input: Dictionary containing numpy arrays and strings.
        device: Target device for tensors (e.g., 'cuda', 'cpu').
        dtype: Target dtype for floating point tensors.

    Returns:
        Dictionary with numpy arrays converted to PyTorch tensors on the
        specified device. String values are preserved as-is.

    Raises:
        TypeError: If a value type is not supported (not str or np.ndarray).
    """
    torch_input = {}
    for k, v in np_input.items():
        if isinstance(v, str):
            torch_input[k] = v
        elif isinstance(v, np.ndarray):
            # .copy() ensures the array is contiguous for PyTorch to use it
            tensor = torch.tensor(v.copy())
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype=dtype)
            torch_input[k] = tensor.to(device)
        else:
            raise TypeError(f"Unsupported type {type(v)} for key {k}.")
    return torch_input


def libero2torch(
    obs: dict[str, np.ndarray], cfg, device: str, dtype: torch.dtype
) -> dict[str, str | torch.Tensor]:
    """Convert LIBERO observation to PyTorch tensors.

    Args:
        obs: LIBERO observation dictionary containing robot state and images.
        cfg: Configuration object with task language, state dimensions, etc.
        device: Target device for tensors (e.g., 'cuda', 'cpu').
        dtype: Target dtype for floating point tensors.

    Returns:
        Dictionary with observations converted to PyTorch tensors on the
        specified device, including camera images, state, prompt, and padding flags.
    """
    np_input = _libero2np(obs, cfg)
    torch_input = _np2torch(np_input, device, dtype)
    return torch_input


def summarize_libero_results(results: list[int]) -> dict:
    """Summarize LIBERO evaluation results.

    Args:
        results: List of integer results where:
            - Positive values indicate success (number of steps taken).
            - -1 indicates failure.
            - -2 indicates crash.

    Returns:
        Dictionary containing summary statistics including success/failure/crash
        rates, counts, indices, and average steps taken for successful episodes.
    """
    if not results:
        return {"message": "No results to summarize."}

    success_indices = [i for i, r in enumerate(results) if r >= 0]
    failure_indices = [i for i, r in enumerate(results) if r == -1]
    crashed_indices = [i for i, r in enumerate(results) if r == -2]

    success_rate = len(success_indices) / len(results)
    failure_rate = len(failure_indices) / len(results)
    crashed_rate = len(crashed_indices) / len(results)

    avg_steps_taken = float(np.mean([r for r in results if r >= 0])) if success_indices else None

    return {
        "total_simulations": len(results),
        "success_indices": success_indices,
        "failure_indices": failure_indices,
        "crashed_indices": crashed_indices,
        "success_count": len(success_indices),
        "failure_count": len(failure_indices),
        "crashed_count": len(crashed_indices),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "crashed_rate": crashed_rate,
        "steps_taken": results,
        "avg_steps_taken_until_success": avg_steps_taken,
    }


class LiberoObservationRecorder:
    """Context manager for recording LIBERO observations to video files.

    This class is not multi-processing safe. Each process should use a different
    (folder, camera_name) pair.

    Args:
        folder: Directory path where video files will be saved. If None, recording
            is disabled.
        camera_names: List of camera names to record. If None, no cameras are recorded.
        fps: Frames per second for the output videos. Defaults to 10.
        extension: Video file extension. Defaults to "mp4".
    """

    def __init__(self, folder, camera_names=None, fps=10, extension="mp4"):
        if folder is None:
            logging.debug("No folder specified for video recording. Skipping.")
            self.writers = []
            self.camera_names = []
            return

        self.camera_names = camera_names or []
        folder = Path(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)
        video_files = [folder / f"{cam}.{extension}" for cam in self.camera_names]
        logging.debug("Creating video files: %s", video_files)
        self.writers = [imageio.get_writer(vf, fps=fps) for vf in video_files]

    def __enter__(self):
        return self

    def record(self, obs):
        """Record a single observation frame.

        Args:
            obs: Observation dictionary containing camera images keyed by camera name.
        """
        for writer, camera in zip(self.writers, self.camera_names, strict=True):
            writer.append_data(np.rot90(obs[camera], k=2))

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.debug("Closing video writers.")
        for writer in self.writers:
            writer.close()
        logging.debug("Video writers closed.")
        return False
