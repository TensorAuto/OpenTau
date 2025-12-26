import logging
from pathlib import Path

import imageio
import numpy as np
import torch
from einops import rearrange
from robosuite.utils.transform_utils import quat2axisangle


def rotate_numpy_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(float) / 255.0
    image = np.rot90(image, 2)
    return rearrange(image, "H W C -> C H W")


def _libero2np(obs: dict[str, np.ndarray], cfg) -> dict[str, str | np.ndarray]:
    eef_pos = obs["robot0_eef_pos"]
    eef_angle = quat2axisangle(obs["robot0_eef_quat"])
    gripper_pos = obs["robot0_gripper_qpos"]

    state = np.hstack((eef_pos, eef_angle, gripper_pos))

    agent_view = rotate_numpy_image(obs["agentview_image"])
    wrist_view = rotate_numpy_image(obs["robot0_eye_in_hand_image"])

    return {
        "camera0": agent_view,
        "camera1": wrist_view,
        "local_camera0": agent_view,
        "local_camera1": wrist_view,
        "prompt": cfg.libero.task.language,
        "state": np.pad(state, (0, cfg.max_state_dim - len(state))),
        "img_is_pad": np.zeros(cfg.num_cams, dtype=bool),
        "local_img_is_pad": np.zeros(cfg.action_expert_num_cams, dtype=bool),
        "action_is_pad": np.zeros(cfg.action_chunk, dtype=bool),
    }


def _np2torch(
    np_input: dict[str, str | np.ndarray], device: str, dtype: torch.dtype
) -> dict[str, str | torch.Tensor]:
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
    r"""Convert Libero observation to PyTorch tensors."""
    np_input = _libero2np(obs, cfg)
    torch_input = _np2torch(np_input, device, dtype)
    return torch_input


def summarize_libero_results(results: list[int]) -> dict:
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


# This is not multi-processing safe, so every process should use a different (folder, camera_name) pair.
class LiberoObservationRecorder:
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
        for writer, camera in zip(self.writers, self.camera_names, strict=True):
            writer.append_data(np.rot90(obs[camera], k=2))

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.debug("Closing video writers.")
        for writer in self.writers:
            writer.close()
        logging.debug("Video writers closed.")
        return False
