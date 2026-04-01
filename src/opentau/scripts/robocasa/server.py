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

"""Policy WebSocket server for RoboCasa ``client.py`` / ``client_async.py`` with OpenTau loading.

Implements the same wire protocol as ``robocasa_server``, plus **batched** messages for
``robocasa.scripts.client_async``:

    Single request:

        Client -> server (MessagePack): ``{ "images": {...}, "state": [...], "prompt": "..." }``
        Server -> client: ``[[float, ...], ...]``  # shape ``(T, action_dim)``

    Batched (``client_async.py``):

        Client -> server:
            ``{ "batch": true, "items": [ { "images": {...}, "state": [...], "prompt": "..." }, ... ] }``
        Server -> client:
            ``[ [[float, ...], ...], ... ]``  # per item: full action chunk (``T`` steps × ``action_dim``)

**OpenTau mode** (default): loads the policy from ``policy.pretrained_path`` in the config.
Each request runs ``policy.sample_actions`` (no internal queue on the server). The reply is the
full predicted chunk per environment: shape ``(n_action_steps, action_dim)`` (trimmed/padded to
``--robocasa_action_dim``). **Batched** requests stack observations and call ``sample_actions`` once.

Run::

    python -m opentau.scripts.robocasa_server_async \\
        --config_path /path/to/train_config.json \\
        --robocasa_action_dim 16 --robocasa_port 8765

Dependencies: ``websockets``, ``msgpack``, ``opencv-python`` (optional but recommended for JPEG decode).
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import asdict
from pprint import pformat
from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

import websockets

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import attempt_torch_compile, auto_torch_device, init_logging

logger = logging.getLogger(__name__)

# WebSocket server options (defaults; override with --robocasa_* before ``TrainPipelineConfig`` parse).
ROBOCASA_HOST: str = "0.0.0.0"  # nosec B104 — default listen; use ``--robocasa_host`` to restrict
ROBOCASA_PORT: int = 8765
ROBOCASA_ACTION_DIM: int = 16
ROBOCASA_TORCH_COMPILE: bool = True

# Fallback zero chunk length when sending an error response (client should discard).
_ERROR_RESPONSE_CHUNK_STEPS: int = 8


def _parse_robocasa_cli() -> None:
    """Read ``--robocasa_*`` flags into module globals and remove them from ``sys.argv``."""
    global ROBOCASA_HOST, ROBOCASA_PORT, ROBOCASA_ACTION_DIM, ROBOCASA_TORCH_COMPILE

    def _bool_arg(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "y")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--robocasa_host", type=str, default=None)
    p.add_argument("--robocasa_port", type=int, default=None)
    p.add_argument("--robocasa_action_dim", type=int, default=None)
    p.add_argument("--robocasa_torch_compile", type=str, default=None)
    args, rest = p.parse_known_args(sys.argv[1:])
    if args.robocasa_host is not None:
        ROBOCASA_HOST = args.robocasa_host
    if args.robocasa_port is not None:
        ROBOCASA_PORT = args.robocasa_port
    if args.robocasa_action_dim is not None:
        ROBOCASA_ACTION_DIM = args.robocasa_action_dim
    if args.robocasa_torch_compile is not None:
        ROBOCASA_TORCH_COMPILE = _bool_arg(args.robocasa_torch_compile)
    sys.argv = [sys.argv[0]] + rest


# Camera keys must match ``client.DEFAULT_CAMERA_NAMES``; order maps to camera0, camera1, ...
ROBOCASA_CAMERA_ORDER = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)


def jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to HxWx3 uint8 RGB."""
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to decode JPEG images on the server.")
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed (invalid JPEG?)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_all_images(images: Dict[str, bytes]) -> Dict[str, np.ndarray]:
    """Decode all camera JPEGs to RGB numpy arrays (uint8, H, W, 3)."""
    out: Dict[str, np.ndarray] = {}
    for k in ROBOCASA_CAMERA_ORDER:
        if k not in images:
            raise KeyError(f"Missing image key {k!r}; got {list(images.keys())}")
        out[k] = jpeg_bytes_to_rgb(images[k])
    return out


def unpack_payload_dict(data: dict) -> Tuple[Dict[str, bytes], np.ndarray, str]:
    """Parse one policy request body (single message or one element of a batch)."""
    images = data.get("images")
    state = data.get("state")
    prompt = data.get("prompt", "")
    if not isinstance(images, dict):
        raise ValueError("Expected 'images' dict")
    if not isinstance(state, list):
        raise ValueError("Expected 'state' list")
    images = {str(k): (v if isinstance(v, (bytes, bytearray)) else bytes(v)) for k, v in images.items()}
    state_vec = np.asarray(state, dtype=np.float64)
    prompt_str = str(prompt) if prompt is not None else ""
    return images, state_vec, prompt_str


def pack_action(action_chunk: np.ndarray) -> bytes:
    """MessagePack-encode one env's action chunk ``(T, action_dim)`` as nested lists."""
    a = np.asarray(action_chunk, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected action chunk of shape (T, action_dim), got shape {a.shape}")
    return msgpack.packb(a.tolist(), use_bin_type=True)


def pack_actions_batch(chunks: List[np.ndarray]) -> bytes:
    """Encode one ``(T, action_dim)`` chunk per batch row (same order as request ``items``)."""
    return msgpack.packb(
        [np.asarray(c, dtype=np.float64).tolist() for c in chunks],
        use_bin_type=True,
    )


def _numpy_rgb_to_camera_tensor(
    rgb_uint8: np.ndarray,
    resolution: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """RGB uint8 (H,W,3) -> (1,3,H,W) float [0,1] on device."""
    pil = Image.fromarray(rgb_uint8)
    pil = pil.resize((resolution[1], resolution[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def build_opentau_batch(
    cfg: TrainPipelineConfig,
    images_rgb: Dict[str, np.ndarray],
    state_vec: np.ndarray,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Map RoboCasa observation dict to OpenTau policy batch (batch size 1)."""
    num_cams = cfg.num_cams
    resolution = cfg.resolution
    batch: dict[str, torch.Tensor] = {}
    img_is_pad: list[bool] = []

    for cam_idx in range(num_cams):
        if cam_idx < len(ROBOCASA_CAMERA_ORDER):
            key = ROBOCASA_CAMERA_ORDER[cam_idx]
            rgb = images_rgb[key]
            batch[f"camera{cam_idx}"] = _numpy_rgb_to_camera_tensor(rgb, resolution, device, dtype)
            img_is_pad.append(False)
        else:
            batch[f"camera{cam_idx}"] = torch.zeros((1, 3, *resolution), dtype=dtype, device=device)
            img_is_pad.append(True)

    state_list = state_vec.astype(np.float64).ravel().tolist()
    if len(state_list) < cfg.max_state_dim:
        state_list.extend([0.0] * (cfg.max_state_dim - len(state_list)))
    state_list = state_list[: cfg.max_state_dim]
    batch["state"] = torch.tensor([state_list], dtype=dtype, device=device)
    raw_prompt = prompt.strip() if prompt else ""
    batch["prompt"] = [str(raw_prompt) or ""]
    batch["img_is_pad"] = torch.tensor([img_is_pad], dtype=torch.bool, device=device)
    return batch


def build_opentau_batch_multi(
    cfg: TrainPipelineConfig,
    items: List[Tuple[Dict[str, np.ndarray], np.ndarray, str]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Stack multiple RoboCasa observations into one OpenTau batch (batch size B)."""
    b = len(items)
    if b == 0:
        raise ValueError("empty batch")
    num_cams = cfg.num_cams
    resolution = cfg.resolution
    batch: dict[str, torch.Tensor] = {}
    img_is_pad_rows: list[list[bool]] = []

    for cam_idx in range(num_cams):
        if cam_idx < len(ROBOCASA_CAMERA_ORDER):
            key = ROBOCASA_CAMERA_ORDER[cam_idx]
            cam_tensors: list[torch.Tensor] = []
            for images_rgb, _state_vec, _prompt in items:
                rgb = images_rgb[key]
                t = _numpy_rgb_to_camera_tensor(rgb, resolution, device, dtype)
                cam_tensors.append(t.squeeze(0))
            batch[f"camera{cam_idx}"] = torch.stack(cam_tensors, dim=0)
            img_is_pad_rows.append([False] * b)
        else:
            batch[f"camera{cam_idx}"] = torch.zeros((b, 3, *resolution), dtype=dtype, device=device)
            img_is_pad_rows.append([True] * b)

    img_is_pad_arr = np.array(img_is_pad_rows, dtype=bool).T
    batch["img_is_pad"] = torch.tensor(img_is_pad_arr, dtype=torch.bool, device=device)

    state_rows: list[list[float]] = []
    prompts: list[str] = []
    for _images_rgb, state_vec, prompt in items:
        state_list = state_vec.astype(np.float64).ravel().tolist()
        if len(state_list) < cfg.max_state_dim:
            state_list.extend([0.0] * (cfg.max_state_dim - len(state_list)))
        state_list = state_list[: cfg.max_state_dim]
        state_rows.append(state_list)
        raw_prompt = prompt.strip() if prompt else ""
        prompts.append(str(raw_prompt) or "")

    batch["state"] = torch.tensor(state_rows, dtype=dtype, device=device)
    batch["prompt"] = prompts
    return batch


class OpenTauRoboCasaPolicy:
    """Loads an OpenTau policy from ``TrainPipelineConfig`` and runs inference."""

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        *,
        compile_model: bool = True,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = auto_torch_device()
        self.dtype = torch.bfloat16
        if seed is not None:
            set_seed(seed)

        logger.info("Loading OpenTau policy type=%s from %s", cfg.policy.type, cfg.policy.pretrained_path)
        policy_class = get_policy_class(cfg.policy.type)
        self.policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
        self.policy.to(device=self.device, dtype=self.dtype)
        self.policy.eval()

        if compile_model:
            self.policy.model.sample_actions = attempt_torch_compile(
                self.policy.model.sample_actions, device_hint=self.device
            )

        self.policy.reset()

        dummy = build_opentau_batch(
            cfg,
            {
                ROBOCASA_CAMERA_ORDER[0]: np.zeros((*cfg.resolution, 3), dtype=np.uint8),
                ROBOCASA_CAMERA_ORDER[1]: np.zeros((*cfg.resolution, 3), dtype=np.uint8),
                ROBOCASA_CAMERA_ORDER[2]: np.zeros((*cfg.resolution, 3), dtype=np.uint8),
            },
            np.zeros(cfg.max_state_dim, dtype=np.float64),
            "warmup",
            self.device,
            self.dtype,
        )
        with torch.inference_mode():
            _ = self.policy.sample_actions(dummy)
            _ = self.policy.sample_actions(dummy)
        self.policy.reset()
        logger.info("OpenTau policy ready on %s", self.device)

    def infer(
        self,
        images_rgb: Dict[str, np.ndarray],
        state_vec: np.ndarray,
        prompt: str,
        action_dim: int,
    ) -> np.ndarray:
        """Return full action chunk ``(T, action_dim)`` from ``sample_actions`` (trim/pad last dim)."""
        batch = build_opentau_batch(self.cfg, images_rgb, state_vec, prompt, self.device, self.dtype)
        with torch.inference_mode():
            act = self.policy.sample_actions(batch)
        # (1, T, policy_dim)
        act_np = act.squeeze(0).to("cpu", torch.float32).numpy()
        if act_np.ndim != 2:
            raise ValueError(f"Expected policy output (T, D), got shape {act_np.shape}")
        t_steps, policy_adim = act_np.shape
        out = np.zeros((t_steps, action_dim), dtype=np.float64)
        n = min(action_dim, policy_adim)
        out[:, :n] = act_np[:, :n].astype(np.float64)
        return out

    def infer_batch(
        self,
        decoded_items: List[Tuple[Dict[str, np.ndarray], np.ndarray, str]],
        action_dim: int,
    ) -> List[np.ndarray]:
        """One ``sample_actions`` on a stacked batch; one ``(T, action_dim)`` chunk per env."""
        batch = build_opentau_batch_multi(self.cfg, decoded_items, self.device, self.dtype)
        b = len(decoded_items)
        with torch.inference_mode():
            act = self.policy.sample_actions(batch)
        act_np = act.to("cpu", torch.float32).numpy()
        if act_np.ndim == 2:
            act_np = act_np.reshape(1, *act_np.shape)
        if act_np.ndim != 3:
            raise ValueError(f"Expected policy output (B, T, D), got shape {act_np.shape}")
        if act_np.shape[0] > b:
            act_np = act_np[:b]
        elif act_np.shape[0] < b:
            raise ValueError(f"Policy returned batch dim {act_np.shape[0]} < input batch {b}")
        _, t_steps, policy_adim = act_np.shape
        outs: List[np.ndarray] = []
        for row in range(b):
            out = np.zeros((t_steps, action_dim), dtype=np.float64)
            n = min(action_dim, policy_adim)
            out[:, :n] = act_np[row, :, :n].astype(np.float64)
            outs.append(out)
        return outs


def make_handler(action_dim: int, runner: OpenTauRoboCasaPolicy):
    async def _handler(websocket: Any):
        async for message in websocket:
            try:
                data = msgpack.unpackb(message, raw=False)
                if not isinstance(data, dict):
                    raise ValueError("Expected dict payload")

                if data.get("batch") is True:
                    items = data.get("items")
                    if not isinstance(items, list):
                        raise ValueError("Batch request requires 'items' list")
                    decoded: List[Tuple[Dict[str, np.ndarray], np.ndarray, str]] = []
                    for item in items:
                        if not isinstance(item, dict):
                            raise ValueError("Each batch item must be a dict")
                        images_jpeg, state, prompt = unpack_payload_dict(item)
                        images_rgb = decode_all_images(images_jpeg)
                        decoded.append((images_rgb, state, prompt))

                    actions_out = runner.infer_batch(decoded, action_dim)
                    await websocket.send(pack_actions_batch(actions_out))
                else:
                    images_jpeg, state, prompt = unpack_payload_dict(data)
                    images_rgb = decode_all_images(images_jpeg)
                    action = runner.infer(images_rgb, state, prompt, action_dim)
                    await websocket.send(pack_action(action))
            except Exception as e:
                logger.exception("Policy step failed: %s", e)
                try:
                    data = msgpack.unpackb(message, raw=False)
                except Exception:
                    data = None
                if isinstance(data, dict) and data.get("batch") is True:
                    items = data.get("items") if isinstance(data.get("items"), list) else []
                    n = len(items)
                    zero_chunk = [[0.0] * action_dim for _ in range(_ERROR_RESPONSE_CHUNK_STEPS)]
                    await websocket.send(msgpack.packb([zero_chunk for _ in range(n)], use_bin_type=True))
                else:
                    await websocket.send(
                        pack_action(np.zeros((_ERROR_RESPONSE_CHUNK_STEPS, action_dim), dtype=np.float64))
                    )

    return _handler


async def run_server(
    host: str,
    port: int,
    action_dim: int,
    runner: OpenTauRoboCasaPolicy,
) -> None:
    async with websockets.serve(
        make_handler(action_dim, runner),
        host,
        port,
        max_size=None,
        ping_timeout=None,
    ):
        print(
            f"RoboCasa policy server (async/batch) listening on ws://{host}:{port} "
            f"(action_dim={action_dim}). Waiting for client…"
        )
        await asyncio.Future()  # run forever


@parser.wrap()
def robocasa_async_main(cfg: TrainPipelineConfig) -> None:
    """Start the RoboCasa WebSocket policy server (single + batched) using OpenTau config parsing."""
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "%s\nRoboCasa globals: host=%s port=%s action_dim=%s torch_compile=%s",
        pformat(asdict(cfg)),
        ROBOCASA_HOST,
        ROBOCASA_PORT,
        ROBOCASA_ACTION_DIM,
        ROBOCASA_TORCH_COMPILE,
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    runner = OpenTauRoboCasaPolicy(
        cfg,
        compile_model=ROBOCASA_TORCH_COMPILE,
        seed=cfg.seed,
    )

    asyncio.run(
        run_server(
            host=ROBOCASA_HOST,
            port=ROBOCASA_PORT,
            action_dim=ROBOCASA_ACTION_DIM,
            runner=runner,
        )
    )


if __name__ == "__main__":
    _parse_robocasa_cli()
    init_logging()
    robocasa_async_main()
