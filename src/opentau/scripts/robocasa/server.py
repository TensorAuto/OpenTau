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

    Single (``client.py``):

        Client -> server (MessagePack): ``{ "images": {...}, "state": [...], "prompt": "..." }``
        Server -> client: ``list[float]``

    Batched (``client_async.py``):

        Client -> server:
            ``{ "batch": true, "items": [ { "images": {...}, "state": [...], "prompt": "..." }, ... ] }``
        Server -> client:
            ``[ list[float], ... ]``  # one action per item, same order and length as ``items``

**OpenTau mode** (default): loads the policy from ``policy.pretrained_path`` in the config.
For single requests, each step calls ``policy.select_action`` (internal action queue).
For **batched** requests, observations are stacked and ``select_action`` runs **once** on the
full batch (same as vector-env rollouts).

**Stub mode** (``--robocasa_use_stub=true``): small random actions.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
ROBOCASA_USE_STUB: bool = False


def _parse_robocasa_cli() -> None:
    """Parse ``--robocasa_*`` flags into module globals and strip them from ``sys.argv``.

    Must run before ``robocasa_async_main`` so ``argparse`` in the OpenTau config
    parser does not see RoboCasa-specific flags.

    Side effects:
        Updates ``ROBOCASA_HOST``, ``ROBOCASA_PORT``, ``ROBOCASA_ACTION_DIM``,
        ``ROBOCASA_TORCH_COMPILE``, ``ROBOCASA_USE_STUB``, and replaces ``sys.argv``
        with a copy containing only non-RoboCasa arguments.
    """
    global ROBOCASA_HOST, ROBOCASA_PORT, ROBOCASA_ACTION_DIM, ROBOCASA_TORCH_COMPILE, ROBOCASA_USE_STUB

    def _bool_arg(value: str) -> bool:
        """Parse a string as a boolean (true/1/yes/y, case-insensitive)."""

        return value.lower() in ("true", "1", "yes", "y")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--robocasa_host", type=str, default=None)
    p.add_argument("--robocasa_port", type=int, default=None)
    p.add_argument("--robocasa_action_dim", type=int, default=None)
    p.add_argument("--robocasa_torch_compile", type=str, default=None)
    p.add_argument("--robocasa_use_stub", type=str, default=None)
    args, rest = p.parse_known_args(sys.argv[1:])
    if args.robocasa_host is not None:
        ROBOCASA_HOST = args.robocasa_host
    if args.robocasa_port is not None:
        ROBOCASA_PORT = args.robocasa_port
    if args.robocasa_action_dim is not None:
        ROBOCASA_ACTION_DIM = args.robocasa_action_dim
    if args.robocasa_torch_compile is not None:
        ROBOCASA_TORCH_COMPILE = _bool_arg(args.robocasa_torch_compile)
    if args.robocasa_use_stub is not None:
        ROBOCASA_USE_STUB = _bool_arg(args.robocasa_use_stub)
    sys.argv = [sys.argv[0]] + rest


# Camera keys must match ``client.DEFAULT_CAMERA_NAMES``; order maps to camera0, camera1, ...
ROBOCASA_CAMERA_ORDER = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)


def jpeg_bytes_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode a JPEG bytestring to an RGB image array.

    Args:
        jpeg_bytes: Raw JPEG file bytes.

    Returns:
        ``uint8`` array of shape ``(H, W, 3)`` in RGB order.

    Raises:
        RuntimeError: If OpenCV (``cv2``) is not installed.
        ValueError: If ``cv2.imdecode`` fails (invalid JPEG).
    """
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to decode JPEG images on the server.")
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed (invalid JPEG?)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_all_images(images: Dict[str, bytes]) -> Dict[str, np.ndarray]:
    """Decode all camera JPEG blobs for ``ROBOCASA_CAMERA_ORDER``.

    Args:
        images: Map from camera name to JPEG bytes (must include every key in
            ``ROBOCASA_CAMERA_ORDER``).

    Returns:
        Map from camera name to ``uint8`` RGB arrays ``(H, W, 3)``.

    Raises:
        KeyError: If a required camera key is missing.
    """
    out: Dict[str, np.ndarray] = {}
    for k in ROBOCASA_CAMERA_ORDER:
        if k not in images:
            raise KeyError(f"Missing image key {k!r}; got {list(images.keys())}")
        out[k] = jpeg_bytes_to_rgb(images[k])
    return out


def unpack_payload_dict(data: dict) -> Tuple[Dict[str, bytes], np.ndarray, str]:
    """Parse one policy request body (single message or one batch item).

    Args:
        data: Decoded MessagePack dict with ``images``, ``state``, and optional
            ``prompt``.

    Returns:
        Tuple of ``(images_dict, state_vector, prompt_string)``. Image values are
        normalized to ``bytes``.

    Raises:
        ValueError: If ``images`` is not a dict or ``state`` is not a list.
    """
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


def unpack_request(message: bytes) -> Tuple[Dict[str, bytes], np.ndarray, str]:
    """Decode a single (non-batched) MessagePack WebSocket frame into observation fields.

    Args:
        message: Raw binary MessagePack payload.

    Returns:
        Same as ``unpack_payload_dict``.

    Raises:
        ValueError: If the top-level value is not a dict.
    """
    data = msgpack.unpackb(message, raw=False)
    if not isinstance(data, dict):
        raise ValueError("Expected dict payload")
    return unpack_payload_dict(data)


PolicyFn = Callable[
    [Dict[str, np.ndarray], np.ndarray, str, int, np.random.Generator],
    np.ndarray,
]


def default_policy(
    images_rgb: Dict[str, np.ndarray],
    state: np.ndarray,
    prompt: str,
    action_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stub policy: uniform random actions in ``[-0.05, 0.05]``.

    Args:
        images_rgb: Per-camera ``uint8`` RGB arrays (unused in stub).
        state: Proprioceptive state vector (unused in stub).
        prompt: Task text (unused in stub).
        action_dim: Flat action size.
        rng: NumPy random generator.

    Returns:
        ``float64`` array of shape ``(action_dim,)``.
    """
    del images_rgb, prompt  # unused in stub
    _ = state  # available for your model
    return rng.uniform(-0.05, 0.05, size=(action_dim,)).astype(np.float64)


def policy_forward(
    images_rgb: Dict[str, np.ndarray],
    state: np.ndarray,
    prompt: str,
    action_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Default policy entrypoint; replace implementation while keeping the signature.

    Currently delegates to ``default_policy``.

    Args:
        images_rgb: Per-camera ``uint8`` RGB arrays.
        state: Proprioceptive state vector.
        prompt: Task text.
        action_dim: Flat action size.
        rng: NumPy random generator.

    Returns:
        Flat action vector of shape ``(action_dim,)``.
    """
    return default_policy(images_rgb, state, prompt, action_dim, rng)


def pack_action(action: np.ndarray) -> bytes:
    """Serialize a single flat action as MessagePack bytes for the WebSocket reply.

    Args:
        action: 1D action vector (any shape that ravel-s to the policy dimension).

    Returns:
        MessagePack-encoded bytes (list of floats).
    """
    a = np.asarray(action, dtype=np.float64).ravel()
    return msgpack.packb(a.tolist(), use_bin_type=True)


def pack_actions_batch(actions: List[np.ndarray]) -> bytes:
    """Serialize a list of flat actions for a batched WebSocket reply.

    Args:
        actions: One numpy action per batch row, same order as request ``items``.

    Returns:
        MessagePack-encoded ``list[list[float]]`` bytes.
    """
    return msgpack.packb(
        [np.asarray(a, dtype=np.float64).ravel().tolist() for a in actions],
        use_bin_type=True,
    )


def _numpy_rgb_to_camera_tensor(
    rgb_uint8: np.ndarray,
    resolution: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resize RGB ``uint8`` image to policy resolution and produce a CHW batch slice.

    Args:
        rgb_uint8: Image array ``(H, W, 3)`` RGB.
        resolution: Target ``(height, width)`` as in config (H, W).
        device: Torch device for the output tensor.
        dtype: Floating dtype for normalized pixels.

    Returns:
        Tensor of shape ``(1, 3, H, W)`` with values in ``[0, 1]``.
    """
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
    """Build a single-row OpenTau policy batch from one RoboCasa observation.

    Args:
        cfg: Training pipeline config (camera count, resolution, state dim, etc.).
        images_rgb: Decoded RGB arrays keyed by ``ROBOCASA_CAMERA_ORDER`` names.
        state_vec: Proprio vector; padded or truncated to ``cfg.max_state_dim``.
        prompt: Task string.
        device: Torch device.
        dtype: Floating dtype for tensors.

    Returns:
        Dict of tensors including ``camera*``, ``state``, ``prompt``, ``img_is_pad``.
    """
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
    """Stack multiple decoded observations into one OpenTau batch of batch size ``B``.

    Args:
        cfg: Training pipeline config.
        items: List of ``(images_rgb, state_vec, prompt)`` per environment row.
        device: Torch device.
        dtype: Floating dtype for tensors.

    Returns:
        Batched tensor dict for ``select_action`` / ``sample_actions``.

    Raises:
        ValueError: If ``items`` is empty.
    """
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
        """Load the policy from ``cfg.policy.pretrained_path`` and warm up inference.

        Args:
            cfg: Full training pipeline config (policy type, resolution, state dim, etc.).
            compile_model: If True, apply ``torch.compile`` to ``sample_actions`` when
                supported.
            seed: Optional RNG seed applied before construction via ``set_seed``.

        Side effects:
            Loads weights, moves the model to ``auto_torch_device()`` in bfloat16,
            runs two dummy ``sample_actions`` calls for warmup, then ``reset()`` again.
        """
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
        """Run a single-environment forward pass and return a fixed-length action.

        Args:
            images_rgb: Decoded RGB images per camera.
            state_vec: Proprio state; padded to ``cfg.max_state_dim`` in the batch.
            prompt: Task string.
            action_dim: Desired output length (RoboCasa / CLI); policy output is
                truncated or zero-padded to this size.

        Returns:
            ``float64`` vector of shape ``(action_dim,)``.
        """
        batch = build_opentau_batch(self.cfg, images_rgb, state_vec, prompt, self.device, self.dtype)
        with torch.inference_mode():
            act = self.policy.select_action(batch)
        step0 = act.squeeze(0).to("cpu", torch.float32).numpy().ravel()
        policy_adim = step0.shape[0]
        out = np.zeros(action_dim, dtype=np.float64)
        n = min(action_dim, policy_adim)
        out[:n] = step0[:n].astype(np.float64)
        return out

    def infer_batch(
        self,
        decoded_items: List[Tuple[Dict[str, np.ndarray], np.ndarray, str]],
        action_dim: int,
    ) -> List[np.ndarray]:
        """Run one batched ``select_action`` for multiple observations.

        Args:
            decoded_items: One triple ``(images_rgb, state_vec, prompt)`` per batch row.
            action_dim: Target flat size per row (truncate or pad each output).

        Returns:
            List of ``action_dim``-length ``float64`` arrays, same order as ``decoded_items``.

        Raises:
            ValueError: If the policy output batch size is smaller than the number of
                input rows.
        """
        batch = build_opentau_batch_multi(self.cfg, decoded_items, self.device, self.dtype)
        b = len(decoded_items)
        with torch.inference_mode():
            act = self.policy.select_action(batch)
        act_np = act.to("cpu", torch.float32).numpy()
        if act_np.ndim == 1:
            act_np = act_np.reshape(1, -1)
        # Some policies pad to a fixed max batch size; only return rows for this request.
        if act_np.shape[0] > b:
            act_np = act_np[:b]
        elif act_np.shape[0] < b:
            raise ValueError(f"Policy returned batch dim {act_np.shape[0]} < input batch {b}")
        outs: List[np.ndarray] = []
        for row in range(act_np.shape[0]):
            step0 = act_np[row].ravel()
            policy_adim = step0.shape[0]
            out = np.zeros(action_dim, dtype=np.float64)
            n = min(action_dim, policy_adim)
            out[:n] = step0[:n].astype(np.float64)
            outs.append(out)
        return outs


def make_handler(
    action_dim: int,
    policy: PolicyFn,
    rng: np.random.Generator,
    opentau_runner: Optional[OpenTauRoboCasaPolicy] = None,
):
    """Build the asyncio WebSocket handler for single and batched policy requests.

    Args:
        action_dim: Expected flat action size for validation and zero-fill on errors.
        policy: Callable used when ``opentau_runner`` is None (stub or custom).
        rng: NumPy generator passed to ``policy`` for stochastic stubs.
        opentau_runner: If set, batch paths use ``OpenTauRoboCasaPolicy.infer_batch``
            for efficiency; otherwise batch is looped with ``policy``.

    Returns:
        An async function suitable for ``websockets.serve`` that reads MessagePack
        frames and sends MessagePack-encoded actions.
    """

    async def _handler(websocket: Any):
        """Handle one WebSocket connection: MessagePack in, MessagePack actions out."""

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

                    if opentau_runner is not None:
                        actions_out = opentau_runner.infer_batch(decoded, action_dim)
                    else:
                        actions_out = []
                        for images_rgb, state, prompt in decoded:
                            action = policy(images_rgb, state, prompt, action_dim, rng)
                            if action.shape[0] != action_dim:
                                raise ValueError(
                                    f"Policy returned shape {action.shape}, expected ({action_dim},)"
                                )
                            actions_out.append(action)

                    await websocket.send(pack_actions_batch(actions_out))
                else:
                    images_jpeg, state, prompt = unpack_payload_dict(data)
                    images_rgb = decode_all_images(images_jpeg)
                    action = policy(images_rgb, state, prompt, action_dim, rng)
                    if action.shape[0] != action_dim:
                        raise ValueError(f"Policy returned shape {action.shape}, expected ({action_dim},)")
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
                    zeros = [np.zeros(action_dim, dtype=np.float64).tolist() for _ in range(n)]
                    await websocket.send(msgpack.packb(zeros, use_bin_type=True))
                else:
                    await websocket.send(pack_action(np.zeros(action_dim, dtype=np.float64)))

    return _handler


def make_opentau_handler(runner: OpenTauRoboCasaPolicy) -> PolicyFn:
    """Adapt ``OpenTauRoboCasaPolicy`` to the generic ``PolicyFn`` signature.

    The returned callable ignores ``rng`` (deterministic inference).

    Args:
        runner: Loaded policy wrapper.

    Returns:
        A ``PolicyFn`` that forwards to ``OpenTauRoboCasaPolicy.infer``.
    """

    def _policy(
        images_rgb: Dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
        adim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Single-env policy shim; ``rng`` is unused."""

        del rng
        return runner.infer(images_rgb, state, prompt, adim)

    return _policy


async def run_server(
    host: str,
    port: int,
    action_dim: int,
    policy: Optional[PolicyFn] = None,
    seed: int = 0,
    opentau_runner: Optional[OpenTauRoboCasaPolicy] = None,
) -> None:
    """Start the WebSocket server and block until the process is interrupted.

    Args:
        host: Bind address (e.g. ``0.0.0.0``).
        port: TCP port.
        action_dim: Flat action dimension for validation and error fallbacks.
        policy: Policy callable for non-OpenTau or stub mode; defaults to
            ``policy_forward``.
        seed: Seed for the numpy RNG used by stub / default policy.
        opentau_runner: When non-None, batched requests use ``infer_batch`` on this
            object; single requests still go through ``policy`` (typically from
            ``make_opentau_handler``).

    Note:
        Uses ``ping_timeout=None`` and ``max_size=None`` for large payloads and
        long inference times. Runs until cancelled (infinite ``asyncio.Future``).
    """
    rng = np.random.default_rng(seed)
    pol: PolicyFn = policy if policy is not None else policy_forward

    async with websockets.serve(
        make_handler(action_dim, pol, rng, opentau_runner=opentau_runner),
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
    """CLI entry: parse config, optionally load OpenTau policy, and run ``run_server``.

    Honors module globals set by ``_parse_robocasa_cli`` (host, port, action
    dimension, stub vs OpenTau, torch compile). When ``ROBOCASA_USE_STUB`` is True,
    uses ``policy_forward``; otherwise builds ``OpenTauRoboCasaPolicy`` and
    a handler from ``make_opentau_handler``.

    Args:
        cfg: Parsed ``TrainPipelineConfig`` from OpenTau's argparse (includes
            ``policy``, ``seed``, etc.).
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "%s\nRoboCasa globals: host=%s port=%s action_dim=%s torch_compile=%s use_stub=%s",
        pformat(asdict(cfg)),
        ROBOCASA_HOST,
        ROBOCASA_PORT,
        ROBOCASA_ACTION_DIM,
        ROBOCASA_TORCH_COMPILE,
        ROBOCASA_USE_STUB,
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    seed = int(cfg.seed) if cfg.seed is not None else 0

    policy_fn: PolicyFn
    runner: Optional[OpenTauRoboCasaPolicy] = None
    if ROBOCASA_USE_STUB:
        policy_fn = policy_forward
    else:
        runner = OpenTauRoboCasaPolicy(
            cfg,
            compile_model=ROBOCASA_TORCH_COMPILE,
            seed=cfg.seed,
        )
        policy_fn = make_opentau_handler(runner)

    asyncio.run(
        run_server(
            host=ROBOCASA_HOST,
            port=ROBOCASA_PORT,
            action_dim=ROBOCASA_ACTION_DIM,
            policy=policy_fn,
            seed=seed,
            opentau_runner=runner,
        )
    )


if __name__ == "__main__":
    _parse_robocasa_cli()
    init_logging()
    robocasa_async_main()
