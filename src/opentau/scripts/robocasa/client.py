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

"""
Threaded batched remote policy client for RoboCasa.

Runs **n_parallel** environment threads. Each thread pulls rollouts from a shared queue
until **num_rollouts** episodes are finished. The **main** asyncio loop receives
observations from active workers, batches them into one WebSocket message, and routes
returned action chunks back to the corresponding threads.

Batch protocol (MessagePack over WebSocket, binary frames) matches ``client.py`` /
``robocasa.scripts.server``:

    Client -> server: {
        "batch": true,
        "items": [
            { "images": { camera_name: bytes (JPEG), ... }, "state": list[float], "prompt": str },
            ...
        ],
    }

    Server -> client: list[list[list[float]]]  # one action chunk per item, same order as ``items``

The number of ``items`` (and thus the batch size) is **only** the count of workers
that need a new chunk right now. As workers finish their rollout queue and exit, batch
size shrinks from at most ``num_parallel`` down to 1 for the final active worker(s).
The policy server must return exactly ``len(items)`` actions, not a fixed width of
``num_parallel``.

Rollout records and ``rollouts.json`` match ``client.py`` (``env_name``, ``seed``,
``length``, ``success`` per rollout; summary includes ``num_rollouts``,
``num_parallel_envs``, ``output_directory``).

Requires ``websockets``, ``msgpack``, ``opencv-python`` (``cv2``). The WebSocket client
sets ``ping_timeout=None`` so MuJoCo stepping and JPEG encoding do not trip keepalive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union

import imageio
import msgpack
import numpy as np
import websockets

import robocasa  # noqa: F401
from robocasa.scripts.client import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_CAMERA_WIDTH,
    build_proprio_vector,
    encode_all_cameras_jpeg,
    flip_image_obs,
    get_task_prompt,
)
from robocasa.utils.env_utils import convert_action_pi05, create_env


@dataclass
class ObsMsg:
    """Worker needs a policy action for this observation (packed client payload)."""

    payload: dict[str, Any]


@dataclass
class DoneMsg:
    """Episode finished; no server call for this message."""

    rollout_idx: int
    length: int
    success: bool


@dataclass
class ExitMsg:
    """Worker thread has no more rollouts and is exiting."""


WorkerToMain = Union[ObsMsg, DoneMsg, ExitMsg]

_SERVER_TRUNCATED_ACTION_BATCH_WARNED = False


def _normalize_batched_actions_response(
    actions_batch: Any,
    num_expected: int,
) -> list[Any]:
    """
    Ensure ``actions_batch`` is a list of length ``num_expected``, one action chunk per batch row.

    When ``num_expected == 1``, some servers may return one chunk directly
    (``list[list[float]]``) or one flat action (``list[float]``) instead of
    ``[chunk]``; wrap those cases.

    When the server returns *more* rows than ``num_expected`` (e.g. fixed max batch
    width while the client sends a partial batch), the excess rows are dropped.
    """
    global _SERVER_TRUNCATED_ACTION_BATCH_WARNED
    if not isinstance(actions_batch, list):
        raise ValueError(f"Batched server response must be a list, got {type(actions_batch).__name__}")
    if len(actions_batch) == num_expected:
        return actions_batch
    if num_expected == 1 and len(actions_batch) > 0:
        first = actions_batch[0]
        if isinstance(first, (int, float, np.floating, np.integer, list, tuple, np.ndarray)):
            return [actions_batch]
    if len(actions_batch) > num_expected:
        if not _SERVER_TRUNCATED_ACTION_BATCH_WARNED:
            warnings.warn(
                f"Policy server returned {len(actions_batch)} actions for a batch of "
                f"{num_expected}; using the first {num_expected}. Prefer fixing the server "
                f"to return exactly len(items) actions.",
                UserWarning,
                stacklevel=2,
            )
            _SERVER_TRUNCATED_ACTION_BATCH_WARNED = True
        return actions_batch[:num_expected]
    raise ValueError(
        f"Batched actions length {len(actions_batch)} != batch size {num_expected} "
        f"(partial batches must still return one action list per observation)"
    )


def _normalize_action_chunk_for_worker(raw_action_chunk: Any) -> list[np.ndarray]:
    """Convert one server row into a list of flat action vectors."""
    arr = np.asarray(raw_action_chunk, dtype=np.float64)
    if arr.ndim == 1:
        return [arr.ravel()]
    if arr.ndim != 2:
        raise ValueError(f"Expected action chunk rank 1 or 2, got shape {arr.shape}")
    return [arr[i].ravel() for i in range(arr.shape[0])]


def _worker_loop(
    *,
    rollout_queue: queue.Queue[int | None],
    to_main: queue.Queue[WorkerToMain],
    from_main: queue.Queue[Any],
    env_name: str,
    split,
    start_seed: int,
    main_dir: str,
    jpeg_quality: int,
    max_episode_steps: int | None,
    render: bool,
    action_dim_holder: list[int | None],
    action_dim_lock: threading.Lock,
) -> None:
    """One thread: sequential rollouts from ``rollout_queue`` until empty."""
    while True:
        try:
            # get the next rollout index from the queue and its protected by a lock
            rollout_idx = rollout_queue.get_nowait()
        except queue.Empty:
            to_main.put(ExitMsg())
            return

        seed = start_seed + rollout_idx
        if not render:
            # create the video subdirectory for the rollout
            sub = os.path.join(main_dir, f"rollout_{rollout_idx:04d}_seed_{seed}")
            os.makedirs(sub, exist_ok=True)
            video_writers: dict[str, Any] | None = {}
            for cam in DEFAULT_CAMERA_NAMES:
                path = os.path.join(sub, f"{cam}.mp4")
                video_writers[cam] = imageio.get_writer(path, fps=20)
        else:
            video_writers = None

        # create the environment
        env = create_env(
            env_name,
            split=split,
            seed=seed,
            render_onscreen=render,
            camera_names=list(DEFAULT_CAMERA_NAMES),
            camera_widths=DEFAULT_CAMERA_WIDTH,
            camera_heights=DEFAULT_CAMERA_HEIGHT,
            has_offscreen_renderer=not render,
            use_camera_obs=not render,
        )
        try:
            # reset the environment and get the initial observation and action dimension
            obs = env.reset()
            # flip the image observations as mujoco returns flipped images
            obs = flip_image_obs(obs, DEFAULT_CAMERA_NAMES)
            # get the action dimension and store it in the action dimension holder
            with action_dim_lock:
                if action_dim_holder[0] is None:
                    ad = env.action_dim
                    if ad is None:
                        raise RuntimeError("env.action_dim is None after reset()")
                    action_dim_holder[0] = ad
            step_count = 0
            pending_actions: list[np.ndarray] = []

            while True:
                if render:
                    images: dict[str, Any] = {}
                else:
                    # encode the image observations as JPEG
                    images = encode_all_cameras_jpeg(obs, DEFAULT_CAMERA_NAMES, jpeg_quality=jpeg_quality)
                    # write the image observations to the video writers
                    if video_writers is not None:
                        for cam in DEFAULT_CAMERA_NAMES:
                            cam_key = f"{cam}_image"
                            if cam_key in obs:
                                video_writers[cam].append_data(obs[cam_key])

                # build the state vector in desried order and get the task prompt
                state = build_proprio_vector(obs).tolist()
                prompt = get_task_prompt(env)
                payload_obs = {"images": images, "state": state, "prompt": prompt}

                # request a new policy chunk only when local chunk is exhausted
                if len(pending_actions) == 0:
                    # send the payload to the main thread
                    to_main.put(ObsMsg(payload=payload_obs))
                    raw_action_chunk = from_main.get()
                    pending_actions = _normalize_action_chunk_for_worker(raw_action_chunk)
                    if len(pending_actions) == 0:
                        raise ValueError("Server returned an empty action chunk")

                # take one action from the local chunk
                action = pending_actions.pop(0)
                # build action vector in desired order
                action = convert_action_pi05(action)

                # check if the action dimension is correct
                ad = action_dim_holder[0]
                assert ad is not None
                if action.shape[0] != ad:
                    raise ValueError(f"Policy returned action dim {action.shape[0]}, expected {ad}")

                # step the environment and get the new observation
                obs, _r, _d, _i = env.step(action)
                # flip the image observations as mujoco returns flipped images
                obs = flip_image_obs(obs, DEFAULT_CAMERA_NAMES)
                step_count += 1

                # check if the episode is over
                episode_over = bool(env._check_success()) or (
                    max_episode_steps is not None and step_count >= max_episode_steps
                )
                if episode_over:
                    success = bool(env._check_success())
                    to_main.put(
                        DoneMsg(
                            rollout_idx=rollout_idx,
                            length=step_count,
                            success=success,
                        )
                    )
                    break
        finally:
            if video_writers is not None:
                for w in video_writers.values():
                    w.close()
            env.close()


async def _run_coordinator(
    *,
    ws_uri: str,
    n_workers: int,
    to_mains: list[queue.Queue[WorkerToMain]],
    from_mains: list[queue.Queue[Any]],
    results_by_rollout: dict[int, tuple[int, bool]],
    results_lock: threading.Lock,
) -> None:
    """
    For each timestep, read from all active workers in parallel until each has produced
    one ``ObsMsg`` (skipping ``DoneMsg``) or ``ExitMsg``. This avoids deadlock when one
    worker finishes an episode and is slow to start the next rollout while others already
    have the next observation ready.
    """
    loop = asyncio.get_event_loop()

    def _get(q: queue.Queue[WorkerToMain]) -> WorkerToMain:
        return q.get()

    async def _drain_to_obs_or_exit(wid: int) -> tuple[int, ObsMsg | None, bool]:
        """Returns (worker_id, ObsMsg or None if exiting, is_exit)."""
        while True:
            msg = await loop.run_in_executor(None, _get, to_mains[wid])
            if isinstance(msg, ExitMsg):
                return (wid, None, True)
            if isinstance(msg, DoneMsg):
                with results_lock:
                    results_by_rollout[msg.rollout_idx] = (msg.length, msg.success)
                continue
            if isinstance(msg, ObsMsg):
                return (wid, msg, False)
            raise TypeError(f"Unexpected message: {type(msg)}")

    async with websockets.connect(
        ws_uri,
        max_size=None,
        ping_timeout=None,
    ) as websocket:
        active: set[int] = set(range(n_workers))

        while active:
            wids = sorted(active)
            # gather the observations from the active workers
            gathered = await asyncio.gather(*[_drain_to_obs_or_exit(wid) for wid in wids])

            for wid, _obs, is_exit in gathered:
                if is_exit:
                    # remove the finished worker from the active set
                    active.discard(wid)

            batch_pairs: list[tuple[int, ObsMsg]] = [
                (wid, om) for (wid, om, ex) in gathered if not ex and om is not None
            ]
            batch_pairs.sort(key=lambda x: x[0])

            if not batch_pairs:
                if not active:
                    break
                raise RuntimeError("internal error: no observations to batch but workers are still active")

            batch_items = [om.payload for _wid, om in batch_pairs]
            batch_workers = [wid for wid, _om in batch_pairs]
            batch_size = len(batch_items)
            # batch_size is often < n_workers as workers finish rollouts and exit.

            batch_payload = {"batch": True, "items": batch_items}
            await websocket.send(msgpack.packb(batch_payload, use_bin_type=True))
            raw = await websocket.recv()
            actions_batch = msgpack.unpackb(raw, raw=False)
            actions_batch = _normalize_batched_actions_response(actions_batch, batch_size)

            for wid, act in zip(batch_workers, actions_batch, strict=False):
                from_mains[wid].put(act)


async def run_policy_loop_threaded(
    *,
    ws_uri: str,
    env_name: str,
    split,
    start_seed: int,
    num_rollouts: int,
    num_parallel: int,
    output_dir: str | None,
    jpeg_quality: int,
    max_episode_steps: int | None,
    render: bool = False,
) -> None:
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be >= 1")
    if num_parallel < 1:
        raise ValueError("num_parallel must be >= 1")

    # number of threads to be created
    n_workers = min(num_parallel, num_rollouts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = output_dir or f"{env_name}_async_{timestamp}"
    os.makedirs(main_dir, exist_ok=True)

    print(
        f"Output directory: {main_dir!r} — {num_rollouts} rollout(s), "
        f"{n_workers} parallel worker thread(s), seeds {start_seed}..{start_seed + num_rollouts - 1}"
    )

    # queue to store the rollout indices
    rollout_queue: queue.Queue[int | None] = queue.Queue()
    for i in range(num_rollouts):
        rollout_queue.put(i)

    # queues to send messages from the coordinator to the workers and from the workers to the coordinator
    to_mains: list[queue.Queue[WorkerToMain]] = [queue.Queue() for _ in range(n_workers)]
    from_mains: list[queue.Queue[Any]] = [queue.Queue() for _ in range(n_workers)]

    # dictionary to store the results by rollout index
    results_by_rollout: dict[int, tuple[int, bool]] = {}
    # lock to synchronize access to the results dictionary
    results_lock = threading.Lock()
    # holder for the action dimension
    action_dim_holder: list[int | None] = [None]
    # lock to synchronize access to the action dimension
    action_dim_lock = threading.Lock()
    # list to store the threads
    threads: list[threading.Thread] = []
    for wid in range(n_workers):
        t = threading.Thread(
            target=_worker_loop,
            kwargs={
                "rollout_queue": rollout_queue,
                "to_main": to_mains[wid],
                "from_main": from_mains[wid],
                "env_name": env_name,
                "split": split,
                "start_seed": start_seed,
                "main_dir": main_dir,
                "jpeg_quality": jpeg_quality,
                "max_episode_steps": max_episode_steps,
                "render": render,
                "action_dim_holder": action_dim_holder,
                "action_dim_lock": action_dim_lock,
            },
            name=f"robocasa-env-{wid}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    await _run_coordinator(
        ws_uri=ws_uri,
        n_workers=n_workers,
        to_mains=to_mains,
        from_mains=from_mains,
        results_by_rollout=results_by_rollout,
        results_lock=results_lock,
    )

    for t in threads:
        t.join(timeout=600.0)
        if t.is_alive():
            raise RuntimeError(f"Worker thread {t.name!r} did not exit in time")

    ad = action_dim_holder[0]
    if ad is not None:
        print(
            f"RoboCasa env={env_name!r} split={split!r} action_dim={ad} "
            f"cameras={list(DEFAULT_CAMERA_NAMES)} "
            f"({DEFAULT_CAMERA_WIDTH}x{DEFAULT_CAMERA_HEIGHT})"
        )

    rollout_records: list[dict[str, Any]] = []
    for ridx in range(num_rollouts):
        if ridx not in results_by_rollout:
            raise RuntimeError(f"Missing result for rollout index {ridx}")
        length, success = results_by_rollout[ridx]
        seed = start_seed + ridx
        rollout_records.append(
            {
                "env_name": env_name,
                "seed": seed,
                "length": length,
                "success": success,
            }
        )
        print(f"Rollout {ridx + 1}/{num_rollouts} seed={seed} length={length} success={success}")

    summary_path = os.path.join(main_dir, "rollouts.json")
    summary = {
        "env_name": env_name,
        "start_seed": start_seed,
        "num_rollouts": num_rollouts,
        "num_parallel_envs": n_workers,
        "output_directory": os.path.abspath(main_dir),
        "rollouts": rollout_records,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "env_name",
        metavar="ENV_NAME",
        help="RoboCasa kitchen task (registered class name), same as client.py",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("ROBOCASA_POLICY_HOST", "localhost"),
        help=(
            "Policy server hostname or IP (default: localhost). "
            "Use a real host — not the literal word HOST from examples."
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ROBOCASA_POLICY_PORT", "8765")),
        help="Policy server port (or set ROBOCASA_POLICY_PORT)",
    )
    p.add_argument(
        "--split",
        default="all",
        choices=[None, "all", "pretrain", "target"],
        help="Dataset split passed to create_env (default: all)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for rollout index 0; rollout i uses seed + i",
    )
    p.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Total number of episodes (rollouts) to run",
    )
    p.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Number of parallel environment threads (capped at num-rollouts)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for rollouts.json and per-rollout video subfolders",
    )
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 0-100")
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=1500,
        help="Cap steps per episode (in addition to env success)",
    )
    p.add_argument("--render", action="store_true", help="Render onscreen (no videos)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_rollouts < 1:
        raise SystemExit("error: --num-rollouts must be >= 1")
    if args.num_parallel < 1:
        raise SystemExit("error: --num-parallel must be >= 1")
    host = args.host.strip()
    if host.lower() == "host":
        raise SystemExit(
            "error: --host must be a real hostname or IP (e.g. localhost or 127.0.0.1), "
            "not the placeholder HOST."
        )
    uri = f"ws://{host}:{args.port}"
    asyncio.run(
        run_policy_loop_threaded(
            ws_uri=uri,
            env_name=args.env_name,
            split=args.split,
            start_seed=args.seed,
            num_rollouts=args.num_rollouts,
            num_parallel=args.num_parallel,
            output_dir=args.output_dir,
            jpeg_quality=args.jpeg_quality,
            max_episode_steps=args.max_episode_steps,
            render=args.render,
        )
    )


if __name__ == "__main__":
    main()
