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

import ctypes
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from multiprocessing import Array, Pipe, Process, SimpleQueue
from multiprocessing.connection import Connection, wait
from pathlib import Path
from pprint import pformat

import numpy as np
import psutil
import torch
from einops import rearrange
from torch.utils.data._utils.collate import default_collate

from opentau.configs import parser
from opentau.configs.libero import TrainConfigWithLiberoEval
from opentau.policies.factory import get_policy_class
from opentau.utils.libero import LiberoObservationRecorder, summarize_libero_results
from opentau.utils.libero import _libero2np as libero2np
from opentau.utils.libero import _np2torch as np2torch
from opentau.utils.monkey_patch import gym_is_gymnasium_patch
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device

# Sent by client process to indicate simulation completion and signal that the pipe is to be closed
SENTINEL = "<SENTINEL>"

LIBERO_ACTION_DIM = 7


@dataclass
class Config(TrainConfigWithLiberoEval):
    parallel_simulation_count: int = 4
    max_wait_sec: float = 1.0
    logging_dir: str | None = None


@dataclass
class Request:
    r"""Request sent from the CPU LIBERO simulation process to the GPU policy."""

    sim_id: int
    step_id: int
    observation: dict[str, np.ndarray | str]


@dataclass
class Response:
    r"""Response sent from the GPU policy to the CPU LIBERO simulation process."""

    chunked_action: np.ndarray


class ConnectionBuffer:
    def __init__(
        self,
        conns: list[Connection],
        max_wait_sec: float,
        max_batch_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        r"""Gathers a batch of inputs. Wait for no more than `max_wait_time` seconds,
        or until `max_batch_size` is reached."""
        self.conns = conns
        self.max_wait = max_wait_sec
        self.max_batch = max_batch_size
        self.device = device
        self.dtype = dtype
        self.batch_inputs = []
        self.response_list = []
        self.last_yield_time = None

    def _should_yield(self):
        # Don't yield empty batches
        if not self.batch_inputs:
            return False

        return (
            len(self.batch_inputs) >= self.max_batch
            or time.monotonic() - self.last_yield_time >= self.max_wait
            or not self.conns
        )

    def get_batch(self):
        self.last_yield_time = time.monotonic()

        while self.conns or self.batch_inputs:
            timeout = self.last_yield_time + self.max_wait - time.monotonic()
            selected = wait(self.conns, timeout=max(timeout, 0.0)) if self.conns else []
            for ready in selected:
                try:
                    req = ready.recv()
                    if req != SENTINEL:
                        xs = np2torch(req.observation, self.device, self.dtype)
                except Exception as e:  # In case the simulation process crashed
                    logging.error(str(e))
                    req = SENTINEL

                if req == SENTINEL:
                    logging.debug("Removing connection")
                    self.conns.remove(ready)
                    ready.close()
                    continue

                logging.debug(f"Received a request from sim {req.sim_id} at step {req.step_id}")

                self.batch_inputs.append(xs)
                self.response_list.append(ready)
                if self._should_yield():
                    break

            if self._should_yield():
                bi, br = self.batch_inputs, self.response_list
                self.batch_inputs, self.response_list = [], []
                self.last_yield_time = time.monotonic()
                yield bi, br


def start_parent_check_thread():
    def is_process_active(pid):
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def kill_child_processes(parent_pid):
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            try:
                os.kill(child.pid, signal.SIGKILL)
                logging.warning(f"Killed pid {child.pid}")
            except BaseException as e:
                logging.warning(f"Killing pid {child.pid} failed {str(e)}")

    def check_parent_alive():
        parent_pid = os.getppid()
        while True:
            if not is_process_active(parent_pid):
                logging.warning(f"Parent is dead, kill self {os.getpid()}")
                kill_child_processes(os.getpid())
                os.kill(os.getpid(), signal.SIGKILL)

            time.sleep(10)

    thread = threading.Thread(target=check_parent_alive, daemon=True)
    thread.start()


def server(cfg: Config, conns: list[Connection], device: str, dtype: torch.dtype):
    r"""Runs a server in the main process that creates a policy and listens for observations from clients"""
    init_proc_logging(None, cfg)
    logging.info(pformat(asdict(cfg)))

    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=dtype)
    policy.eval()

    connection_buffer = ConnectionBuffer(
        conns,
        max_wait_sec=cfg.max_wait_sec,
        max_batch_size=cfg.batch_size,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for batch_inputs, resp_conns in connection_buffer.get_batch():
            if not batch_inputs:
                logging.debug("Got empty batch, continuing.")
                continue
            logging.debug(f"Received batch of size {len(batch_inputs)}")
            batch_inputs = default_collate(batch_inputs)
            # We return the entire action chunk and let the simulation process handle the caching.
            batch_chunked_actions = policy.sample_actions(batch_inputs)
            batch_chunked_actions = rearrange(
                batch_chunked_actions, "chunk batch action -> batch chunk action"
            )
            batch_chunked_actions = batch_chunked_actions.numpy(force=True)
            batch_chunked_actions = batch_chunked_actions[:, : cfg.libero.chunk_usage, :LIBERO_ACTION_DIM]
            # gripper open/close should be -1 or 1
            batch_chunked_actions[:, :, -1] = 2.0 * (batch_chunked_actions[:, :, -1] > 0) - 1.0

            for chunked_actions, conn in zip(batch_chunked_actions, resp_conns, strict=True):
                resp = Response(chunked_action=chunked_actions)
                logging.debug(f"sending action of shape {resp.chunked_action.shape} to simulation")
                conn.send(resp)


def simulation(worker_id: int, cfg: Config, job_q: SimpleQueue, results_arr: Array, conn: Connection):
    r"""Runs a simulation in a separate process. Sends observations to the server and receives actions."""
    init_proc_logging(worker_id, cfg)
    start_parent_check_thread()

    # Patch gym before importing OffScreenRenderEnv at the start of the sim process.
    gym_is_gymnasium_patch()
    from libero.libero.envs import OffScreenRenderEnv

    init_states = cfg.libero.init_states
    while True:
        sim_id = job_q.get()
        if sim_id == SENTINEL:
            logging.debug(f"Simulation process {os.getpid()} received SENTINEL, exiting.")
            conn.send(SENTINEL)
            conn.close()
            return

        # This environment provides interaction with the policy without rendering a UI.
        # To record videos, we use the `LiberoObservationRecorder` class and manually record frames.
        env = OffScreenRenderEnv(
            bddl_file_name=cfg.libero.bddl_file,
            camera_heights=cfg.resolution[0],
            camera_widths=cfg.resolution[1],
        )
        env.seed(sim_id)
        env.set_init_state(init_states[sim_id % len(init_states)])
        video_root = cfg.libero.video_dir and (
            Path(cfg.libero.video_dir) / cfg.libero.suite / str(cfg.libero.id) / str(sim_id)
        )
        camera_names = ["agentview_image", "robot0_eye_in_hand_image"]
        with LiberoObservationRecorder(video_root, camera_names=camera_names) as recorder:
            obs = env.reset()
            # Warm up the environment with a few no-op steps
            for _ in range(5):
                obs, *_ = env.step([0.0] * LIBERO_ACTION_DIM)
            recorder.record(obs)
            action_cache = []

            finish_step = -1
            for step_id in range(1, cfg.libero.max_steps + 1):
                if len(action_cache) == 0:
                    req = Request(sim_id=sim_id, step_id=step_id, observation=libero2np(obs, cfg))
                    logging.debug(f"Sending observation at step {step_id}")
                    conn.send(req)
                    resp = conn.recv()
                    logging.debug(f"Received action chunk with shape: {resp.chunked_action.shape}")
                    action_cache = deque(resp.chunked_action)

                action = action_cache.popleft()
                obs, reward, done, info = env.step(action)
                recorder.record(obs)

                logging.debug(f"Step: {step_id}, Reward: {reward}, Done: {done}, Info: {info}")
                if done or reward > 0:
                    finish_step = step_id
                    break

        logging.info(f"Result is {finish_step=}")

        if sim_id > len(results_arr):
            # Should never happen
            logging.error(f"sim_id {sim_id} exceeds results array size {len(results_arr)}")

        results_arr[sim_id] = finish_step


def init_proc_logging(worker_id: int | None, cfg: Config):
    r"""Initialize logging for server or worker processes."""
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    if cfg.logging_dir is not None:
        filename = f"worker_{worker_id:03d}.log" if worker_id is not None else "server.log"
        directory = Path(cfg.logging_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(directory / filename))

    prefix = "SERVER" if worker_id is None else f"WORKER-{worker_id:03d}"
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format=f"{prefix}: %(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info(f"Initialized in process {os.getpid()} by parent {os.getppid()}")


@parser.wrap()
def main(cfg: Config):
    device = auto_torch_device()
    dtype = torch.bfloat16

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # job queue contains simulation IDs to be processed, and `SENTINEL`s to signal completion
    job_queue = SimpleQueue()
    for sim_id in range(cfg.libero.n_simulations):
        job_queue.put(sim_id)
    for _ in range(cfg.parallel_simulation_count):
        job_queue.put(SENTINEL)

    # Shared memory mapping for results. Since each simulation is only handled by one process, no lock is needed.
    # -2 indicates uninitialized, -1 indicates failure to complete the task.
    results_arr = Array(ctypes.c_int64, [-2] * cfg.libero.n_simulations, lock=False)

    sim_procs, conns = [], []
    for worker_id in range(cfg.parallel_simulation_count):
        server_conn, client_conn = Pipe()
        conns.append(server_conn)

        # TODO ensure p is killed if the main process is killed
        # TODO ensure that when p is killed, the client_conn is closed
        p = Process(
            target=simulation,
            args=(
                worker_id,
                cfg,  # cfg must be unpickle-able in sub-processes
                job_queue,
                results_arr,
                client_conn,
            ),
        )
        sim_procs.append(p)

        p.start()  # Start the process before closing the client connection
        client_conn.close()

    server(cfg, conns, device, dtype)

    logging.debug("Joining simulation processes...")
    for p in sim_procs:
        p.join()

    logging.debug("All simulations completed. Gathering results...")
    summary = summarize_libero_results(results_arr[:])
    logging.info(str(summary))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
