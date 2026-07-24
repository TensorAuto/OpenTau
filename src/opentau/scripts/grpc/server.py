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

"""gRPC server for robot policy inference on GPU.

This server loads an ML policy model and serves inference requests from
robots running ROS 2. Designed to run on a server with a ML GPU.

Optionally (``--planner.enabled=true``) it also spins up a Gemini Robotics-ER
high-level planner that runs asynchronously alongside the VLA policy:

- The request ``prompt`` is treated as the *overall task*. The planner consumes
  the latest cached observation (images, state), the task and its own
  memory-as-language string, and produces the subtask the VLA policy is
  actually conditioned on.
- The planner runs on its own free-running background loop, replanning every
  ``planner.interval_s`` wall-clock seconds — fully decoupled from request
  arrival. It skips a cycle when no new observation has arrived since the last
  plan, so an idle server makes no API calls.
- The VLA inference path never blocks on the planner — it reads whatever
  subtask is currently available. The one exception is the start of a task
  (the first request, or the first after the prompt changes), which blocks
  (up to ``planner.first_plan_timeout_s``) so the policy starts on a real
  subtask rather than the raw task.

With the planner disabled (the default) the server serves the VLA policy only.
The wire API is identical either way.

Usage:
    python src/opentau/scripts/grpc/server.py --config_path=/path/to/config.json \\
        --server.port=50051 --server.max_workers=4

    # with the high-level planner
    GEMINI_API_KEY=... python src/opentau/scripts/grpc/server.py \\
        --config_path=/path/to/config.json \\
        --server.port=50051 --planner.enabled=true --planner.interval_s=5
"""

import io
import logging
import threading
import time
import traceback
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from PIL import Image

import grpc
from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.planner.gemini_er_planner import GeminiERPlanner
from opentau.policies.factory import get_policy_class
from opentau.policies.utils import to_dtype_preserving_siglip_float32
from opentau.scripts.grpc import auth, robot_inference_pb2, robot_inference_pb2_grpc
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    init_logging,
)

logger = logging.getLogger(__name__)

RequestHook = Callable[[str], None]


def _noop_request_hook(_method: str) -> None:
    """Default request hook used when an embedding application supplies none."""


class RobotPolicyServicer(robot_inference_pb2_grpc.RobotPolicyServiceServicer):
    """gRPC servicer implementing the RobotPolicyService.

    When ``cfg.planner.enabled`` is set, a high-level planner runs alongside
    the policy on its own free-running daemon thread, replanning periodically
    from the latest cached observation. Shared planner state (latest
    observation, task, subtask, memory) is guarded by a single lock; the
    network-bound Gemini call happens outside the lock, so it never stalls
    policy inference.
    """

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        request_hook: RequestHook | None = None,
    ):
        """Initialize the servicer with model and configuration.

        Args:
            cfg: Training pipeline configuration including policy, server and
                planner settings.
            request_hook: Optional non-blocking callback invoked for every
                accepted inference request. Hook failures are logged and never
                fail the RPC.
        """
        self.cfg = cfg
        self._request_hook = request_hook or _noop_request_hook
        self.device = auto_torch_device()
        self.dtype = torch.bfloat16

        logger.info(f"Initializing policy on device: {self.device}")

        # Load the policy model
        self._load_policy()

        # Optionally spin up the high-level planner
        self._init_planner(cfg.planner)

    def _notify_request(self, method: str) -> None:
        """Run the embedding application's request hook without affecting RPCs."""
        try:
            self._request_hook(method)
        except Exception:
            logger.exception("gRPC request hook failed for %s", method)

    def _init_planner(self, pcfg):
        """Set up the high-level planner and start its background loop (if enabled)."""
        self._planner: GeminiERPlanner | None = None
        self._planner_thread: threading.Thread | None = None

        self._interval_s = pcfg.interval_s
        self._first_plan_timeout_s = pcfg.first_plan_timeout_s
        self._include_state = pcfg.include_state

        self._lock = threading.Lock()
        # Shared state, all guarded by self._lock.
        self._latest_images: list[tuple[bytes, str]] = []
        self._latest_state: list[float] | None = None
        self._task: str | None = None
        self._subtask: str | None = None
        self._memory: str = ""
        # Incremented whenever a request caches a fresh observation; the loop
        # only replans when there is something new to plan on, so an idle
        # server makes no planner API calls.
        self._obs_seq: int = 0
        self._last_planned_seq: int = 0
        # Bumped on task change so a stale in-flight plan cannot commit.
        self._plan_generation: int = 0

        # Loop control. ``_subtask_ready`` is set when a plan for the current
        # task has been committed (start-of-inference requests wait on it);
        # ``_replan_event`` short-circuits the interval sleep on task change.
        self._stop_event = threading.Event()
        self._subtask_ready = threading.Event()
        self._replan_event = threading.Event()

        if pcfg.enabled:
            # Fails fast here if no API key is resolvable.
            self._planner = GeminiERPlanner(
                model=pcfg.model,
                api_key_env=pcfg.api_key_env,
                max_output_tokens=pcfg.max_output_tokens,
                temperature=pcfg.temperature,
                system_prompt_key=pcfg.system_prompt_key,
                user_prompt_key=pcfg.user_prompt_key,
            )
            self._planner_thread = threading.Thread(target=self._planner_loop, name="planner", daemon=True)
            self._planner_thread.start()
            logger.info(
                f"High-level planner enabled: model={pcfg.model}, interval_s={pcfg.interval_s}, "
                f"first_plan_timeout_s={pcfg.first_plan_timeout_s}"
            )

    def _update_shared_state(self, request: robot_inference_pb2.ObservationRequest) -> None:
        """Cache the latest observation; reset planner state on task change."""
        with self._lock:
            self._latest_images = [(img.image_data, img.encoding) for img in request.images]
            self._latest_state = list(request.robot_state.state) or None
            self._obs_seq += 1
            if request.prompt != self._task:
                logger.info(f"New task: {request.prompt!r} — resetting planner memory")
                self._task = request.prompt
                self._subtask = None
                self._memory = ""
                self._plan_generation += 1
                self._subtask_ready.clear()
                self._replan_event.set()

    def _planner_loop(self) -> None:
        """Free-running planner: replan every ``interval_s`` on fresh observations.

        Runs fully decoupled from request handling — requests only cache the
        latest observation; this loop periodically turns it into a subtask.
        """
        idle_poll_s = 0.05
        while not self._stop_event.is_set():
            with self._lock:
                has_new_obs = self._obs_seq != self._last_planned_seq
            if not has_new_obs:
                # Nothing new to plan on (no robot connected, or it stopped
                # sending) — idle cheaply instead of burning API calls.
                self._stop_event.wait(idle_poll_s)
                continue
            # The plan below runs on the freshest state, so it satisfies any
            # replan request raised up to this point.
            self._replan_event.clear()
            self._run_plan()
            # Sleep out the planning interval; a task change (or shutdown)
            # short-circuits it so a new task replans immediately.
            self._replan_event.wait(timeout=self._interval_s)

    def _run_plan(self) -> None:
        """One planner step: snapshot state, call the model, commit the result."""
        with self._lock:
            task = self._task
            images = list(self._latest_images)
            state = self._latest_state if self._include_state else None
            memory = self._memory
            generation = self._plan_generation
            self._last_planned_seq = self._obs_seq
        try:
            # Network-bound call, deliberately outside the lock.
            result = self._planner.plan(task=task, images=images, state=state, memory=memory)
            with self._lock:
                if self._plan_generation == generation:
                    self._subtask = result.subtask
                    self._memory = result.memory
                    self._subtask_ready.set()
            logger.info(
                f"Planner ({result.latency_s:.2f}s) subtask={result.subtask!r} memory={result.memory!r}"
            )
        except Exception:
            # A planner failure must never crash the server or clear an
            # existing subtask — a stale subtask beats no subtask.
            logger.exception("High-level planner call failed; keeping previous subtask")

    def _current_subtask(self) -> str | None:
        with self._lock:
            return self._subtask

    def _apply_planner(self, request: robot_inference_pb2.ObservationRequest) -> None:
        """Rewrite ``request.prompt`` to the planner's current subtask.

        No-op when the planner is disabled. The request prompt is treated as
        the overall task; the policy is conditioned on the subtask instead.
        Non-blocking except at the start of inference for a task (subtask not
        yet available), where it waits for the planner loop's first plan.
        """
        if self._planner is None:
            return

        self._update_shared_state(request)
        # Start of inference for this task (no subtask yet): wait for the
        # initial subtask so the policy starts on a real subtask rather than
        # the raw task.
        if self._current_subtask() is None and not self._subtask_ready.wait(
            timeout=self._first_plan_timeout_s
        ):
            logger.warning(
                f"First plan not ready after {self._first_plan_timeout_s}s; "
                "falling back to the raw task prompt"
            )

        subtask = self._current_subtask()
        if subtask is not None:
            request.prompt = subtask

    def close(self) -> None:
        """Stop the planner loop (an in-flight Gemini call may delay exit briefly)."""
        self._stop_event.set()
        self._replan_event.set()
        if self._planner_thread is not None:
            self._planner_thread.join(timeout=5)

    def _load_policy(self):
        """Load the policy model from pretrained weights."""
        logger.info(f"Loading policy from: {self.cfg.policy.pretrained_path}")

        policy_class = get_policy_class(self.cfg.policy.type)
        self.policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path, config=self.cfg.policy)
        # Preserve the float32-pinned SigLIP embeddings across the bf16 cast (openpi parity);
        # a plain .to(bfloat16) would round them back. Serving is single-process, so this is safe.
        to_dtype_preserving_siglip_float32(self.policy, device=self.device, dtype=self.dtype)
        self.policy.eval()
        self.policy.model.sample_actions = attempt_torch_compile(
            self.policy.model.sample_actions, device_hint=self.device
        )

        self.policy.reset()

        camera_observations = {
            f"camera{i}": torch.zeros((1, 3, *self.cfg.resolution), dtype=self.dtype, device=self.device)
            for i in range(self.cfg.num_cams)
        }
        observation = {
            **camera_observations,
            "state": torch.zeros((1, self.cfg.max_state_dim), dtype=self.dtype, device=self.device),
            "prompt": ["Pick up yellow lego block and put it in the bin"],
            "img_is_pad": torch.zeros((1, 1), dtype=torch.bool, device=self.device),
        }
        # Mirror `_prepare_observation`'s norm-head tagging for the warmup
        # call — otherwise a multi-head checkpoint trips
        # `_resolve_dataset_index` at compile time rather than at the first
        # real request. Prefer (robot_type, control_mode) when both are set.
        warmup_robot_type = getattr(self.cfg.server, "robot_type", None)
        warmup_control_mode = getattr(self.cfg.server, "control_mode", None)
        warmup_dataset_repo_id = getattr(self.cfg.server, "dataset_repo_id", None)
        if warmup_robot_type is not None and warmup_control_mode is not None:
            observation["robot_type"] = warmup_robot_type
            observation["control_mode"] = warmup_control_mode
        elif warmup_dataset_repo_id is not None:
            observation["dataset_repo_id"] = warmup_dataset_repo_id
        action_prefix = torch.zeros(
            (1, self.cfg.action_chunk, self.cfg.max_action_dim), dtype=self.dtype, device=self.device
        )
        delay = torch.tensor(0, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            # One warmup call right after compiling
            # two warmup calls are needed right after compiling
            # the first warmup call is needed for compiling
            # the second warmup call is needed for kernel autotuning
            _ = self.policy.sample_actions(observation, action_prefix=action_prefix, delay=delay)
            _ = self.policy.sample_actions(observation, action_prefix=action_prefix, delay=delay)
            logger.info("Policy loaded successfully")

    def _decode_image(self, camera_image: robot_inference_pb2.CameraImage) -> torch.Tensor:
        """Decode an image from the protobuf message.

        Args:
            camera_image: CameraImage protobuf message.

        Returns:
            Tensor of shape (1, C, H, W) normalized to [0, 1].
        """
        if camera_image.encoding in ["jpeg", "png"]:
            # Decode compressed image
            image = Image.open(io.BytesIO(camera_image.image_data))
            image = image.convert("RGB")
            image = image.resize(self.cfg.resolution[::-1])  # PIL uses (W, H)
            image_array = np.array(image, dtype=np.float32) / 255.0
        elif camera_image.encoding == "raw":
            # Raw image data - assume it's already in the right shape
            image_array = np.frombuffer(camera_image.image_data, dtype=np.float32)
            # Reshape assuming square image with 3 channels
            side = int(np.sqrt(len(image_array) / 3))
            image_array = image_array.reshape(side, side, 3)
            # Resize if needed
            if (side, side) != self.cfg.resolution:
                image = Image.fromarray((image_array * 255).astype(np.uint8))
                image = image.resize(self.cfg.resolution[::-1])
                image_array = np.array(image, dtype=np.float32) / 255.0
        else:
            raise ValueError(f"Unknown image encoding: {camera_image.encoding}")

        # Convert to (C, H, W) tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(device=self.device, dtype=self.dtype)

    def _prepare_observation(
        self, request: robot_inference_pb2.ObservationRequest
    ) -> dict[str, torch.Tensor]:
        """Convert a protobuf observation request to the policy input format.

        Args:
            request: ObservationRequest protobuf message.

        Returns:
            Dictionary of tensors matching the policy's expected input format.
        """
        batch = {}

        # Process camera images
        img_is_pad = []
        for i, camera_image in enumerate(request.images):
            camera_name = f"camera{i}"
            batch[camera_name] = self._decode_image(camera_image)
            img_is_pad.append(False)

        # Fill in missing cameras with zeros
        for i in range(len(request.images), self.cfg.num_cams):
            batch[f"camera{i}"] = torch.zeros(
                (1, 3, *self.cfg.resolution),
                dtype=self.dtype,
                device=self.device,
            )
            img_is_pad.append(True)

        batch["img_is_pad"] = torch.tensor([img_is_pad], dtype=torch.bool, device=self.device)

        # Process robot state
        if request.robot_state.state:
            state = list(request.robot_state.state)
            # Pad to max_state_dim if needed
            if len(state) < self.cfg.max_state_dim:
                state.extend([0.0] * (self.cfg.max_state_dim - len(state)))
            batch["state"] = torch.tensor(
                [state[: self.cfg.max_state_dim]],
                dtype=self.dtype,
                device=self.device,
            )
        else:
            raise ValueError("Robot state is required but was not provided in the request")

        # Process prompt
        batch["prompt"] = [request.prompt] if request.prompt else [""]

        # Tag the batch with the training-time norm head to use for
        # per-sample Normalize/Unnormalize. Prefer the
        # `(robot_type, control_mode)` pair when both are configured (the
        # new per-`(robot_type, control_mode)` aggregation route); else use
        # `dataset_repo_id` (back-compat, also resolves on legacy
        # per-dataset checkpoints). When all are `None`, the policy's
        # `_resolve_dataset_index` single-head fallback handles things.
        server_robot_type = getattr(self.cfg.server, "robot_type", None)
        server_control_mode = getattr(self.cfg.server, "control_mode", None)
        server_dataset_repo_id = getattr(self.cfg.server, "dataset_repo_id", None)
        if server_robot_type is not None and server_control_mode is not None:
            batch["robot_type"] = server_robot_type
            batch["control_mode"] = server_control_mode
        elif server_dataset_repo_id is not None:
            batch["dataset_repo_id"] = server_dataset_repo_id
        if request.prefix_action:
            prefix_action = torch.tensor(
                np.array(
                    [av.values for av in request.prefix_action],
                    dtype=np.float32,
                ),
                dtype=self.dtype,
                device=self.device,
            )

            prefix_action = rearrange(prefix_action, "c d -> 1 c d")

            prefix_action = BaseDataset.pad_vector(prefix_action, self.cfg.max_action_dim)

            prefix_action = F.pad(
                prefix_action,
                (0, 0, 0, self.cfg.action_chunk - prefix_action.shape[1]),
            )
            action_prefix = prefix_action
            delay = torch.tensor(request.delay, dtype=torch.long, device=self.device)
        else:
            action_prefix = torch.zeros(
                (1, self.cfg.action_chunk, self.cfg.max_action_dim), dtype=self.dtype, device=self.device
            )
            delay = torch.tensor(0, dtype=torch.long, device=self.device)

        return batch, action_prefix, delay

    def GetActionChunk(
        self,
        request: robot_inference_pb2.ObservationRequest,
        context: grpc.ServicerContext,
    ) -> robot_inference_pb2.ActionChunkResponse:
        """Handle a single action chunk inference request.

        Args:
            request: ObservationRequest containing observations.
            context: gRPC context.

        Returns:
            ActionChunkResponse containing the predicted action chunk.
        """
        self._notify_request("GetActionChunk")
        start_time = time.perf_counter()
        response = robot_inference_pb2.ActionChunkResponse()
        response.request_id = request.request_id
        response.timestamp_ns = time.time_ns()

        try:
            # Substitute the high-level planner's subtask for the prompt (no-op
            # when the planner is disabled).
            self._apply_planner(request)

            # Prepare observation batch
            batch, action_prefix, delay = self._prepare_observation(request)

            # Run inference
            with torch.inference_mode():
                action_chunk = self.policy.sample_actions(batch, action_prefix=action_prefix, delay=delay)
                # action_chunk shape: (batch_size=1, n_action_steps, action_dim)
                # Remove batch dimension and convert to numpy
                action_chunk = action_chunk.squeeze(0).to("cpu", torch.float32).numpy()

            # Populate 2D action chunk structure
            for action_vector in action_chunk:
                action_vec_msg = robot_inference_pb2.ActionVector()
                action_vec_msg.values.extend(action_vector.tolist())
                response.action_chunk.append(action_vec_msg)

        except ValueError as e:
            # Invalid request (e.g., missing required fields)
            logger.error(f"Invalid request: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        except Exception as e:
            # Unexpected error during inference
            traceback.print_exc()
            logger.exception("Error during inference")
            context.abort(grpc.StatusCode.INTERNAL, f"Inference error: {e}")

        response.inference_time_ms = (time.perf_counter() - start_time) * 1000
        return response

    def StreamActionChunks(
        self,
        request_iterator: Iterator[robot_inference_pb2.ObservationRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[robot_inference_pb2.ActionChunkResponse]:
        """Handle streaming action chunk inference requests.

        Args:
            request_iterator: Iterator of ObservationRequest messages.
            context: gRPC context.

        Yields:
            ActionChunkResponse messages for each observation.
        """
        for request in request_iterator:
            if context.is_active():
                yield self.GetActionChunk(request, context)
            else:
                break

    def HealthCheck(
        self,
        request: robot_inference_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> robot_inference_pb2.HealthCheckResponse:
        """Check server health and GPU status.

        Args:
            request: HealthCheckRequest message.
            context: gRPC context.

        Returns:
            HealthCheckResponse with server status.
        """
        response = robot_inference_pb2.HealthCheckResponse()
        response.healthy = True
        response.status = "Server is running"
        response.model_name = self.cfg.policy.type
        response.device = str(self.device)

        if torch.cuda.is_available():
            response.gpu_memory_used_gb = torch.cuda.memory_allocated() / 1e9
            response.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        return response


def serve(cfg: TrainPipelineConfig, request_hook: RequestHook | None = None):
    """Start the gRPC server.

    Args:
        cfg: Training pipeline configuration including server settings.
        request_hook: Optional callback invoked for each inference request.
    """
    server_cfg = cfg.server

    # Optional API-key auth, activated by the TUNER_INFERENCE_API_KEY env var.
    # When unset, the server keeps its historical no-auth behavior.
    interceptors = []
    auth_interceptor = auth.interceptor_from_env()
    if auth_interceptor is not None:
        interceptors.append(auth_interceptor)
        logger.info("API-key authentication enabled (require %s header)", auth.API_KEY_HEADER)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_cfg.max_workers),
        interceptors=interceptors,
        options=[
            ("grpc.max_send_message_length", server_cfg.max_send_message_length),
            ("grpc.max_receive_message_length", server_cfg.max_receive_message_length),
        ],
    )

    servicer = RobotPolicyServicer(cfg, request_hook=request_hook)
    robot_inference_pb2_grpc.add_RobotPolicyServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{server_cfg.port}")
    server.start()

    logger.info(f"Server started on port {server_cfg.port}")
    logger.info(f"Policy: {cfg.policy.type}")
    logger.info(f"Device: {servicer.device}")
    logger.info(f"Max workers: {server_cfg.max_workers}")
    logger.info("Waiting for requests...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace=5)
        servicer.close()


@parser.wrap()
def server_main(cfg: TrainPipelineConfig):
    """Main entry point for the gRPC server.

    Args:
        cfg: Training pipeline configuration parsed from CLI/config file.
    """
    logging.info(pformat(asdict(cfg)))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    serve(cfg)


if __name__ == "__main__":
    init_logging()
    server_main()
