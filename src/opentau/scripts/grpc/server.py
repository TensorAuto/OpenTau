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

Usage:
    python src/opentau/scripts/grpc/server.py --config_path=/path/to/config.json \\
        --server.port=50051 --server.max_workers=4
"""

import io
import logging
import traceback
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

import grpc
from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.scripts.grpc import robot_inference_pb2, robot_inference_pb2_grpc
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    init_logging,
)

logger = logging.getLogger(__name__)


class RobotPolicyServicer(robot_inference_pb2_grpc.RobotPolicyServiceServicer):
    """gRPC servicer implementing the RobotPolicyService."""

    def __init__(self, cfg: TrainPipelineConfig):
        """Initialize the servicer with model and configuration.

        Args:
            cfg: Training pipeline configuration including policy settings.
        """
        self.cfg = cfg
        self.device = auto_torch_device()
        self.dtype = torch.bfloat16

        logger.info(f"Initializing policy on device: {self.device}")

        # Load the policy model
        self._load_policy()

    def _load_policy(self):
        """Load the policy model from pretrained weights."""
        logger.info(f"Loading policy from: {self.cfg.policy.pretrained_path}")

        policy_class = get_policy_class(self.cfg.policy.type)
        self.policy = policy_class.from_pretrained(self.cfg.policy.pretrained_path, config=self.cfg.policy)
        self.policy.to(device=self.device, dtype=self.dtype)
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
        if request.prefix_action:
            prefix_action = torch.tensor(
                np.array(
                    [[av.values for av in request.prefix_action]],
                    dtype=np.float32,
                ),
                dtype=self.dtype,
                device=self.device,
            )

            if prefix_action.shape[2] < self.cfg.max_action_dim:
                prefix_action = F.pad(
                    prefix_action,
                    (0, self.cfg.max_action_dim - prefix_action.shape[2]),
                )

            prefix_action = F.pad(
                prefix_action,
                (0, 0, 0, self.cfg.action_chunk - prefix_action.shape[1]),
            )
            action_prefix = prefix_action
            delay = request.delay
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
        import time

        start_time = time.perf_counter()
        response = robot_inference_pb2.ActionChunkResponse()
        response.request_id = request.request_id
        response.timestamp_ns = time.time_ns()

        try:
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


def serve(cfg: TrainPipelineConfig):
    """Start the gRPC server.

    Args:
        cfg: Training pipeline configuration including server settings.
    """
    server_cfg = cfg.server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_cfg.max_workers),
        options=[
            ("grpc.max_send_message_length", server_cfg.max_send_message_length),
            ("grpc.max_receive_message_length", server_cfg.max_receive_message_length),
        ],
    )

    servicer = RobotPolicyServicer(cfg)
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
