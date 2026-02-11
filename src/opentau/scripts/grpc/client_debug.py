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

"""Minimal gRPC client for debugging the inference server (no ROS2).

Calls GetActionChunk with dummy observations so you can test server.py
without setting up a ROS2 environment.

Usage:
    python -m opentau.scripts.grpc.client_debug --server_address localhost:50051
"""

import argparse
import io
import sys
import time

import numpy as np
from PIL import Image

import grpc
from opentau.scripts.grpc import robot_inference_pb2, robot_inference_pb2_grpc


def make_dummy_request(
    num_state: int = 20,
    num_images: int = 1,
    image_size: tuple[int, int] = (224, 224),
    prompt: str = "pick up the red block",
    request_id: str | None = None,
) -> robot_inference_pb2.ObservationRequest:
    """Build an ObservationRequest with dummy values."""
    request = robot_inference_pb2.ObservationRequest()
    request.request_id = request_id or f"debug_{int(time.time_ns())}"
    request.timestamp_ns = time.time_ns()
    request.prompt = prompt
    request.delay = 0

    # Dummy robot state (required by server)
    request.robot_state.state.extend([0.0] * num_state)

    # Dummy JPEG image(s) – server will pad if it expects more cameras
    for _ in range(num_images):
        arr = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        img = robot_inference_pb2.CameraImage()
        img.image_data = buf.getvalue()
        img.encoding = "jpeg"
        request.images.append(img)

    return request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug client: call GetActionChunk with dummy values (no ROS2)."
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="Server address (host:port)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--health_only",
        action="store_true",
        help="Only run health check and exit",
    )
    parser.add_argument(
        "--num_state",
        type=int,
        default=20,
        help="Length of dummy state vector",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of dummy camera images to send",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="pick up the red block",
        help="Language prompt for the policy",
    )
    args = parser.parse_args()

    channel = grpc.insecure_channel(
        args.server_address,
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    stub = robot_inference_pb2_grpc.RobotPolicyServiceStub(channel)

    try:
        # Health check first
        health = stub.HealthCheck(
            robot_inference_pb2.HealthCheckRequest(),
            timeout=args.timeout,
        )
        print(f"Health: {health.healthy} | model={health.model_name} | device={health.device}")
        if not health.healthy:
            print("Server reported unhealthy.")
            return 1
        if args.health_only:
            return 0

        # GetActionChunk with dummy request
        request = make_dummy_request(
            num_state=args.num_state,
            num_images=args.num_images,
            prompt=args.prompt,
        )
        t0 = time.perf_counter()
        response = stub.GetActionChunk(request, timeout=args.timeout)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"GetActionChunk OK (client elapsed={elapsed:.1f} ms)")
        print(f"  request_id={response.request_id}")
        print(f"  inference_time_ms={response.inference_time_ms:.2f}")
        print(f"  action_chunk: {len(response.action_chunk)} steps")
        if response.action_chunk:
            dim = len(response.action_chunk[0].values)
            print(f"  action_dim={dim}")
            # Print first step as sample
            first = np.array(response.action_chunk[0].values, dtype=np.float32)
            print(f"  first step sample: min={first.min():.3f} max={first.max():.3f} mean={first.mean():.3f}")
        return 0

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}", file=sys.stderr)
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    sys.exit(main())
