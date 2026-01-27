#!/usr/bin/env python

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

"""gRPC client for robot policy inference.

This client runs on the robot and sends observations to a remote gRPC server
for ML inference. Designed to integrate with ROS 2 Humble.

Usage:
    # Standalone test
    python src/opentau/scripts/grpc/client.py --server_address 192.168.1.100:50051

    # With ROS 2 (see ROS2PolicyClient class)
"""

import argparse
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

import grpc

# Import generated protobuf classes
from opentau.scripts.grpc import robot_inference_pb2, robot_inference_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for the gRPC client."""

    server_address: str = "localhost:50051"
    timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_delay_seconds: float = 0.1
    image_encoding: str = "jpeg"  # "jpeg", "png", or "raw"
    jpeg_quality: int = 85


class PolicyClient:
    """gRPC client for communicating with the policy inference server."""

    def __init__(self, config: ClientConfig):
        """Initialize the client.

        Args:
            config: Client configuration.
        """
        self.config = config
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[robot_inference_pb2_grpc.RobotPolicyServiceStub] = None
        self._connected = False
        self._request_counter = 0

    def connect(self) -> bool:
        """Establish connection to the server.

        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            self._channel = grpc.insecure_channel(
                self.config.server_address,
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                ],
            )
            self._stub = robot_inference_pb2_grpc.RobotPolicyServiceStub(self._channel)

            # Test connection with health check
            response = self._stub.HealthCheck(
                robot_inference_pb2.HealthCheckRequest(),
                timeout=self.config.timeout_seconds,
            )
            self._connected = response.healthy
            logger.info(
                f"Connected to server: {self.config.server_address}, "
                f"model: {response.model_name}, device: {response.device}"
            )
            return self._connected

        except grpc.RpcError as e:
            logger.error(f"Failed to connect to server: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Close the connection to the server."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False
            logger.info("Disconnected from server")

    def is_connected(self) -> bool:
        """Check if the client is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected and self._channel is not None

    def _encode_image(self, image: np.ndarray) -> robot_inference_pb2.CameraImage:
        """Encode an image for transmission.

        Args:
            image: Image array of shape (H, W, C) with values in [0, 255] or [0, 1].

        Returns:
            CameraImage protobuf message.
        """
        # Normalize image to [0, 255] uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        camera_image = robot_inference_pb2.CameraImage()

        if self.config.image_encoding == "jpeg":
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
            camera_image.image_data = buffer.getvalue()
            camera_image.encoding = "jpeg"
        elif self.config.image_encoding == "png":
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            camera_image.image_data = buffer.getvalue()
            camera_image.encoding = "png"
        else:  # raw
            camera_image.image_data = image.astype(np.float32).tobytes()
            camera_image.encoding = "raw"

        return camera_image

    def get_action_chunk(
        self,
        images: list[np.ndarray],
        state: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray, float]:
        """Get action chunk from the policy server.

        Args:
            images: List of image arrays (H, W, C) for each camera.
            state: Robot state vector.
            prompt: Language instruction.

        Returns:
            Tuple of (action chunk array, inference time in ms).

        Raises:
            RuntimeError: If not connected or inference fails.
        """
        if not self.is_connected():
            raise RuntimeError("Client is not connected to server")

        self._request_counter += 1
        request = robot_inference_pb2.ObservationRequest()
        request.request_id = f"req_{self._request_counter}"
        request.timestamp_ns = time.time_ns()
        request.prompt = prompt

        # Add images
        for image in images:
            camera_image = self._encode_image(image)
            request.images.append(camera_image)

        # Add state
        request.robot_state.state.extend(state.flatten().tolist())

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._stub.GetActionChunk(request, timeout=self.config.timeout_seconds)

                action_chunk = np.array(response.action_chunk, dtype=np.float32)
                return action_chunk, response.inference_time_ms

            except grpc.RpcError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)

        raise RuntimeError(f"Failed after {self.config.max_retries} retries: {last_error}")

    def health_check(self) -> dict:
        """Check server health.

        Returns:
            Dictionary with health status information.
        """
        if not self._stub:
            return {"healthy": False, "status": "Not connected"}

        try:
            response = self._stub.HealthCheck(
                robot_inference_pb2.HealthCheckRequest(),
                timeout=self.config.timeout_seconds,
            )
            return {
                "healthy": response.healthy,
                "status": response.status,
                "model_name": response.model_name,
                "device": response.device,
                "gpu_memory_used_gb": response.gpu_memory_used_gb,
                "gpu_memory_total_gb": response.gpu_memory_total_gb,
            }
        except grpc.RpcError as e:
            return {"healthy": False, "status": str(e)}


# =============================================================================
# ROS 2 Integration
# =============================================================================

# ROS 2 imports are optional - only imported when ROS2PolicyClient is used
ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String

    ROS2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ROS2Config:
    """Configuration for the ROS 2 policy client node."""

    # gRPC settings
    server_address: str = "localhost:50051"
    timeout_seconds: float = 2.0

    # Topic names
    camera_topics: list[str] = field(default_factory=lambda: ["/camera0/image_raw", "/camera1/image_raw"])
    state_topic: str = "/joint_states"
    prompt_topic: str = "/policy/prompt"
    action_topic: str = "/policy/action"

    # Control settings
    control_frequency_hz: float = 10.0
    default_prompt: str = ""


if ROS2_AVAILABLE:

    class ROS2PolicyClient(Node):
        """ROS 2 node that interfaces with the gRPC policy server.

        This node subscribes to camera images and robot state, sends them to
        the gRPC server, and publishes the resulting action chunks.

        Example usage:
            ```python
            import rclpy
            from opentau.grpc.client import ROS2PolicyClient, ROS2Config

            rclpy.init()
            config = ROS2Config(
                server_address="192.168.1.100:50051",
                camera_topics=["/camera/color/image_raw"],
                control_frequency_hz=30.0,
            )
            node = ROS2PolicyClient(config)
            rclpy.spin(node)
            ```
        """

        def __init__(self, config: ROS2Config):
            """Initialize the ROS 2 policy client node.

            Args:
                config: ROS 2 configuration.
            """
            super().__init__("policy_client")
            self.config = config
            self.cv_bridge = CvBridge()

            # Initialize gRPC client
            client_config = ClientConfig(
                server_address=config.server_address,
                timeout_seconds=config.timeout_seconds,
            )
            self.policy_client = PolicyClient(client_config)

            # State storage - use list to maintain camera order
            self._latest_images: list[Optional[np.ndarray]] = [None] * len(config.camera_topics)
            self._latest_state: Optional[np.ndarray] = None
            self._current_prompt: str = config.default_prompt

            # Create subscribers
            self._image_subs = []
            for i, topic in enumerate(config.camera_topics):
                sub = self.create_subscription(
                    RosImage,
                    topic,
                    lambda msg, idx=i: self._image_callback(msg, idx),
                    10,
                )
                self._image_subs.append(sub)
                self.get_logger().info(f"Subscribed to {topic} as camera{i}")

            self._state_sub = self.create_subscription(
                JointState,
                config.state_topic,
                self._state_callback,
                10,
            )

            self._prompt_sub = self.create_subscription(
                String,
                config.prompt_topic,
                self._prompt_callback,
                10,
            )

            # Create publisher
            self._action_pub = self.create_publisher(
                Float64MultiArray,
                config.action_topic,
                10,
            )

            # Create control timer
            self._control_timer = self.create_timer(
                1.0 / config.control_frequency_hz,
                self._control_callback,
            )

            # Connect to server
            self.get_logger().info(f"Connecting to gRPC server at {config.server_address}")
            if not self.policy_client.connect():
                self.get_logger().error("Failed to connect to gRPC server")
            else:
                self.get_logger().info("Connected to gRPC server")

        def _image_callback(self, msg: RosImage, camera_idx: int):
            """Handle incoming image messages.

            Args:
                msg: ROS Image message.
                camera_idx: Index of the camera.
            """
            try:
                # Convert ROS image to numpy array
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                self._latest_images[camera_idx] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")

        def _state_callback(self, msg: JointState):
            """Handle incoming joint state messages.

            Args:
                msg: JointState message.
            """
            # Combine positions and velocities into state vector
            positions = list(msg.position) if msg.position else []
            velocities = list(msg.velocity) if msg.velocity else []
            self._latest_state = np.array(positions + velocities, dtype=np.float32)

        def _prompt_callback(self, msg: String):
            """Handle incoming prompt messages.

            Args:
                msg: String message containing the prompt.
            """
            self._current_prompt = msg.data
            self.get_logger().info(f"Updated prompt: {self._current_prompt}")

        def _control_callback(self):
            """Main control loop callback."""
            if not self.policy_client.is_connected():
                self.get_logger().warn_throttle(self.get_clock(), 5000, "Not connected to gRPC server")
                return

            # Check if we have all required data
            if any(img is None for img in self._latest_images):
                return

            if self._latest_state is None:
                return

            if not self._current_prompt:
                return

            try:
                # Get action chunk from server
                action_chunk, inference_time_ms = self.policy_client.get_action_chunk(
                    images=self._latest_images.copy(),
                    state=self._latest_state,
                    prompt=self._current_prompt,
                )

                # Publish action chunk
                action_msg = Float64MultiArray()
                action_msg.data = action_chunk.tolist()
                self._action_pub.publish(action_msg)

                self.get_logger().debug(
                    f"Action chunk: {action_chunk[:3]}..., inference time: {inference_time_ms:.1f}ms"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to get action chunk: {e}")

        def destroy_node(self):
            """Clean up resources when node is destroyed."""
            self.policy_client.disconnect()
            super().destroy_node()


def main():
    """Main entry point for standalone client testing."""
    parser = argparse.ArgumentParser(description="gRPC Robot Policy Client")
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="Server address (host:port)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test inference with dummy data",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ClientConfig(server_address=args.server_address)
    client = PolicyClient(config)

    if not client.connect():
        logger.error("Failed to connect to server")
        return

    # Health check
    health = client.health_check()
    logger.info(f"Server health: {health}")

    if args.test:
        # Test with dummy data
        logger.info("Running test inference...")

        # Create dummy images (224x224 RGB) as a list
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        ]

        # Create dummy state
        state = np.random.randn(32).astype(np.float32)

        # Get action chunk
        try:
            action_chunk, inference_time = client.get_action_chunk(
                images=images,
                state=state,
                prompt="Pick up the red block",
            )
            logger.info(f"Action chunk shape: {action_chunk.shape}")
            logger.info(f"Action chunk (first 5): {action_chunk}")
            logger.info(f"Inference time: {inference_time:.2f}ms")
        except Exception as e:
            logger.error(f"Test failed: {e}")

    client.disconnect()


if __name__ == "__main__":
    main()
