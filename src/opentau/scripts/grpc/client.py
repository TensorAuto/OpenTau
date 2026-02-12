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
for ML inference. It subscribes to /joint_states for robot state, creates
fake images for inference, and publishes motor commands to
/motor_command_controller/motor_commands.

Usage:
    python src/opentau/scripts/grpc/client.py \
        --server_address 192.168.1.100:50051 \
        --prompt "pick up the red block"
"""

import argparse
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy

# Import ROS 2 message types
from interfaces.msg import MotorCommands, RawMotorCommand
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import JointState

import grpc
from opentau.scripts.grpc import robot_inference_pb2, robot_inference_pb2_grpc

logger = logging.getLogger(__name__)

# Topics
MOTOR_COMMANDS_TOPIC = "/motor_command_controller/motor_commands"
JOINT_STATES_TOPIC = "/joint_states"

# Joint configuration (example robot).
JOINT_NAMES: list[str] = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "elbow_flex_joint",
    "wrist_roll_joint",
    "wrist_yaw_joint",
    "hip_pitch_joint",
    "hip_roll_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
    "gripper_finger_joint",
]


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

                # Convert repeated ActionVector messages into a 2D numpy array:
                # shape = (chunk_length, action_dim)
                if not response.action_chunk:
                    action_chunk = np.zeros((0,), dtype=np.float32)
                else:
                    action_vectors = [
                        np.asarray(action_vector.values, dtype=np.float32)
                        for action_vector in response.action_chunk
                    ]
                    # Stack into (T, D) array where T is chunk length.
                    action_chunk = np.stack(action_vectors, axis=0)

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


@dataclass
class ROS2Config:
    """Configuration for the ROS 2 policy client node."""

    # gRPC settings
    server_address: str = "localhost:50051"
    timeout_seconds: float = 30.0  # Allow longer timeout for ML inference warmup

    # Topic names
    state_topic: str = JOINT_STATES_TOPIC
    motor_commands_topic: str = MOTOR_COMMANDS_TOPIC

    # Control settings
    control_frequency_hz: float = 10.0
    prompt: str = ""

    # Fake image settings
    num_cameras: int = 2
    image_height: int = 224
    image_width: int = 224


class ROS2PolicyClient(Node):
    """ROS 2 node that interfaces with the gRPC policy server.

    This node subscribes to joint states, creates fake images for inference,
    sends them to the gRPC server, and publishes the resulting actions as
    motor commands. The prompt is provided via command-line argument.

    Example usage:
        ```python
        import rclpy
        from motor_command_controller.grpc.client import ROS2PolicyClient, ROS2Config

        rclpy.init()
        config = ROS2Config(
            server_address="192.168.1.100:50051",
            control_frequency_hz=30.0,
            prompt="pick up the red block",
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
        # Fixed joint ordering for this example robot.
        self.joint_names: list[str] = JOINT_NAMES

        # Initialize gRPC client
        client_config = ClientConfig(
            server_address=config.server_address,
            timeout_seconds=config.timeout_seconds,
        )
        self.policy_client = PolicyClient(client_config)

        # State storage
        self._latest_positions: Optional[list[float]] = None
        self._latest_velocities: Optional[list[float]] = None
        self._prompt: str = config.prompt

        # Create subscriber for joint states
        self._state_sub = self.create_subscription(
            JointState,
            config.state_topic,
            self._state_callback,
            10,
        )
        self.get_logger().info(f"Subscribed to {config.state_topic}")
        self.get_logger().info(f"Using prompt: {self._prompt}")

        # Create publisher for motor commands
        self._motor_commands_pub = self.create_publisher(
            MotorCommands,
            config.motor_commands_topic,
            10,
        )
        self.get_logger().info(f"Publishing to {config.motor_commands_topic}")

        # Create control timer
        self._control_timer = self.create_timer(
            # 1.0 / config.control_frequency_hz,
            1,
            self._control_callback,
        )

        # Connect to server
        self.get_logger().info(f"Connecting to gRPC server at {config.server_address}")
        if not self.policy_client.connect():
            self.get_logger().error("Failed to connect to gRPC server")
        else:
            self.get_logger().info("Connected to gRPC server")

    def _create_fake_images(self) -> list[np.ndarray]:
        """Create fake images for inference.

        Returns:
            List of fake RGB images.
        """
        images = []
        for _ in range(self.config.num_cameras):
            # Create a random RGB image
            image = np.random.randint(
                0,
                255,
                (self.config.image_height, self.config.image_width, 3),
                dtype=np.uint8,
            )
            images.append(image)
        return images

    def _state_callback(self, msg: JointState):
        """Handle incoming joint state messages.

        Args:
            msg: JointState message.
        """
        # For this example script, we simply take the first N joints from the
        # incoming message, where N is the number of fake joints.
        num_joints = len(self.joint_names)
        if len(msg.position) < num_joints or len(msg.velocity) < num_joints:
            self.get_logger().warning(
                f"Received joint_states with fewer than {num_joints} joints; waiting for full state.",
                throttle_duration_sec=5.0,
            )
            return

        self._latest_positions = list(msg.position[:num_joints])
        self._latest_velocities = list(msg.velocity[:num_joints])

    def _control_callback(self):
        """Main control loop callback."""
        if not self.policy_client.is_connected():
            self.get_logger().warning(
                "Not connected to gRPC server",
                throttle_duration_sec=5.0,
            )
            return

        # Check if we have joint state data
        if self._latest_positions is None or self._latest_velocities is None:
            self.get_logger().warning(
                "Waiting for joint_states...",
                throttle_duration_sec=5.0,
            )
            return

        # Check if we have a prompt
        if not self._prompt:
            self.get_logger().warning(
                "No prompt provided. Use --prompt argument.",
                throttle_duration_sec=5.0,
            )
            return

        try:
            # Create state vector (positions + velocities)
            state = np.array(
                self._latest_positions + self._latest_velocities,
                dtype=np.float32,
            )

            # Create fake images
            images = self._create_fake_images()

            self.get_logger().info("Sending request to server")
            self.get_logger().info(f"Images list length: {len(images)}")
            if images:
                self.get_logger().info(f"First image shape: {images[0].shape}")
            else:
                self.get_logger().info("Images list is empty")
            self.get_logger().info(f"State shape: {state.shape}")

            # Get action chunk from server
            action_chunk, inference_time_ms = self.policy_client.get_action_chunk(
                images=images,
                state=state,
                prompt=self._prompt,
            )
            self.get_logger().info("Received action chunk from server")
            self.get_logger().info(f"Action chunk shape: {action_chunk.shape}")
            self.get_logger().info(f"Inference time: {inference_time_ms:.1f} ms")

            # Publish motor commands
            self._publish_motor_commands(action_chunk)

            self.get_logger().debug(
                f"Action chunk: {action_chunk[:3]}..., inference time: {inference_time_ms:.1f}ms"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to get action chunk: {e}")

    def _publish_motor_commands(self, action_chunk: np.ndarray):
        """Publish motor commands from action chunk.

        Args:
            action_chunk: Action chunk from the policy server.
                Expected to be either:
                - A 2D array of shape (chunk_length, num_joints) where each row
                  is a full action vector for all joints, or
                - A 1D array of shape (num_joints,) for a single action vector.
        """
        msg = MotorCommands()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        # Select the action vector to apply.
        # If we have a chunk of actions, use the most recent one.
        if action_chunk.ndim == 2:
            if action_chunk.shape[0] == 0:
                self.get_logger().error("Received empty action chunk (no timesteps)")
                return
            action = action_chunk[-1]
        elif action_chunk.ndim == 1:
            action = action_chunk
        else:
            self.get_logger().error(
                f"Unexpected action_chunk shape: {action_chunk.shape} (ndim={action_chunk.ndim})"
            )
            return

        # Create motor commands for each joint
        # Assumes action contains target positions for each joint
        num_joints = len(self.joint_names)
        if len(action) < num_joints:
            self.get_logger().error(f"Action vector too small: {len(action)} < {num_joints}")
            return

        msg.commands = [RawMotorCommand(q=float(action[i])) for i, joint_name in enumerate(self.joint_names)]

        self._motor_commands_pub.publish(msg)

    def publish_damping_command(self):
        """Publish a damping command to safely stop the robot."""
        if self._latest_positions is None:
            return

        msg = MotorCommands()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        msg.commands = [
            RawMotorCommand(q=self._latest_positions[i]) for i, joint_name in enumerate(self.joint_names)
        ]
        self._motor_commands_pub.publish(msg)

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        # Send damping command before shutting down
        self.publish_damping_command()
        self.policy_client.disconnect()
        super().destroy_node()


def main():
    """Main entry point for the ROS 2 gRPC policy client."""
    parser = argparse.ArgumentParser(description="gRPC Robot Policy Client (ROS 2)")
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="Server address (host:port)",
    )
    parser.add_argument(
        "--control_frequency",
        type=float,
        default=30.0,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Language prompt for the policy (required)",
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=2,
        help="Number of fake cameras to simulate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="gRPC timeout in seconds (increase for slow inference)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize ROS 2
    rclpy.init()

    config = ROS2Config(
        server_address=args.server_address,
        timeout_seconds=args.timeout,
        control_frequency_hz=args.control_frequency,
        prompt=args.prompt,
        num_cameras=args.num_cameras,
    )

    node = ROS2PolicyClient(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
