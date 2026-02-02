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

"""ROS 2 to LeRobot dataset extractors and utilities.

This module provides feature extractors for converting ROS 2 topic messages
(e.g., joint_states, images) into LeRobot dataset features. Extractors are
keyed by enum value in EXTRACTORS and used by the convert_ros_to_lerobot script.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from opentau.configs.ros2lerobot import RosToLeRobotConfig


def get_nested_item(obj: Any, flattened_key: str, sep: str = ".") -> Any:
    """Get a nested item from an object using a flattened attribute path.

    Args:
        obj: Object with nested attributes to access (e.g., ROS message).
        flattened_key: Dot-separated path to the attribute (e.g., "a.b.c").
        sep: Separator used in the flattened key. Defaults to ".".

    Returns:
        The value at the nested path specified by the flattened key.

    Example:
        >>> dct = {"a": {"b": {"c": 42}}}
        >>> get_nested_item(dct, "a.b.c")
        42
    """
    split_keys = flattened_key.split(sep)
    getter = getattr(obj, split_keys[0])
    if len(split_keys) == 1:
        return getter

    for key in split_keys[1:]:
        getter = getattr(getter, key)

    return getter


class FeatureExtractor(ABC):
    """Abstract base class for extracting a dataset feature from a ROS 2 message."""

    def __init__(self, cfg: RosToLeRobotConfig):
        """Initialize the extractor with conversion config.

        Args:
            cfg: ROS to LeRobot conversion config (joint order, features, etc.).
        """
        self.cfg = cfg

    @abstractmethod
    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        """Extract the feature value from a single message.

        Args:
            msg: Deserialized ROS 2 message.
            ros_topic: Topic name the message was received on.
            attribute: Message attribute path to extract (e.g., "position", "data").

        Returns:
            Extracted value (e.g., list of floats, numpy array), or None/empty
            on failure.
        """
        pass


class StateExtractor(FeatureExtractor):
    """Extracts observation.state from ROS 2 joint_states (position + velocity)."""

    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        """Extract state as position and velocity ordered by config joint_order.

        Args:
            msg: Joint state message with name, position, velocity (e.g., sensor_msgs/JointState).
            ros_topic: Topic name the message was received on.
            attribute: Attribute path for values (e.g., "position").

        Returns:
            List of floats: positions for each joint in joint_order, then velocities,
            or empty list if extraction fails.
        """
        # Handle Joint Ordering
        if not self.cfg.joint_order:
            if hasattr(msg, "name"):
                self.cfg.joint_order = msg.name
                logging.info(
                    f"Auto-detected joint order ({len(self.cfg.joint_order)} joints): {self.cfg.joint_order}"
                )
            else:
                logging.warning("Message does not have 'name' attribute, cannot auto-detect joint order.")
                return []

        # Create a map for this message {name: index}
        if hasattr(msg, "name"):
            current_map = {name: i for i, name in enumerate(msg.name)}
        else:
            # Fallback if msg doesn't have names but we have joint_order and data seems to match?
            # For now assume joint_states structure
            return []

        extracted_values = []
        extracted_velocities = []
        try:
            # Check if attribute exists on msg (at top level or nested?)

            raw_values = get_nested_item(msg, attribute, sep=".")
            raw_velocities = get_nested_item(msg, "velocity", sep=".")
            for j_name in self.cfg.joint_order:
                if j_name in current_map:
                    idx = current_map[j_name]
                    if len(raw_values) > idx:
                        extracted_values.append(raw_values[idx])
                        extracted_velocities.append(raw_velocities[idx])
                    else:
                        extracted_values.append(0.0)
                        extracted_velocities.append(0.0)
                else:
                    # Joint missing in this message
                    extracted_values.append(0.0)
                    extracted_velocities.append(0.0)
            return extracted_values + extracted_velocities

        except (KeyError, AttributeError, TypeError) as e:
            logging.warning(f"Error extracting {attribute} from {ros_topic}: {e}")
            return []


class ActionExtractor(FeatureExtractor):
    """Extracts action (e.g., target joint positions) from ROS 2 control messages."""

    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        """Extract action values ordered by config joint_order.

        Args:
            msg: Message with joint_names and attribute (e.g., trajectory point with .q).
            ros_topic: Topic name the message was received on.
            attribute: Attribute path to joint values (e.g., "points.positions" or similar).

        Returns:
            List of floats for each joint in joint_order, or empty list if extraction fails.
        """
        # Handle Joint Ordering
        if not self.cfg.joint_order:
            if hasattr(msg, "joint_names"):
                self.cfg.joint_order = msg.joint_names
                logging.info(
                    f"Auto-detected joint order ({len(self.cfg.joint_order)} joints): {self.cfg.joint_order}"
                )
            else:
                logging.warning("Message does not have 'name' attribute, cannot auto-detect joint order.")
                return []

        # Create a map for this message {name: index}
        if hasattr(msg, "joint_names"):
            current_map = {name: i for i, name in enumerate(msg.joint_names)}
        else:
            # Fallback if msg doesn't have names but we have joint_order and data seems to match?
            # For now assume joint_states structure
            return []

        extracted_values = []
        try:
            # Check if attribute exists on msg (at top level or nested?)

            raw_values = get_nested_item(msg, attribute, sep=".")
            raw_q = [raw_value.q for raw_value in raw_values]

            for j_name in self.cfg.joint_order:
                if j_name in current_map:
                    idx = current_map[j_name]
                    if len(raw_values) > idx:
                        extracted_values.append(raw_q[idx])
                    else:
                        extracted_values.append(0.0)
                else:
                    # Joint missing in this message
                    extracted_values.append(0.0)

            return extracted_values

        except (KeyError, AttributeError, TypeError) as e:
            logging.warning(f"Error extracting {attribute} from {ros_topic}: {e}")
            return []


class ImageExtractor(FeatureExtractor):
    """Extracts observation.image from ROS 2 compressed or raw image messages."""

    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        """Decode image bytes to a numpy array (H, W, C) in RGB.

        Args:
            msg: Image message with .data (bytes), e.g., sensor_msgs/CompressedImage.
            ros_topic: Topic name the message was received on.
            attribute: Attribute path (e.g., "data"); typically "data" for image payload.

        Returns:
            numpy array of shape (H, W, 3) uint8 RGB, or None if decoding fails.
            RGBA is converted to RGB by dropping the alpha channel.
        """
        try:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(msg.data))
            # Convert to numpy array
            image_np = np.array(image)
            # Handle RGBA if necessary, or just ensure RGB
            if image_np.shape[-1] == 4:
                image_np = image_np[..., :3]
            return image_np

        except (KeyError, AttributeError, TypeError, Exception) as e:
            logging.warning(f"Error extracting {attribute} from {ros_topic}: {e}")
            return None


# Mapping of dataset feature enum values to extractor classes.
# Used by convert_ros_to_lerobot to dispatch per-topic extraction.
EXTRACTORS = {
    "state": StateExtractor,
    "action": ActionExtractor,
    "image": ImageExtractor,
}
