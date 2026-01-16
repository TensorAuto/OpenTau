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

import logging
from abc import ABC, abstractmethod
from typing import Any

from opentau.configs.ros2lerobot import RosToLeRobotConfig


def get_nested_item(obj: Any, flattened_key: str, sep: str = ".") -> Any:
    """Get a nested item from a dictionary-like object using a flattened key.

    Args:
        obj: Dictionary-like object to access.
        flattened_key: Flattened key path (e.g., "a/b/c").
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
    def __init__(self, cfg: RosToLeRobotConfig):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        pass


class StateExtractor(FeatureExtractor):
    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
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
        try:
            # Check if attribute exists on msg (at top level or nested?)

            raw_values = get_nested_item(msg, attribute, sep=".")

            for j_name in self.cfg.joint_order:
                if j_name in current_map:
                    idx = current_map[j_name]
                    if len(raw_values) > idx:
                        extracted_values.append(raw_values[idx])
                    else:
                        extracted_values.append(0.0)
                else:
                    # Joint missing in this message
                    extracted_values.append(0.0)

            return extracted_values

        except (KeyError, AttributeError, TypeError) as e:
            logging.warning(f"Error extracting {attribute} from {ros_topic}: {e}")
            return []


# Mapping of enum values to extractors
EXTRACTORS = {
    "state": StateExtractor,
}
