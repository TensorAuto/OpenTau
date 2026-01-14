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

from dataclasses import dataclass, field

from opentau.configs.types import ROSFeature


@dataclass
class RosToLeRobotConfig:
    """Configuration for ROS to LeRobot conversion settings.

    This configuration is used for converting ROS bags to LeRobot dataset format.

    Attributes:
        input_path (str): Path to the directory containing input ROS bags.
            Defaults to "input_path".
        output_path (str): Path where the converted LeRobot dataset will be saved.
            Defaults to "output_path".
        fps (int): Frames per second for the output dataset. Timestamps will be
            synchronized to this frequency. Defaults to 10.
        joint_order (list[str]): List of joint names defining the order of elements
            in joint state vectors. Defaults to an empty list.
        robot_type (str): Type of robot being used. Defaults to "unknown".
        use_videos (bool): Whether to use videos in the output dataset. Defaults to False.
        dataset_features (dict[str, ROSFeature]): Dictionary mapping dataset feature
            names (e.g., "observation.state") to their corresponding ROS topic
            configuration. Defaults to an empty dictionary.
        ros_topic_mapping (dict[str, str]): Dictionary mapping users ROS topic names to
            universal ROS topic names. Defaults to an empty dictionary.
    """

    input_path: str = "input_path"
    output_path: str = "output_path"
    fps: int = 10
    joint_order: list[str] = field(default_factory=list)
    robot_type: str = "unknown"
    use_videos: bool = False
    dataset_features: dict[str, ROSFeature] = field(default_factory=dict)
    ros_topic_mapping: dict[str, str] = field(default_factory=dict)
