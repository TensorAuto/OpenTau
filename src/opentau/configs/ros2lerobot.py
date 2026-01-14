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
    """

    input_path: str = "input_path"
    output_path: str = "output_path"
    fps: int = 10
    joint_order: list[str] = field(default_factory=list)
    dataset_features: dict[str, ROSFeature] = field(default_factory=dict)
