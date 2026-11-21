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

"""The shipped ROS conversion example must declare the shapes its extractors produce.

``add_frame`` validates every frame against the declared feature shapes, so a
mismatch makes every bag fail and the converted dataset ends up empty.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import opentau
from opentau.configs.ros2lerobot import RosToLeRobotConfig
from opentau.utils.ros2lerobot import EXTRACTORS

_EXAMPLE_CONFIG = Path(opentau.__file__).parent.parent.parent / "configs" / "examples" / "ros2lerobot.json"


def _fake_msg(enum_value: str, joint_order: list[str]) -> SimpleNamespace:
    """Builds a minimal message with every configured joint, as the extractor expects it.

    Args:
        enum_value: ``"state"`` for a JointState-like message, ``"action"`` for a motor command message.
        joint_order: Joint names from the config.

    Returns:
        A message carrying one value per joint.
    """
    n = len(joint_order)
    if enum_value == "state":
        return SimpleNamespace(name=list(joint_order), position=[0.1] * n, velocity=[0.2] * n)
    return SimpleNamespace(joint_names=list(joint_order), commands=[SimpleNamespace(q=0.3) for _ in range(n)])


@pytest.mark.parametrize("feature_name", ["observation.state", "action"])
def test_example_config_shape_matches_extractor_output(feature_name: str):
    """The declared shape equals the length the extractor returns for the configured joints.

    Args:
        feature_name: The dataset feature under test.
    """
    raw = json.loads(_EXAMPLE_CONFIG.read_text())
    feature = raw["dataset_features"][feature_name]
    cfg = RosToLeRobotConfig(joint_order=raw["joint_order"])
    extractor = EXTRACTORS[feature["enum_values"]](cfg)

    msg = _fake_msg(feature["enum_values"], raw["joint_order"])
    values = extractor(msg, feature["ros_topic"], feature["topic_attribute"])

    assert values, f"{feature_name} extractor returned nothing"
    assert feature["shape"] == [len(values)]
