# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Calibrate an SO-100/SO-101 arm (follower robot or leader teleoperator).

Calibration files are stored under ``~/.cache/huggingface/lerobot/calibration``
(same location as upstream LeRobot, so existing calibrations are reused).

Examples:
    python -m opentau.scripts.so101.calibrate \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower

    python -m opentau.scripts.so101.calibrate \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from opentau.scripts.so101.robots import (  # noqa: F401  (registers draccus choices)
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from opentau.scripts.so101.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
)
from opentau.scripts.so101.utils import init_logging


@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")
        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    device.connect(calibrate=False)
    try:
        device.calibrate()
    finally:
        device.disconnect()


def main() -> None:
    calibrate()


if __name__ == "__main__":
    main()
