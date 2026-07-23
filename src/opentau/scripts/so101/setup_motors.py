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

"""Assign motor IDs/baudrates for a freshly assembled SO-100/SO-101 arm.

Connect ONE motor at a time when prompted.

Examples:
    python -m opentau.scripts.so101.setup_motors \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0

    python -m opentau.scripts.so101.setup_motors \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
"""

from dataclasses import dataclass

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

COMPATIBLE_DEVICES = ("so100_follower", "so101_follower", "so100_leader", "so101_leader")


@dataclass
class SetupConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")
        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError(f"{cfg.device.type} not in {COMPATIBLE_DEVICES}")

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    device.setup_motors()


def main() -> None:
    setup_motors()


if __name__ == "__main__":
    main()
