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

"""Teleoperate an SO-100/SO-101 follower with a leader arm.

Example:
    python -m opentau.scripts.so101.teleoperate \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader

With live camera display (requires rerun-sdk):
    ... --robot.cameras='{"front": {"type": "opencv", "index_or_path": "/dev/video0",
        "width": 640, "height": 480, "fps": 30}}' --display_data=true
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus

from opentau.scripts.so101.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
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
from opentau.scripts.so101.utils import init_logging, move_cursor_up, precise_sleep


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig = field(default_factory=lambda: None)  # type: ignore[assignment]
    robot: RobotConfig = field(default_factory=lambda: None)  # type: ignore[assignment]
    # Limit the maximum frames per second.
    fps: int = 30
    # Limit the maximum teleoperation time in seconds. By default, no limit.
    teleop_time_s: float | None = None
    # Display camera streams and joint data in rerun.
    display_data: bool = False

    def __post_init__(self):
        if self.teleop is None or self.robot is None:
            raise ValueError("Both --robot and --teleop must be provided.")


def _init_rerun(session_name: str = "so101_teleoperate") -> None:
    import rerun as rr

    rr.init(session_name)
    rr.spawn(memory_limit="10%")


def _log_rerun_data(observation: dict, action: dict) -> None:
    import numpy as np
    import rerun as rr

    for key, val in observation.items():
        if isinstance(val, np.ndarray) and val.ndim == 3:
            rr.log(f"observation/{key}", rr.Image(val), static=True)
        elif isinstance(val, (int, float)):
            rr.log(f"observation/{key}", rr.Scalars(float(val)))
    for key, val in action.items():
        if isinstance(val, (int, float)):
            rr.log(f"action/{key}", rr.Scalars(float(val)))


def teleop_loop(cfg: TeleoperateConfig, robot, teleop) -> None:
    display_len = max(len(key) for key in robot.action_features)
    end_t = time.perf_counter() + cfg.teleop_time_s if cfg.teleop_time_s else None
    while end_t is None or time.perf_counter() < end_t:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if cfg.display_data:
            observation = robot.get_observation()
            _log_rerun_data(observation, action)
        sent = robot.send_action(action)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / cfg.fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in sent.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(len(sent) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        _init_rerun()

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()
    try:
        teleop_loop(cfg, robot, teleop)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        teleop.disconnect()


def main() -> None:
    teleoperate()


if __name__ == "__main__":
    main()
