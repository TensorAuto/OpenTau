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

"""SO-100/SO-101 arm support for OpenTau.

A self-contained vendor of the LeRobot 0.4.4 hardware stack (Feetech motor
bus, SO follower robot, SO leader teleoperator, OpenCV cameras) plus CLI
scripts for calibration, teleoperation, and dataset recording. Recording
writes OpenTau's native LeRobotDataset format via ``opentau.datasets``.

Requires the ``so101`` extra: ``uv sync --extra so101``.

CLI (also exposed as opentau-so101-* console scripts):
    python -m opentau.scripts.so101.find_port
    python -m opentau.scripts.so101.setup_motors --robot.type=so101_follower --robot.port=...
    python -m opentau.scripts.so101.calibrate --robot.type=so101_follower --robot.port=... --robot.id=...
    python -m opentau.scripts.so101.teleoperate --robot.type=so101_follower ... --teleop.type=so101_leader ...
    python -m opentau.scripts.so101.record --robot.type=so101_follower ... --dataset.repo_id=...
    python -m opentau.scripts.so101.find_cameras
"""
