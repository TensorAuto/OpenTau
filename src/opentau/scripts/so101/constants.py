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

"""Filesystem constants for the SO-101 hardware stack.

Kept byte-compatible with upstream LeRobot so calibration files recorded with
either stack resolve to the same default location
(``~/.cache/huggingface/lerobot/calibration``).
"""

import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_STATE = f"{OBS_STR}.state"
OBS_IMAGES = f"{OBS_STR}.images"
ACTION = "action"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

default_calibration_path = HF_LEROBOT_HOME / "calibration"
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", default_calibration_path)).expanduser()
