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
"""Constants used throughout the OpenTau library.

This module defines key constants for:
- Observation and action keys used in datasets and environments
- File and directory names for checkpoints, training state, and model storage
- Cache directory configuration for HuggingFace Hub integration

These constants ensure consistent naming conventions across the codebase and
provide a centralized location for configuration values.
"""

import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STATE = "observation.state"
OBS_ENV = "observation.environment_state"  # TODO: remove
OBS_ROBOT = "state"  # TODO: remove
OBS_IMAGE = "observation.image"  # TODO: remove
OBS_IMAGES = "observation.images"  # TODO: remove
ACTION = "actions"  # TODO: remove
OBS_ENV_STATE = "observation.environment_state"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "opentau"
HF_OPENTAU_HOME = Path(os.getenv("HF_OPENTAU_HOME", default_cache_path)).expanduser()
