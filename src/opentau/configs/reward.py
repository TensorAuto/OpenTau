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
"""Reward configuration module.

This module provides the RewardConfig class which contains configuration
parameters for reward computation in reinforcement learning scenarios.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward computation settings.

    This configuration is used for reward modeling and computation in reinforcement
    learning scenarios.

    Args:
        number_of_bins: Number of bins used for reward discretization or binning.
            Defaults to 201.
        C_neg: Negative constant used in reward computation. Defaults to -1000.0.
        reward_normalizer: Normalization factor for rewards. Defaults to 400.
        N_steps_look_ahead: Number of steps to look ahead when computing rewards.
            Defaults to 50.
    """

    number_of_bins: int = 201
    C_neg: float = -1000.0
    reward_normalizer: int = 400
    N_steps_look_ahead: int = 50
