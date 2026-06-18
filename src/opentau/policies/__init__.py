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

"""Policies module for OpenTau.

This module exports the configuration classes for available policies,
such as PI0, PI05, and Value policy.
"""

from .cosmos3.configuration_cosmos3 import Cosmos3Config as Cosmos3Config
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi05.configuration_pi05 import PI05Config as PI05Config
from .pi05.configuration_pi05 import PI05ContinuousStateConfig as PI05ContinuousStateConfig
from .pi05_mem.configuration_pi05 import PI05MemConfig as PI05MemConfig
from .pi07.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07HighLevelPlannerConfig,
)
from .pi07.low_level.configuration_pi07_low_level import (
    PI07LowLevelConfig as PI07LowLevelConfig,
)

# Side-effect imports: the two ``pi07_paligemma`` config modules each carry an
# ``@PreTrainedConfig.register_subclass(...)`` decorator that draccus relies on
# to resolve ``--policy.type=pi07_paligemma_low_level`` /
# ``pi07_paligemma_high_level_planner``. The high-level config class shares the
# name ``PI07HighLevelPlannerConfig`` with its pi07 counterpart, so we cannot
# re-export it from this package without shadowing the latter; importers reach
# the class directly via its full module path or through ``get_policy_class``.
from .pi07_paligemma.high_level_planner import (
    configuration_pi07_high_level as _pi07_paligemma_high_level_config,  # noqa: F401
)
from .pi07_paligemma.low_level import (
    configuration_pi07_low_level as _pi07_paligemma_low_level_config,  # noqa: F401
)
from .value.configuration_value import ValueConfig as ValueConfig
