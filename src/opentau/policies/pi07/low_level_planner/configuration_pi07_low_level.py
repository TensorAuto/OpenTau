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
"""DEPRECATED shim: use ``opentau.policies.pi07.low_level.configuration_pi07_low_level``.

``PI07LowLevelPlannerConfig`` is preserved as an alias for
``PI07LowLevelConfig`` so older configs and pickled checkpoints continue
to import. See ``opentau.policies.pi07.low_level_planner.__init__`` for
the deprecation warning.
"""

from opentau.policies.pi07.low_level.configuration_pi07_low_level import (
    PI07LowLevelConfig as PI07LowLevelConfig,
)

PI07LowLevelPlannerConfig = PI07LowLevelConfig
