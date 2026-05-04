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
"""DEPRECATED: use ``opentau.policies.pi07.low_level`` instead.

The ``low_level_planner`` directory is preserved as a thin re-export
shim so older user code and pickled checkpoints that reference the old
import path / class names keep working. New code should import from
``opentau.policies.pi07.low_level``. This shim will be removed in a
future release.

Class renames:

- ``PI07LowLevelPlannerConfig`` → ``PI07LowLevelConfig``
- ``PI07LowLevelPlannerPolicy`` → ``PI07LowLevelPolicy``
- ``PI07LowLevelPlannerFlowMatching`` → ``PI07LowLevelFlowMatching``
"""

import warnings as _warnings

_warnings.warn(
    "opentau.policies.pi07.low_level_planner is deprecated; "
    "use opentau.policies.pi07.low_level instead. "
    "PI07LowLevelPlannerConfig was renamed to PI07LowLevelConfig, "
    "PI07LowLevelPlannerPolicy to PI07LowLevelPolicy, and "
    "PI07LowLevelPlannerFlowMatching to PI07LowLevelFlowMatching.",
    DeprecationWarning,
    stacklevel=2,
)
