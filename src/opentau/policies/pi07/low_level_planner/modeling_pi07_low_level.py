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
"""DEPRECATED shim: use ``opentau.policies.pi07.low_level.modeling_pi07_low_level``.

``PI07LowLevelPlannerPolicy`` and ``PI07LowLevelPlannerFlowMatching``
are preserved as aliases for ``PI07LowLevelPolicy`` /
``PI07LowLevelFlowMatching``. See
``opentau.policies.pi07.low_level_planner.__init__`` for the deprecation
warning.
"""

from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    PI07LowLevelFlowMatching as PI07LowLevelFlowMatching,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    PI07LowLevelPolicy as PI07LowLevelPolicy,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    create_sinusoidal_pos_embedding as create_sinusoidal_pos_embedding,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    make_att_2d_masks as make_att_2d_masks,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    pad_discrete_tokens as pad_discrete_tokens,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    resize_with_pad as resize_with_pad,
)

PI07LowLevelPlannerPolicy = PI07LowLevelPolicy
PI07LowLevelPlannerFlowMatching = PI07LowLevelFlowMatching
