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
"""Smoke tests for the ``low_level_planner`` -> ``low_level`` deprecation shim.

These verify that the legacy import paths still resolve and that the
legacy class names are aliases for the new ones, so older user code and
pickled checkpoints continue to load.
"""

import importlib
import sys
import warnings

import pytest

# Ensure each test sees a fresh import of the shim modules so the
# package-level DeprecationWarning fires at predictable times. The shim's
# __init__.py emits the warning at module load.
_SHIM_MODULES = (
    "opentau.policies.pi07.low_level_planner",
    "opentau.policies.pi07.low_level_planner.configuration_pi07_low_level",
    "opentau.policies.pi07.low_level_planner.modeling_pi07_low_level",
    "opentau.policies.pi07.low_level_planner.video_encoder",
)


@pytest.fixture
def fresh_shim_imports():
    saved = {name: sys.modules.pop(name) for name in _SHIM_MODULES if name in sys.modules}
    try:
        yield
    finally:
        for name in _SHIM_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_package_init_emits_deprecation_warning(fresh_shim_imports):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        importlib.import_module("opentau.policies.pi07.low_level_planner")
    deprecation = [r for r in records if issubclass(r.category, DeprecationWarning)]
    assert deprecation, "shim package should emit a DeprecationWarning at import"
    assert "low_level_planner is deprecated" in str(deprecation[0].message)
    assert "PI07LowLevelConfig" in str(deprecation[0].message)


def test_legacy_config_alias_resolves_to_new_class(fresh_shim_imports):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from opentau.policies.pi07.low_level.configuration_pi07_low_level import PI07LowLevelConfig
        from opentau.policies.pi07.low_level_planner.configuration_pi07_low_level import (
            PI07LowLevelPlannerConfig,
        )
    assert PI07LowLevelPlannerConfig is PI07LowLevelConfig


def test_legacy_modeling_aliases_resolve_to_new_classes(fresh_shim_imports):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
            PI07LowLevelFlowMatching,
            PI07LowLevelPolicy,
        )
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            PI07LowLevelPlannerFlowMatching,
            PI07LowLevelPlannerPolicy,
        )
    assert PI07LowLevelPlannerPolicy is PI07LowLevelPolicy
    assert PI07LowLevelPlannerFlowMatching is PI07LowLevelFlowMatching


def test_legacy_modeling_helpers_still_importable(fresh_shim_imports):
    """Helpers re-exported by the shim must keep their original identity."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
            make_att_2d_masks,
            pad_discrete_tokens,
            resize_with_pad,
        )
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            make_att_2d_masks as legacy_make_att_2d_masks,
        )
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            pad_discrete_tokens as legacy_pad_discrete_tokens,
        )
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            resize_with_pad as legacy_resize_with_pad,
        )
    assert legacy_make_att_2d_masks is make_att_2d_masks
    assert legacy_pad_discrete_tokens is pad_discrete_tokens
    assert legacy_resize_with_pad is resize_with_pad


def test_legacy_video_encoder_aliases_resolve_to_new_classes(fresh_shim_imports):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from opentau.policies.pi07.low_level.video_encoder import (
            SpaceTimeEncoderLayerWrapper,
            SpaceTimeSiglipVideoEncoder,
            suppress_spacetime_temporal,
        )
        from opentau.policies.pi07.low_level_planner.video_encoder import (
            SpaceTimeEncoderLayerWrapper as legacy_SpaceTimeEncoderLayerWrapper,
        )
        from opentau.policies.pi07.low_level_planner.video_encoder import (
            SpaceTimeSiglipVideoEncoder as legacy_SpaceTimeSiglipVideoEncoder,
        )
        from opentau.policies.pi07.low_level_planner.video_encoder import (
            suppress_spacetime_temporal as legacy_suppress_spacetime_temporal,
        )
    assert legacy_SpaceTimeEncoderLayerWrapper is SpaceTimeEncoderLayerWrapper
    assert legacy_SpaceTimeSiglipVideoEncoder is SpaceTimeSiglipVideoEncoder
    assert legacy_suppress_spacetime_temporal is suppress_spacetime_temporal
