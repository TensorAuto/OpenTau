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

"""Tests for the ``use_torch_compile`` training knob on
:class:`~opentau.configs.policies.PreTrainedConfig` and the in-place compile
helper :meth:`~opentau.policies.pretrained.PreTrainedPolicy.maybe_compile_for_training`.

Pins the contract the training pipeline relies on:

1. Every policy config in the registry inherits ``use_torch_compile = False``
   and ``torch_compile_mode = "default"`` — so adding a new policy never
   silently turns compilation on, and the default eager path is unchanged.
2. Both fields are keyword-settable on every config (draccus plumbing intact).
3. ``maybe_compile_for_training`` is a no-op when the flag is off, installs the
   in-place compiled dispatch when it is on, and — crucially — does NOT change
   ``self.model``'s ``state_dict`` keys (no ``_orig_mod.`` prefix), so existing
   checkpoints stay loadable and the optimizer still sees the same params.

These are all CPU-safe: ``torch.nn.Module.compile`` is lazy (it only swaps the
``__call__`` dispatch; the trace happens on the first forward), so none of this
actually invokes the inductor backend.
"""

import pytest
import torch.nn as nn

from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi05.configuration_pi05 import PI05Config, PI05ContinuousStateConfig
from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07HighLevelConfig,
)
from opentau.policies.pi07.low_level.configuration_pi07_low_level import PI07LowLevelConfig
from opentau.policies.pi07_paligemma.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07PaligemmaHighLevelConfig,
)
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.value.configuration_value import ValueConfig

_POLICY_CONFIG_CASES = [
    (PI0Config, "PI0Config"),
    (PI05Config, "PI05Config"),
    (PI05ContinuousStateConfig, "PI05ContinuousStateConfig"),
    (PI05MemConfig, "PI05MemConfig"),
    (PI06Config, "PI06Config"),
    (PI07LowLevelConfig, "PI07LowLevelConfig"),
    (PI07HighLevelConfig, "PI07HighLevelConfig"),
    (PI07PaligemmaLowLevelConfig, "PI07PaligemmaLowLevelConfig"),
    (PI07PaligemmaHighLevelConfig, "PI07PaligemmaHighLevelConfig"),
    (ValueConfig, "ValueConfig"),
]
ALL_POLICY_CONFIGS = [c for c, _ in _POLICY_CONFIG_CASES]
_POLICY_CONFIG_IDS = [name for _, name in _POLICY_CONFIG_CASES]


@pytest.mark.parametrize("config_cls", ALL_POLICY_CONFIGS, ids=_POLICY_CONFIG_IDS)
def test_compile_defaults_off(config_cls):
    """Every policy config inherits the compile knobs with safe defaults:
    compilation off, mode "default". A regression means a subclass shadowed the
    parent field or a default flipped — either of which would change the default
    training path."""
    cfg = config_cls()
    assert cfg.use_torch_compile is False
    assert cfg.torch_compile_mode == "default"


@pytest.mark.parametrize("config_cls", ALL_POLICY_CONFIGS, ids=_POLICY_CONFIG_IDS)
def test_compile_fields_settable(config_cls):
    """Both fields are keyword-settable on every policy config. Catches a
    draccus renaming or a subclass overriding the parent field with a different
    type."""
    cfg = config_cls(use_torch_compile=True, torch_compile_mode="max-autotune")
    assert cfg.use_torch_compile is True
    assert cfg.torch_compile_mode == "max-autotune"


class _StubConfig:
    """Minimal stand-in for the two fields the helper reads."""

    def __init__(self, use_torch_compile: bool = False, torch_compile_mode: str = "default"):
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode


class _StubPolicy:
    """Duck-typed stand-in for a policy.

    ``maybe_compile_for_training`` only touches ``self.config`` (two fields),
    ``self.model``, and ``type(self).__name__`` — so a minimal object exercises
    the exact logic without constructing a real policy (which would download and
    instantiate a multi-billion-parameter PaliGemma / Gemma 3 backbone).
    """

    maybe_compile_for_training = PreTrainedPolicy.maybe_compile_for_training

    def __init__(self, config, model):
        self.config = config
        self.model = model


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_disabled_is_noop():
    """Flag off → the submodule is left completely untouched (no compiled
    dispatch installed). This is the default training path."""
    model = _tiny_model()
    policy = _StubPolicy(_StubConfig(use_torch_compile=False), model)
    policy.maybe_compile_for_training()
    # nn.Module initializes _compiled_call_impl to None; compile() sets it.
    assert model._compiled_call_impl is None


def test_enabled_installs_compiled_dispatch():
    """Flag on → ``nn.Module.compile`` installs the compiled ``__call__``
    dispatch on the submodule (lazy; no actual trace happens here)."""
    model = _tiny_model()
    policy = _StubPolicy(_StubConfig(use_torch_compile=True), model)
    policy.maybe_compile_for_training()
    assert model._compiled_call_impl is not None


def test_enabled_preserves_state_dict_keys():
    """The whole reason for using in-place ``nn.Module.compile`` instead of
    ``model = torch.compile(model)``: the ``state_dict`` keys must be byte-for-
    byte identical, with no ``_orig_mod.`` prefix, so existing checkpoints keep
    loading and the optimizer keeps seeing the same parameter names."""
    model = _tiny_model()
    before = set(model.state_dict().keys())
    policy = _StubPolicy(_StubConfig(use_torch_compile=True), model)
    policy.maybe_compile_for_training()
    after = set(model.state_dict().keys())
    assert before == after
    assert not any("_orig_mod" in k for k in after)


def test_missing_model_is_safe(caplog):
    """A policy with no inner ``self.model`` nn.Module must not raise — it warns
    and leaves the policy uncompiled (defensive against future policies that
    don't follow the flow-matching layout)."""
    policy = _StubPolicy(_StubConfig(use_torch_compile=True), model=None)
    # Should not raise.
    policy.maybe_compile_for_training()
