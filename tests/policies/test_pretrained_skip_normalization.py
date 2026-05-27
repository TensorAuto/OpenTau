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

"""Cross-policy tests for the ``skip_normalization_weights`` knob on
:class:`~opentau.configs.policies.PreTrainedConfig`.

Pins three guarantees:

1. Every policy config in the registry inherits the field with a ``False``
   default — so adding a new policy without explicitly opting in cannot
   accidentally start stripping buffers at load time.
2. The field is keyword-settable on every config — catches an accidental
   override that shadows the parent field or renames the keyword.
3. The shared strip / inf-guard helpers on
   :class:`~opentau.policies.pretrained.PreTrainedPolicy` behave as the
   per-policy ``from_pretrained`` overrides expect — single source of
   truth for the integration contract.
"""

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import Normalize, Unnormalize
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

# Parallel list of (cls, display_id) — display IDs disambiguate the two
# ``PI07HighLevelPlannerConfig`` classes (one each in ``pi07`` and
# ``pi07_paligemma``, same class name) which would otherwise render as
# ``PI07HighLevelPlannerConfig0`` / ``PI07HighLevelPlannerConfig1`` under
# pytest's default ``ids=lambda c: c.__name__`` and obscure which sibling
# a failure came from.
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
def test_inherited_default_is_false(config_cls):
    """Every policy config inherits ``skip_normalization_weights = False``
    from :class:`PreTrainedConfig`. A regression here means a subclass
    accidentally shadowed the parent field or the parent's default flipped.
    """
    cfg = config_cls()
    assert cfg.skip_normalization_weights is False


@pytest.mark.parametrize("config_cls", ALL_POLICY_CONFIGS, ids=_POLICY_CONFIG_IDS)
def test_can_be_set_true(config_cls):
    """The field is keyword-settable on every policy config. Catches a
    draccus-side renaming, a subclass overriding the field with a
    different type, or the parent field being made private.
    """
    cfg = config_cls(skip_normalization_weights=True)
    assert cfg.skip_normalization_weights is True


def _make_fake_module_with_normalize_buffers(
    *, stats: dict | None
) -> tuple[nn.Module, frozenset[str], frozenset[str]]:
    """Build a small ``nn.Module`` that holds real
    :py:class:`Normalize` / :py:class:`Unnormalize` submodules under the
    canonical attribute names used by every policy. Returns the module
    plus the two key sets the strip helper is expected to act on.
    """
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    parent = nn.Module()
    # New per-dataset normalization API (#336): pass a list of stat dicts via
    # ``per_dataset_stats=`` (or just ``num_datasets=`` to leave buffers at
    # the inf sentinel for the inf-guard test).
    norm_kwargs = {"per_dataset_stats": [stats]} if stats is not None else {"num_datasets": 1}
    parent.normalize_inputs = Normalize(features, norm_map, **norm_kwargs)
    parent.unnormalize_outputs = Unnormalize(features, norm_map, **norm_kwargs)
    # An unrelated parameter that the strip must NOT touch.
    parent.unrelated_layer = nn.Linear(2, 2)

    full_sd = parent.state_dict()
    normalize_keys = frozenset(k for k in full_sd if k.startswith("normalize_inputs."))
    unnormalize_keys = frozenset(k for k in full_sd if k.startswith("unnormalize_outputs."))
    assert normalize_keys, "test fixture: no normalize_inputs.* keys in state_dict"
    assert unnormalize_keys, "test fixture: no unnormalize_outputs.* keys in state_dict"
    return parent, normalize_keys, unnormalize_keys


class _FakeConfig:
    """Stand-in for :class:`PreTrainedConfig` for tests that exercise just
    the strip helper — the helper only reads ``skip_normalization_weights``
    and writes it back, so a duck-typed object is enough and avoids
    instantiating a concrete (abstract-method-requiring) subclass.
    """

    def __init__(self, skip_normalization_weights: bool):
        self.skip_normalization_weights = skip_normalization_weights


class TestStripNormalizationBuffersFromStateDict:
    """Integration tests for the shared strip helper used by every policy."""

    def test_no_op_when_flag_off(self):
        """Default path: flag is off ⇒ helper returns the dict unchanged
        and an empty stripped-keys set. The config flag is not mutated.
        """
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        original_sd = dict(sd)
        config = _FakeConfig(skip_normalization_weights=False)

        out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert out_sd is sd, "no-op path must return the input dict unchanged (no copy)"
        assert dict(out_sd) == original_sd
        assert stripped == frozenset()
        assert config.skip_normalization_weights is False  # unchanged

    def test_strips_only_normalize_buffer_keys_and_resets_flag(self):
        """Flag on with real buffer keys ⇒ helper drops every
        ``normalize_*`` / ``unnormalize_*`` key, leaves unrelated keys
        alone, and resets the flag to False (one-shot)."""
        parent, normalize_keys, unnormalize_keys = _make_fake_module_with_normalize_buffers(
            stats={
                "observation.state": {
                    "mean": np.zeros(4, dtype=np.float32),
                    "std": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "mean": np.zeros(3, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32),
                },
            }
        )
        sd = dict(parent.state_dict())
        unrelated_keys = frozenset(k for k in sd if k.startswith("unrelated_layer."))
        assert unrelated_keys, "test fixture: no unrelated_layer.* keys"
        config = _FakeConfig(skip_normalization_weights=True)

        out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert stripped == normalize_keys | unnormalize_keys
        assert set(out_sd.keys()) == set(sd.keys()) - stripped
        # Unrelated weights survive.
        for key in unrelated_keys:
            assert key in out_sd
            assert torch.equal(out_sd[key], sd[key])
        # One-shot reset of the config flag.
        assert config.skip_normalization_weights is False

    def test_warning_when_flag_set_but_no_keys_match(self, caplog):
        """Flag on with no matching keys in the dict ⇒ helper emits a
        WARNING (not an INFO with "dropped 0 keys") so a user who flipped
        the flag *expecting* the source-mixture stats to be voided sees
        that the flag was dead this load."""
        config = _FakeConfig(skip_normalization_weights=True)
        sd: dict[str, torch.Tensor] = {"some.other.weight": torch.zeros(3)}

        with caplog.at_level(logging.WARNING):
            out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert stripped == frozenset()
        assert out_sd == sd  # unchanged dict
        assert config.skip_normalization_weights is False  # still consumed
        assert any("had no effect" in rec.getMessage() for rec in caplog.records), (
            f"expected a WARNING mentioning 'had no effect'; got {[r.getMessage() for r in caplog.records]!r}"
        )

    def test_info_lists_dropped_count(self, caplog):
        """Flag on with matching keys ⇒ helper logs an INFO listing the
        dropped key count so operators can grep the startup log."""
        parent, normalize_keys, unnormalize_keys = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        config = _FakeConfig(skip_normalization_weights=True)

        with caplog.at_level(logging.INFO):
            _, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert len(stripped) == len(normalize_keys | unnormalize_keys)
        assert any(f"dropped {len(stripped)} saved" in rec.getMessage() for rec in caplog.records), (
            "expected an INFO line listing the dropped key count; "
            f"got {[r.getMessage() for r in caplog.records]!r}"
        )

    def test_main_process_gates_logging(self, caplog):
        """``is_main_process=False`` ⇒ no log lines, even when keys are
        dropped. Prevents distributed runs from duplicating the message
        on every rank."""
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        config = _FakeConfig(skip_normalization_weights=True)

        with caplog.at_level(logging.INFO):
            _, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(
                sd, config, is_main_process=False
            )

        assert stripped, "fixture must produce at least one stripped key"
        assert not any("skip_normalization_weights" in rec.getMessage() for rec in caplog.records), (
            "no skip_normalization_weights log lines should appear on non-main ranks"
        )


class TestAssertNormalizeBuffersInitialized:
    """Integration tests for the post-load inf-buffer guard."""

    def test_no_op_when_stripped_keys_empty(self):
        """``stripped_keys`` empty ⇒ guard returns without inspecting the
        model. This is the default-load path; running the inf-check
        unconditionally would crash bare-config flows that legitimately
        load weights before ``dataset_stats`` arrives.
        """
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        # parent still has inf buffers — but stripped_keys is empty, so
        # the guard must NOT raise.
        PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=frozenset())

    def test_raises_when_inf_buffers_remain_after_strip(self):
        """``stripped_keys`` non-empty + at least one Normalize buffer at
        ``inf`` ⇒ ValueError pointing the user at ``per_dataset_stats``
        (the actual fix), not at the misleading "use a pretrained model"
        assertion that ``Normalize.forward`` would raise on the next
        batch.
        """
        parent, normalize_keys, _ = _make_fake_module_with_normalize_buffers(stats=None)

        with pytest.raises(ValueError, match=r"skip_normalization_weights=True requires `per_dataset_stats`"):
            PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=normalize_keys)

    def test_passes_when_stats_passed_through(self):
        """``stripped_keys`` non-empty but ``dataset_stats`` did fill in
        the buffers ⇒ no inf params ⇒ guard returns silently. This is
        the success path the strip is designed for.
        """
        parent, _, unnormalize_keys = _make_fake_module_with_normalize_buffers(
            stats={
                "observation.state": {
                    "mean": np.zeros(4, dtype=np.float32),
                    "std": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "mean": np.zeros(3, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32),
                },
            }
        )
        # No exception raised.
        PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=unnormalize_keys)
