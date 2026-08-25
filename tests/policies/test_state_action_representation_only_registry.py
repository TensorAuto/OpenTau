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

"""Registry-wide invariants for ``train_state_action_representation_only``.

The flag has to be plumbed through every policy that has a state/action
representation to train, and must be *absent* from the policies that do not
(the two high-level planners and ``value`` own no state/action projections and
no live discrete-action modules, so the flag there would silently train nothing
and produce a flat loss curve with no error).

That "everywhere or nowhere, and we know which" property is exactly what a
hand-maintained per-policy list gets wrong over time: ``train_vision_encoder_only``
reached the pi07_paligemma planner but not the pi07 one, and nothing failed. These
tests pin the capability set by *set equality* so drift in either direction fails.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from dataclasses import fields
from pathlib import Path

import draccus
import pytest

from opentau.policies.factory import make_policy_config

FLAG = "train_state_action_representation_only"

#: Every policy type ``make_policy_config`` accepts. Deliberately NOT derived from
#: ``opentau.available_policies``, which omits ``pi07_paligemma_low_level``,
#: ``pi07_paligemma_high_level_planner`` and ``pi05_continuous_state`` — iterating
#: that list would let the whole pi07_paligemma family escape this sweep.
ALL_POLICY_TYPES = frozenset(
    {
        "pi0",
        "pi05",
        "pi05_continuous_state",
        "pi05_mem",
        "pi05_ttt",
        "pi06",
        "pi07_paligemma_high_level_planner",
        "pi07_paligemma_low_level",
        "pi07_high_level",
        "pi07_low_level",
        "cosmos3",
        "cosmos3_nano",
        "value",
    }
)

#: Policies that own a state/action representation worth training on its own:
#: state/action projections and/or a discrete-action embedding + head.
EXPECTED_CAPABLE = frozenset(
    {
        "pi0",  # projections only — no discrete-action pathway
        "pi05",
        "pi05_continuous_state",
        "pi05_mem",
        "pi05_ttt",  # a pi05 variant: same projections and discrete-action pathway
        "pi06",  # no state_proj (discrete state), but has the rest
        "pi07_low_level",
        "pi07_paligemma_low_level",
        "cosmos3",  # projections only — no discrete-action pathway
        "cosmos3_nano",
    }
)

#: Policies where the flag must NOT exist: no projections, no live discrete modules.
EXPECTED_INCAPABLE = ALL_POLICY_TYPES - EXPECTED_CAPABLE


def _config_for(policy_type: str):
    with warnings.catch_warnings():
        # pi05_continuous_state is deprecated but still routable.
        warnings.simplefilter("ignore", DeprecationWarning)
        return make_policy_config(policy_type)


def test_factory_branch_list_is_complete():
    """``ALL_POLICY_TYPES`` must equal the factory's actual ``policy_type == "..."``
    branches.

    The load-bearing direction is *new* policy types: a policy added to the factory
    but not to ``ALL_POLICY_TYPES`` would never be swept by the set-equality test
    below, so the flag could be forgotten on it silently. Parsing the comparisons
    (rather than every string literal in the function) is what makes that direction
    detectable.
    """
    tree = ast.parse(inspect.getsource(make_policy_config).lstrip())
    branch_types = {
        comparator.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "policy_type"
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }

    assert branch_types == set(ALL_POLICY_TYPES), (
        f"policy types in the factory but not in ALL_POLICY_TYPES: "
        f"{sorted(branch_types - ALL_POLICY_TYPES)}; "
        f"in ALL_POLICY_TYPES but not in the factory: {sorted(ALL_POLICY_TYPES - branch_types)}"
    )


def test_exactly_the_capable_policies_carry_the_flag():
    """Set equality in both directions: no capable policy is missing the flag, and
    no incapable policy grew one."""
    carrying = {t for t in ALL_POLICY_TYPES if FLAG in {f.name for f in fields(_config_for(t))}}

    assert carrying == set(EXPECTED_CAPABLE), (
        f"missing the flag: {sorted(EXPECTED_CAPABLE - carrying)}; "
        f"unexpectedly carrying it: {sorted(carrying - EXPECTED_CAPABLE)}"
    )


@pytest.mark.parametrize("policy_type", sorted(EXPECTED_CAPABLE))
def test_flag_defaults_to_false(policy_type):
    """Default-off is the no-op guarantee: an existing config must behave exactly
    as it did before this flag existed."""
    assert getattr(_config_for(policy_type), FLAG) is False


@pytest.mark.parametrize("policy_type", sorted(EXPECTED_CAPABLE))
def test_flag_can_be_enabled(policy_type):
    kwargs = {FLAG: True}
    if policy_type in ("cosmos3", "cosmos3_nano"):
        # cosmos3 defaults train_expert_only=True, which is mutually exclusive.
        kwargs["train_expert_only"] = False
    assert getattr(_config_for(policy_type).__class__(**kwargs), FLAG) is True


@pytest.mark.parametrize("policy_type", sorted(EXPECTED_INCAPABLE))
def test_incapable_policies_reject_the_flag(policy_type):
    """A user who sets the flag on a planner or the value policy must get an error,
    not a run that silently trains nothing."""
    with pytest.raises(TypeError):
        _config_for(policy_type).__class__(**{FLAG: True})


@pytest.mark.parametrize("policy_type", sorted(EXPECTED_CAPABLE))
def test_draccus_round_trips_the_flag(policy_type):
    kwargs = {FLAG: True}
    if policy_type in ("cosmos3", "cosmos3_nano"):
        kwargs["train_expert_only"] = False
    cfg = _config_for(policy_type).__class__(**kwargs)

    reparsed = draccus.decode(type(cfg), draccus.encode(cfg))

    assert getattr(reparsed, FLAG) is True


@pytest.mark.parametrize("policy_type", sorted(EXPECTED_CAPABLE))
def test_legacy_config_without_the_field_decodes_to_false(policy_type):
    """A checkpoint config written before this flag existed must still load, and
    must load with the flag off."""
    cfg = _config_for(policy_type)
    payload = draccus.encode(cfg)
    payload.pop(FLAG, None)

    reparsed = draccus.decode(type(cfg), payload)

    assert getattr(reparsed, FLAG) is False


def test_every_modeling_file_with_a_vision_only_freeze_also_handles_this_flag():
    """AST completeness check, in the style of
    ``tests/policies/test_from_pretrained_revision_overrides.py``.

    Any modeling module that calls ``freeze_policy_level_params_for_vision_only``
    owns policy-level projections, and therefore must also call this flag's helper.
    A new policy that copies the vision-only freeze but forgets this one fails here
    rather than silently training its whole trunk.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "opentau" / "policies"
    offenders = []
    for path in sorted(root.rglob("modeling_*.py")):
        source = path.read_text()
        if "freeze_policy_level_params_for_vision_only(" not in source:
            continue
        if "freeze_policy_level_params_for_state_action_representation_only(" not in source:
            offenders.append(str(path.relative_to(root)))

    assert not offenders, (
        "these modeling files freeze policy-level params for train_vision_encoder_only but "
        f"never handle {FLAG}: {offenders}"
    )
