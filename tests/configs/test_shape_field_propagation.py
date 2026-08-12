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

"""Pins the Standard-Data-Format width contract between the pipeline and the policy.

``TrainPipelineConfig`` pushes ``max_state_dim`` / ``max_action_dim`` / ``action_chunk``
onto the policy config, because the dataloader shapes every sample from the *pipeline*
values while the policy sizes its projections from its *own* fields. The three must move
together or the model is built for tensors it is never fed.

Until this module existed nothing asserted that. The propagation had assigned to
``policy.max_action_state`` -- a name no policy config declares -- since the initial
commit, so ``max_action_dim`` never actually propagated while its two siblings did, and
the resulting split was invisible to the suite.
"""

import ast
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from opentau.configs import parser
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.train import PIPELINE_TO_POLICY_SHAPE_FIELDS, TrainPipelineConfig
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.value.configuration_value import ValueConfig

TRAIN_CONFIG_SOURCE = Path("src/opentau/configs/train.py")
ARTIFACT_DIR = Path("tests/artifacts/configs")

# Deliberately disagrees with every policy default (32 / 32 / 50) so a test that passes
# only because both sides happen to hold the same number cannot exist here.
PIPELINE_WIDTHS = {"max_state_dim": 7, "max_action_dim": 9, "action_chunk": 13}


def _all_policy_configs():
    """Every *production* policy config class, keyed by its ``policy.type`` choice.

    Filtered to ``opentau.*`` classes because importing a test module that registers its
    own ``PreTrainedConfig`` subclass would otherwise change this list depending on which
    tests ran -- collection has to be identical across ``-n auto`` workers.
    """
    return sorted(
        (name, cls)
        for name, cls in PreTrainedConfig.get_known_choices().items()
        if cls.__module__.startswith("opentau.")
    )


def _declared_fields(policy_config) -> set[str]:
    return {f.name for f in dataclasses.fields(type(policy_config))}


def _make_cfg(dataset_mixture_config, policy_config, **overrides):
    kwargs = {
        "dataset_mixture": dataset_mixture_config,
        "policy": policy_config,
        "batch_size": 8,
        **PIPELINE_WIDTHS,
        **overrides,
    }
    # `action_chunk` lands on `policy.chunk_size`, and the policy rejects
    # `n_action_steps > chunk_size` once it does (`validate_action_horizon`).
    # PIPELINE_WIDTHS deliberately disagrees with every policy default -- including the
    # default 50-step execution horizon -- so the horizon has to come along, which is
    # exactly what a real config that sets `action_chunk` has to do. It is not itself a
    # propagated width, so nothing in this module asserts on it.
    if "n_action_steps" in _declared_fields(policy_config):
        policy_config.n_action_steps = kwargs["action_chunk"]
    return TrainPipelineConfig(**kwargs)


def test_all_three_widths_move_together(dataset_mixture_config, policy_config):
    """The direct regression: ``max_action_dim`` must propagate like its siblings.

    With the ``max_action_state`` typo in place, the two sibling assertions passed and
    only the ``max_action_dim`` one failed -- which is exactly the production symptom.
    """
    cfg = _make_cfg(dataset_mixture_config, policy_config)

    assert cfg.policy.max_state_dim == 7
    assert cfg.policy.max_action_dim == 9
    assert cfg.policy.chunk_size == 13


def test_pipeline_width_wins_over_a_disagreeing_policy_width(dataset_mixture_config, policy_config):
    """The conflict case: both sides are set, to *different* values. Pipeline wins.

    Passing only one side would leave the outcome consistent with either precedence,
    so the disagreement has to be constructed explicitly.
    """
    policy_config.max_state_dim = 21
    policy_config.max_action_dim = 22
    policy_config.chunk_size = 23

    cfg = _make_cfg(dataset_mixture_config, policy_config)

    assert (cfg.policy.max_state_dim, cfg.policy.max_action_dim, cfg.policy.chunk_size) == (
        7,
        9,
        13,
    ), "the pipeline-level widths must override the policy-level ones, not the reverse"


def test_pipeline_width_wins_over_the_width_carried_by_a_pretrained_checkpoint(
    dataset_mixture_config, tmp_path
):
    """``validate()`` re-applies the widths after ``--policy.path`` replaces the policy.

    This is the reported failure: a 6-DoF checkpoint loaded under the default 32-wide
    pipeline. The checkpoint's widths must not survive, because the dataloader is
    already padding to the pipeline's.
    """
    checkpoint_policy = {
        **json.loads((ARTIFACT_DIR / "train_config.json").read_text())["policy"],
        # a 6-DoF arm, narrower than the pipeline defaults on every axis
        "max_state_dim": 6,
        "max_action_dim": 6,
        "chunk_size": 8,
        "n_action_steps": 8,  # PI0Config requires n_action_steps <= chunk_size
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_policy))

    with patch.object(parser, "get_path_arg", return_value=tmp_path):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=None,
            output_dir=str(tmp_path / "run"),
            job_name="test_run",
            batch_size=8,
            use_policy_training_preset=True,
            **PIPELINE_WIDTHS,
        )
        cfg.validate()

    assert cfg.policy.max_state_dim == 7
    assert cfg.policy.max_action_dim == 9, (
        "validate() must re-apply max_action_dim after the pretrained reload; "
        "leaving the checkpoint's value is the bug this module exists for"
    )
    assert cfg.policy.chunk_size == 13


@pytest.mark.parametrize("policy_type,policy_cls", _all_policy_configs())
def test_propagation_never_creates_an_attribute_the_policy_does_not_declare(
    dataset_mixture_config, policy_type, policy_cls
):
    """Undeclared targets must be skipped, not invented.

    ``PreTrainedConfig`` subclasses are plain unslotted dataclasses, so ``setattr`` of a
    misspelled or inapplicable name silently succeeds and nothing ever reads it. Swept
    over the whole registry because the policies that lack a width differ per field:
    the two high-level planners declare neither ``chunk_size`` nor ``max_action_dim``,
    and ``value`` declares ``chunk_size`` but not ``max_action_dim``.
    """
    policy = policy_cls()
    declared = _declared_fields(policy)
    before = set(vars(policy))

    cfg = _make_cfg(dataset_mixture_config, policy)

    invented = set(vars(cfg.policy)) - before
    assert not invented, f"{policy_type} gained undeclared attribute(s): {sorted(invented)}"

    for pipeline_field, policy_field in PIPELINE_TO_POLICY_SHAPE_FIELDS.items():
        if policy_field in declared:
            assert getattr(cfg.policy, policy_field) == PIPELINE_WIDTHS[pipeline_field]
        else:
            assert not hasattr(cfg.policy, policy_field)


def test_the_policies_that_lack_a_width_are_exactly_the_known_ones():
    """Pins *why* the presence check is load-bearing, across the whole registry.

    Without this, someone could delete the check, watch the suite stay green on the
    policies that declare all three fields, and never learn that the planners and
    ``value`` are the cases it exists for.
    """
    missing = {
        policy_field: {
            name
            for name, cls in _all_policy_configs()
            if policy_field not in {f.name for f in dataclasses.fields(cls)}
        }
        for policy_field in PIPELINE_TO_POLICY_SHAPE_FIELDS.values()
    }

    assert missing == {
        "max_state_dim": set(),
        "max_action_dim": {
            "pi07_high_level",
            "pi07_paligemma_high_level_planner",
            "value",
        },
        "chunk_size": {"pi07_high_level", "pi07_paligemma_high_level_planner"},
    }, (
        f"registry width coverage changed: {missing}. A policy that newly declares one of "
        "these fields will start receiving the pipeline value (check that is intended); "
        "one that drops a field must not start collecting a junk attribute instead."
    )


def test_the_misspelled_target_is_gone_everywhere():
    """``max_action_state`` is not a field of any policy config, so it must never be set.

    Checked over attribute nodes rather than raw source text, so the prose above -- which
    names the typo deliberately -- does not itself trip the assertion.
    """
    assert "max_action_state" not in PIPELINE_TO_POLICY_SHAPE_FIELDS.values()

    touched = [
        node.lineno
        for node in ast.walk(_train_config_ast())
        if isinstance(node, ast.Attribute) and node.attr == "max_action_state"
    ]
    assert not touched, f"train.py still references .max_action_state at line(s) {touched}"

    for _, policy_cls in _all_policy_configs():
        assert "max_action_state" not in {f.name for f in dataclasses.fields(policy_cls)}


def test_overriding_a_differing_width_warns_and_an_agreeing_one_stays_quiet(
    dataset_mixture_config, policy_config, caplog
):
    """The silent clobber is what made the reported bug hard to see."""
    policy_config.max_action_dim = 6

    with caplog.at_level("WARNING"):
        cfg = _make_cfg(dataset_mixture_config, policy_config)
    assert "policy.max_action_dim=6" in caplog.text

    # A second pass now agrees on every width, so it must not re-warn.
    caplog.clear()
    with caplog.at_level("WARNING"):
        cfg._propagate_shape_fields_to_policy()
    assert "max_action_dim" not in caplog.text


def _train_config_ast() -> ast.Module:
    return ast.parse(TRAIN_CONFIG_SOURCE.read_text())


def test_both_call_sites_go_through_the_single_helper():
    """``__post_init__`` and ``validate()`` must not each hand-roll the propagation.

    The typo survived partly because the block was duplicated: a reader fixing one copy
    has no signal that a second exists. Pinning the call keeps them from drifting apart
    again.
    """
    class_node = next(
        n
        for n in ast.walk(_train_config_ast())
        if isinstance(n, ast.ClassDef) and n.name == "TrainPipelineConfig"
    )
    methods = {n.name: n for n in class_node.body if isinstance(n, ast.FunctionDef)}

    for method in ("__post_init__", "validate"):
        calls = {
            node.func.attr
            for node in ast.walk(methods[method])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_propagate_shape_fields_to_policy" in calls, (
            f"{method}() must delegate the width propagation to the shared helper"
        )


def test_no_shape_width_is_assigned_to_the_policy_outside_the_helper():
    """Guards the helper's presence check from being bypassed by a direct ``setattr``.

    A bare ``self.policy.<width> = ...`` re-opens both failure modes at once: it can
    invent an attribute on a policy that does not declare the field, and it can move one
    width without its siblings.
    """
    guarded = set(PIPELINE_TO_POLICY_SHAPE_FIELDS.values())
    offenders = []

    for node in ast.walk(_train_config_ast()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in guarded
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "policy"
            ):
                offenders.append(f"line {target.lineno}: self.policy.{target.attr}")

    assert not offenders, (
        "assign policy widths only inside _propagate_shape_fields_to_policy(), which "
        f"skips fields the policy does not declare; found: {offenders}"
    )


# --- the policy invariants keyed on the propagated widths --------------------------
#
# `chunk_size` is not just carried; the policy constrains it. `n_action_steps` (the
# inference execution horizon) must be <= `chunk_size`, and a shortened horizon cannot
# be combined with the delay knob. Those checks run in the policy config's
# `__post_init__` -- which is over by the time the propagation loop above assigns -- so
# they were only ever checked against the `chunk_size` the policy *declared*, never
# against the propagated `action_chunk`. `_propagate_shape_fields_to_policy` now re-runs
# them through `PreTrainedConfig.validate_action_horizon`.

POLICIES_WITH_EXECUTION_HORIZON = [
    (name, cls)
    for name, cls in _all_policy_configs()
    if {"chunk_size", "n_action_steps"} <= {f.name for f in dataclasses.fields(cls)}
]


def test_action_chunk_below_the_policy_execution_horizon_is_rejected(dataset_mixture_config, policy_config):
    """The reported case: `action_chunk` set, `n_action_steps` left at its default.

    `PI0Config(n_action_steps=50)` under `action_chunk=13` is exactly the pair the
    policy config raises on when it appears in a JSON config, and it used to be
    reachable silently through propagation.
    """
    with pytest.raises(ValueError, match="n_action_steps") as exc_info:
        TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=policy_config,  # n_action_steps defaults to 50
            batch_size=8,
            action_chunk=13,
        )
    # Blaming `policy.chunk_size` alone would send the reader to a field that reads as
    # untouched in their config; the pipeline field that overwrote it has to be named.
    assert "action_chunk" in str(exc_info.value)


def test_action_chunk_matching_the_execution_horizon_propagates(dataset_mixture_config, policy_config):
    """The legal case still propagates -- the guard rejects only the contradiction."""
    policy_config.n_action_steps = 13

    cfg = _make_cfg(dataset_mixture_config, policy_config)

    assert (cfg.policy.chunk_size, cfg.policy.n_action_steps) == (13, 13)


def test_the_delay_knob_is_rechecked_against_the_propagated_chunk_size(dataset_mixture_config):
    """Re-validation covers every `chunk_size` invariant, not only `n_action_steps`.

    This policy config is **valid standalone** -- its horizon is not shortened, since
    `n_action_steps == chunk_size` -- and becomes illegal only once `action_chunk`
    widens `chunk_size` underneath it.
    """
    policy = PI05Config(chunk_size=10, n_action_steps=10, max_delay=5)

    with pytest.raises(ValueError, match="max_delay"):
        TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=policy,
            batch_size=8,
            action_chunk=50,
        )


def test_a_policy_without_an_execution_horizon_is_left_alone(dataset_mixture_config):
    """`value` declares `chunk_size` but no `n_action_steps`, so there is nothing to
    contradict -- it must still receive the width rather than trip a guard.
    """
    cfg = _make_cfg(dataset_mixture_config, ValueConfig())

    assert cfg.policy.chunk_size == 13
    assert not hasattr(cfg.policy, "n_action_steps")


@pytest.mark.parametrize("policy_type,policy_cls", POLICIES_WITH_EXECUTION_HORIZON)
def test_the_propagated_width_is_rechecked_for_every_policy(dataset_mixture_config, policy_type, policy_cls):
    """Swept over the registry, like the presence check above.

    Each of these configs carried its own copy of the horizon invariant, so a policy
    that stops routing through `validate_action_horizon` -- or a new one that never
    starts -- silently regains the dodge.
    """
    policy = policy_cls()  # defaults hold n_action_steps == chunk_size

    with pytest.raises(ValueError, match="n_action_steps"):
        TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=policy,
            batch_size=8,
            action_chunk=policy.n_action_steps - 1,
        )


@pytest.mark.parametrize("policy_type,policy_cls", POLICIES_WITH_EXECUTION_HORIZON)
def test_the_policy_config_still_enforces_the_horizon_itself(policy_type, policy_cls):
    """The other half: deduplicating the check must not lose it from any config.

    Policy configs are also built directly -- eval, inference, notebooks -- where no
    pipeline propagation runs to catch it for them.
    """
    with pytest.raises(ValueError, match="n_action_steps"):
        policy_cls(chunk_size=10, n_action_steps=50)


def test_validate_rechecks_the_horizon_after_the_pretrained_reload(dataset_mixture_config, tmp_path):
    """`validate()` replaces `self.policy` wholesale, so it needs the same guard.

    Same route as the width test above: `--policy.path` loads a checkpoint's config
    *after* `__post_init__` already propagated onto a different (here `None`) policy.
    """
    checkpoint_policy = json.loads((ARTIFACT_DIR / "train_config.json").read_text())["policy"]
    assert checkpoint_policy["n_action_steps"] == 50, "artifact no longer exercises the conflict"
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_policy))

    with patch.object(parser, "get_path_arg", return_value=tmp_path):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=None,
            output_dir=str(tmp_path / "run"),
            job_name="test_run",
            batch_size=8,
            use_policy_training_preset=True,
            **PIPELINE_WIDTHS,  # action_chunk=13 against the checkpoint's 50-step horizon
        )

        with pytest.raises(ValueError, match="n_action_steps"):
            cfg.validate()
