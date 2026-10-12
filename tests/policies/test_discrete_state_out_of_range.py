#!/usr/bin/env python

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

"""An out-of-range discrete state is logged once per process, not once per call.

``prepare_discrete_state`` bins a normalized state into 256 prompt tokens over [-1, 1] and
clamps whatever falls outside. Under QUANTILE / MIN_MAX normalization a real robot's state
routinely overshoots a little, so the warning every copy used to log on each call flooded a
serving log with one line per request. The rate limit now lives in one helper,
``warn_once_if_state_out_of_range`` (``opentau.policies.utils``). These tests call each
hand-duplicated copy of the method directly, with a stub ``self`` (the method reads only
``name`` and, in pi05, ``config.state_type``), so no multi-billion-parameter model is built.
"""

import ast
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import opentau
from opentau.policies import utils as policy_utils
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.pi06.modeling_pi06 import PI06Policy
from opentau.policies.pi07.high_level_planner.modeling_pi07_high_level import (
    PI07HighLevelPlannerPolicy as PI07HighLevelPolicy,
)
from opentau.policies.pi07_paligemma.high_level_planner.modeling_pi07_high_level import (
    PI07HighLevelPlannerPolicy as PI07PaligemmaHighLevelPolicy,
)
from opentau.policies.utils import warn_once_if_state_out_of_range
from opentau.policies.value.modeling_value import ValueFunction

# EVERY hand-duplicated `prepare_discrete_state`, enumerated with
# `grep -rn "def prepare_discrete_state" src/` (pi05_ttt inherits pi05's copy). Inclusion
# criterion: a copy that logs about the out-of-range state it clamps — all five did, and none
# raised. `scripts/onnx_inference.py::_prepare_discrete_state` is out of scope: it clamps without
# logging. `test_every_copy_routes_through_the_helper` fails when a copy appears that is not listed.
POLICY_CLASSES = [
    PI05Policy,
    PI06Policy,
    PI07HighLevelPolicy,
    PI07PaligemmaHighLevelPolicy,
    ValueFunction,
]

# Normalized states past both edges. Clamped into [-1, 1], `x` lands in bin
# `min(floor((x + 1) * 128), 255)`. The torch copies compute exactly that, and value's
# `np.digitize` agrees because every in-range entry sits either on a bin edge or well inside a bin.
OUT_OF_RANGE = torch.tensor(
    [
        [-3.0, -1.0, -0.5, 0.0, 0.5, 0.999, 1.0, 2.5],
        [0.25, -1.5, 1.5, -0.75, 0.75, -0.999, 0.0, 1.0001],
    ]
)
OUT_OF_RANGE_BINS = ["0 0 64 128 192 255 255 255", "160 0 255 32 224 0 128 255"]
IN_RANGE = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])  # exactly +-1 is in range
IN_RANGE_BINS = ["0 64 128 192 255"]


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Start each test as a fresh process would: no policy has warned yet."""
    monkeypatch.setattr(policy_utils, "_OUT_OF_RANGE_STATE_TALLY", {})


def _prepare(policy_cls, state: torch.Tensor) -> list[str]:
    """Run the real (unbound) ``prepare_discrete_state`` of ``policy_cls`` on ``state``."""
    stub_self = SimpleNamespace(name=policy_cls.name, config=SimpleNamespace(state_type="discrete"))
    batch = {"state": state.clone()}
    out = policy_cls.prepare_discrete_state(stub_self, batch)
    assert torch.equal(batch["state"], state), "the clamp must not write back into the caller's batch"
    return out


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "State values are not normalized" in r.getMessage()]


def _summaries(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "has clamped an out-of-range discrete state" in r.getMessage()]


@pytest.mark.parametrize("policy_cls", POLICY_CLASSES, ids=lambda cls: cls.name)
def test_out_of_range_state_warns_once_and_is_still_clamped(policy_cls, caplog):
    with caplog.at_level(logging.INFO):
        outputs = [_prepare(policy_cls, OUT_OF_RANGE) for _ in range(5)]

    # Every call returns the clamped bins: the call that warned and the silent ones alike.
    assert outputs == [OUT_OF_RANGE_BINS] * 5
    warnings = _warnings(caplog)
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert warnings[0].levelno == logging.WARNING
    assert f"policy {policy_cls.name}: min -3, max 2.5" in warnings[0].getMessage()
    assert not _summaries(caplog)  # the first INFO summary is due at the 10th out-of-range call


@pytest.mark.parametrize("policy_cls", POLICY_CLASSES, ids=lambda cls: cls.name)
def test_in_range_state_neither_warns_nor_uses_up_the_warning(policy_cls, caplog):
    with caplog.at_level(logging.INFO):
        assert [_prepare(policy_cls, IN_RANGE) for _ in range(3)] == [IN_RANGE_BINS] * 3
        assert not _warnings(caplog)

        assert _prepare(policy_cls, OUT_OF_RANGE) == OUT_OF_RANGE_BINS
    assert len(_warnings(caplog)) == 1


def test_each_policy_in_one_process_warns_once(caplog):
    """One policy's warning must not silence another's, and neither repeats."""
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            for policy_cls in POLICY_CLASSES:
                _prepare(policy_cls, OUT_OF_RANGE)

    warnings = _warnings(caplog)
    assert len(warnings) == len(POLICY_CLASSES), [r.getMessage() for r in warnings]
    for policy_cls, record in zip(POLICY_CLASSES, warnings, strict=True):
        assert f"(policy {policy_cls.name}:" in record.getMessage()


def test_summaries_follow_the_10th_100th_and_1000th_out_of_range_call(caplog):
    over, under, inside = torch.tensor([[1.25, 0.0]]), torch.tensor([[-2.0, 0.0]]), torch.zeros(1, 2)
    with caplog.at_level(logging.INFO):
        for i in range(1, 1000):
            warn_once_if_state_out_of_range(under if i == 50 else over, "pi05")
            # An in-range call is not counted: were it, the 1000th summary would fire at i == 500.
            warn_once_if_state_out_of_range(inside, "pi05")
        assert len(_summaries(caplog)) == 2

        warn_once_if_state_out_of_range(over, "pi05")

    assert len(_warnings(caplog)) == 1
    summaries = _summaries(caplog)
    assert [r.levelno for r in summaries] == [logging.INFO] * 3
    # The count, and the extremes over every out-of-range call so far (`under` arrives at call 50).
    assert [r.getMessage().split(" has clamped")[1] for r in summaries] == [
        " an out-of-range discrete state in 10 calls so far (min 0, max 1.25 across them).",
        " an out-of-range discrete state in 100 calls so far (min -2, max 1.25 across them).",
        " an out-of-range discrete state in 1000 calls so far (min -2, max 1.25 across them).",
    ]


def test_every_copy_routes_through_the_helper():
    """A new ``prepare_discrete_state`` copy must be listed above and must not log per call."""
    root = Path(opentau.__file__).parent
    copies = [
        (path.relative_to(root), node)
        for path in sorted(root.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_discrete_state"
    ]
    listed = sorted(Path(inspect.getfile(cls)).relative_to(root) for cls in POLICY_CLASSES)
    assert sorted(path for path, _ in copies) == listed

    for path, fn in copies:
        called = {ast.unparse(node.func) for node in ast.walk(fn) if isinstance(node, ast.Call)}
        assert "warn_once_if_state_out_of_range" in called, path
        assert not [c for c in called if c.startswith(("logging.", "logger.", "warnings."))], path
