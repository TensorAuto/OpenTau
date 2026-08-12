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

"""Structural invariants for the ``accel`` wiring, across every flow-matching policy.

Seven policies carry their own copy of the Euler denoising loop — nothing is inherited,
only ``paligemma_with_expert`` is shared — so the ``accel`` read
(:mod:`opentau.policies.accel`) had to be applied seven times. The behavioral numerics are
pinned once, on pi05, in ``tests/policies/test_pi05_accel.py``; duplicating that suite six
more times would pin the same arithmetic against six near-identical stubs while still
missing the thing that actually differs between the copies — *where in each loop the read
sits*.

So these are AST checks. They cover all seven policies without constructing a
multi-billion-parameter model, and each one is written to fail on the specific silent
corruption it names:

1. ``accel.update(...)`` must read the **same** velocity variable the Euler step consumes,
   and must run **before** the state update. Reading after the update, or reconstructing
   the velocity from consecutive states, silently scores the wrong field wherever a
   real-time-chunking clamp overwrites rows mid-loop.
2. The row mask must be set, or the score silently includes frozen and un-executed rows.
3. The inner sampler's ``accel`` parameter must be last and default to ``None``, or a
   positional caller (the ONNX exporter, the compiled-callable rebinds) breaks.
4. ``select_action`` must clear ``last_accel``, or a queue-pop step is indistinguishable
   from a re-plan and a consumer records the same score ``n_action_steps`` times.
5. ``reset`` must clear the accel state, or it leaks across episodes.

A new flow-matching policy that forgets any of this fails here rather than at the point
where somebody trusts a number it produced.
"""

import ast
import re
from pathlib import Path

import pytest

import opentau

_POLICIES_ROOT = Path(opentau.__file__).parent / "policies"

#: Every policy carrying its own flow-matching Euler loop, as
#: ``(module path, inner sampler class, policy wrapper class)``. Not auto-discovered on
#: purpose: adding a policy here should be a conscious act, and the list doubles as the
#: inventory of "places a change to the denoise loop has to land".
FLOW_MATCHING_POLICIES = [
    ("pi0/modeling_pi0.py", "PI0FlowMatching", "PI0Policy"),
    ("pi05/modeling_pi05.py", "PI05FlowMatching", "PI05Policy"),
    ("pi05_mem/modeling_pi05.py", "PI05MemFlowMatching", "PI05MemPolicy"),
    ("pi06/modeling_pi06.py", "PI06FlowMatching", "PI06Policy"),
    (
        "pi07/low_level/modeling_pi07_low_level.py",
        "PI07LowLevelFlowMatching",
        "PI07LowLevelPolicy",
    ),
    (
        "pi07_paligemma/low_level/modeling_pi07_low_level.py",
        "PI07PaligemmaLowLevelFlowMatching",
        "PI07PaligemmaLowLevelPolicy",
    ),
    ("cosmos3/modeling_cosmos3.py", "Cosmos3FlowMatching", "Cosmos3Policy"),
]

IDS = [entry[0] for entry in FLOW_MATCHING_POLICIES]


def _tree(rel_path: str) -> ast.Module:
    return ast.parse((_POLICIES_ROOT / rel_path).read_text())


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef:
    for node in _class(tree, class_name).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method:
            return node
    raise AssertionError(f"{class_name}.{method} not found")


def _accel_method_calls(node: ast.AST, method: str) -> list[ast.Call]:
    """Every ``<name>.<method>(...)`` call on a variable named ``accel``."""
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == method
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "accel"
    ]


def _euler_update_lines(sampler: ast.FunctionDef) -> list[tuple[int, str]]:
    """Lines that advance the denoise state, as ``(lineno, state variable)``.

    Matches both shapes in the tree: in-place ``x_t += dt * v_t`` (pi0/pi05 family) and
    out-of-place ``x_t = x_t + dt * v_t`` (cosmos3).
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(sampler):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if any(name.startswith("v") for name in names):
                found.append((node.lineno, node.target.id))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(node.value, ast.BinOp):
                continue
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if target.id in names and any(name.startswith("v") for name in names):
                found.append((node.lineno, target.id))
    return found


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_inner_sampler_takes_accel_last_and_defaults_to_none(rel_path, sampler_cls, policy_cls):
    """The parameter must be trailing and optional.

    ``sample_actions`` on the inner module is ``torch.compile``d at four serving entry
    points and ONNX-exported at one, all of which call it positionally. A non-trailing or
    non-defaulted parameter breaks those callers; a missing default also means every caller
    must be updated in lockstep, which is how one gets missed.
    """
    sampler = _method(_tree(rel_path), sampler_cls, "sample_actions")
    args = sampler.args.args + sampler.args.kwonlyargs
    assert args[-1].arg == "accel", f"{sampler_cls}.sample_actions: 'accel' must be the last parameter"

    defaults = sampler.args.defaults + [d for d in sampler.args.kw_defaults if d is not None]
    assert defaults, f"{sampler_cls}.sample_actions: 'accel' must have a default"
    assert isinstance(defaults[-1], ast.Constant) and defaults[-1].value is None, (
        f"{sampler_cls}.sample_actions: 'accel' must default to None so the feature is off "
        "by default and the traced graph is unchanged"
    )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_accel_update_precedes_the_euler_state_update(rel_path, sampler_cls, policy_cls):
    """``accel.update(v_t)`` must run before the state advances.

    Placing it after would still "work" on most configs, which is what makes this worth
    pinning structurally: wherever a real-time-chunking clamp overwrites ``x_t`` at the top
    of the next iteration, the velocity a later read sees no longer corresponds to the step
    it is attributed to.
    """
    sampler = _method(_tree(rel_path), sampler_cls, "sample_actions")
    updates = _accel_method_calls(sampler, "update")
    assert len(updates) == 1, (
        f"{sampler_cls}.sample_actions: expected exactly one accel.update() call, found {len(updates)}"
    )

    euler = _euler_update_lines(sampler)
    assert euler, f"{sampler_cls}.sample_actions: no Euler state update found; update the matcher"
    assert updates[0].lineno < min(line for line, _ in euler), (
        f"{sampler_cls}.sample_actions: accel.update() at line {updates[0].lineno} runs after the "
        f"Euler state update at line {min(line for line, _ in euler)}"
    )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_accel_update_reads_the_loops_own_velocity(rel_path, sampler_cls, policy_cls):
    """The argument must be the same name the Euler step consumes.

    Guards against the two ways this silently goes wrong: passing a *copy* computed by a
    second network call (not free, and not the same field), or passing a velocity
    reconstructed from consecutive states (wrong wherever rows are clamped).
    """
    sampler = _method(_tree(rel_path), sampler_cls, "sample_actions")
    call = _accel_method_calls(sampler, "update")[0]
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name), (
        f"{sampler_cls}.sample_actions: accel.update() must be passed the loop's velocity "
        "variable directly, not an expression"
    )
    velocity = call.args[0].id

    consumed = {
        name.id
        for lineno, _ in _euler_update_lines(sampler)
        for node in ast.walk(sampler)
        if getattr(node, "lineno", None) == lineno
        for name in ast.walk(node)
        if isinstance(name, ast.Name)
    }
    assert velocity in consumed, (
        f"{sampler_cls}.sample_actions: accel.update({velocity}) does not read the variable the "
        f"Euler step consumes ({sorted(consumed)})"
    )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_row_mask_is_set_before_the_loop(rel_path, sampler_cls, policy_cls):
    """Without it, frozen RTC rows and the un-executed chunk tail land in both sums.

    Both carry velocities that describe nothing the robot will do, and both are masked out
    of the training loss for that reason.
    """
    sampler = _method(_tree(rel_path), sampler_cls, "sample_actions")
    calls = _accel_method_calls(sampler, "set_row_mask")
    assert len(calls) == 1, (
        f"{sampler_cls}.sample_actions: expected exactly one accel.set_row_mask() call, found {len(calls)}"
    )
    update = _accel_method_calls(sampler, "update")[0]
    assert calls[0].lineno < update.lineno, (
        f"{sampler_cls}.sample_actions: set_row_mask() must run before the loop's first update()"
    )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_accel_lines_are_guarded_so_they_are_dead_when_disabled(rel_path, sampler_cls, policy_cls):
    """Every ``accel.*`` call must sit under an ``if accel is not None`` test.

    This is what makes the feature free: with ``accel=None`` the branch is a compile-time
    constant, so the traced graph is byte-identical to before the feature existed.
    """
    sampler = _method(_tree(rel_path), sampler_cls, "sample_actions")
    guarded: set[int] = set()
    for node in ast.walk(sampler):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "accel"
            and isinstance(test.ops[0], ast.IsNot)
        ):
            continue
        # `ast.walk` also yields context nodes (`Load`, `Store`, ...) which carry no
        # position, so filter rather than attribute-access blindly.
        guarded |= {
            lineno
            for body in node.body
            for n in ast.walk(body)
            if (lineno := getattr(n, "lineno", None)) is not None
        }

    for method in ("update", "set_row_mask"):
        for call in _accel_method_calls(sampler, method):
            assert call.lineno in guarded, (
                f"{sampler_cls}.sample_actions: accel.{method}() at line {call.lineno} is not "
                "inside an `if accel is not None:` guard, so it is not dead when accel is off"
            )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_select_action_clears_last_accel(rel_path, sampler_cls, policy_cls):
    """One sample call feeds ``n_action_steps`` env steps.

    Without the clear, a consumer reading ``last_accel`` every step records the same score
    ``n_action_steps`` times — inflating the stream and, worse, feeding a conformal
    calibration duplicated samples it treats as independent.
    """
    select = _method(_tree(rel_path), policy_cls, "select_action")
    clears = [
        node.lineno
        for node in ast.walk(select)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        and any(
            isinstance(t, ast.Attribute) and t.attr == "last_accel" and isinstance(t.value, ast.Name)
            for t in node.targets
        )
    ]
    assert clears, f"{policy_cls}.select_action must clear `self.last_accel` so a queue-pop step reads None"


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_reset_clears_the_accel_state(rel_path, sampler_cls, policy_cls):
    """``policy.reset()`` runs once per rollout batch; per-episode state must not survive it."""
    reset = _method(_tree(rel_path), policy_cls, "reset")
    cleared = {
        target.attr
        for node in ast.walk(reset)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }
    assert {"last_accel", "last_accel_provenance"} <= cleared, (
        f"{policy_cls}.reset must clear last_accel and last_accel_provenance, cleared: {sorted(cleared)}"
    )


@pytest.mark.parametrize(("rel_path", "sampler_cls", "policy_cls"), FLOW_MATCHING_POLICIES, ids=IDS)
def test_wrapper_declares_accel_prefix_off_by_default(rel_path, sampler_cls, policy_cls):
    """``accel_prefix`` must be declared in ``__init__`` and initialized to ``None``.

    Declaring it there (rather than letting ``configure_accel`` create it) is what lets
    ``make_meter`` and every consumer use a plain attribute read, and what makes a policy
    family that has *not* been wired detectable via ``hasattr``.
    """
    init = _method(_tree(rel_path), policy_cls, "__init__")
    assigned = [
        node
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Attribute) and target.attr == "accel_prefix"
    ]
    assert assigned, f"{policy_cls}.__init__ must declare `self.accel_prefix`"
    assert all(isinstance(node.value, ast.Constant) and node.value.value is None for node in assigned), (
        f"{policy_cls}.__init__ must initialize `self.accel_prefix` to None (accel is opt-in)"
    )


def test_every_entry_point_that_reads_accel_also_arms_it():
    """A script that consumes ``last_accel`` must also call ``configure_accel``.

    This pins a gap that shipped once and was invisible: ``eval.py`` read ``last_accel``,
    warned about the parallel-task race, and threaded the streams all the way to disk — but
    never armed the proxy, so ``OPENTAU_ACCEL_PREFIX=auto opentau-eval`` silently produced
    empty ``accel`` fields on every episode. Nothing failed; the feature was simply
    unreachable on the path that collects the calibration set.

    The asymmetry is the tell, so that is what this checks: reading without arming is a bug,
    while arming without reading is fine (a server may publish the score on the wire rather
    than consume it itself).

    "Arming" means either calling ``configure_accel`` (the config/env resolution every
    serving entry point uses) or assigning ``accel_prefix`` directly — which ``diagnose_accel``
    legitimately does, since sweeping every prefix is precisely its job and it cannot go
    through a single resolved value.
    """
    scripts_root = Path(opentau.__file__).parent / "scripts"
    offenders = []
    for path in scripts_root.rglob("*.py"):
        source = path.read_text()
        if "last_accel" not in source:
            continue
        arms = "configure_accel" in source or re.search(r"\.accel_prefix\s*=", source)
        if not arms:
            offenders.append(str(path.relative_to(scripts_root)))
    assert not offenders, (
        "These entry points read `policy.last_accel` but never arm the proxy (no "
        "`configure_accel` call and no `accel_prefix` assignment), so accel can never be "
        f"switched on for them and every score comes back empty: {offenders}"
    )


def test_the_registry_covers_every_denoise_loop_in_the_tree():
    """Fail when a policy grows a flow-matching Euler loop without being listed here.

    The whole point of an AST sweep is that it cannot be outgrown silently. A hand-kept
    list that nothing cross-checks is exactly the "two hand-maintained lists fail open"
    failure CLAUDE.md rule 5 describes, so this derives the truth independently: any module
    under ``policies/`` calling ``self.denoise_step(...)`` or otherwise stepping a denoise
    state must appear in :data:`FLOW_MATCHING_POLICIES`.
    """
    listed = {rel for rel, _, _ in FLOW_MATCHING_POLICIES}
    found = set()
    for path in _POLICIES_ROOT.rglob("modeling_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "sample_actions":
                continue
            if _euler_update_lines(node):
                found.add(str(path.relative_to(_POLICIES_ROOT)))

    missing = found - listed
    assert not missing, (
        f"These modules run a flow-matching Euler loop but are not in FLOW_MATCHING_POLICIES, so "
        f"their accel wiring is unpinned: {sorted(missing)}"
    )
    stale = listed - found
    assert not stale, f"These entries no longer have a denoise loop: {sorted(stale)}"
