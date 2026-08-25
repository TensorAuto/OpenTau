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

"""No script may compile a policy's ``sample_actions`` without the rollout-state gate.

A policy whose ``carries_rollout_state`` is True mutates Python-level state
inside ``sample_actions`` — pi05_ttt carries per-episode fast weights and an
integer token position that feeds RoPE. Compiling that either recompiles once
per distinct position and falls back to eager forever, or specializes the
position into the graph and silently freezes the phase. Neither raises.

This is an AST sweep for the same reason ``test_siglip_embedding_dtype.py`` is
one, and CLAUDE.md rule 6 spells out why: the SigLIP cast was swept by hand
across six files, ``robocasa/server.py`` was missed, and the lesson recorded was
"don't grep narrowly and assume the sweep is done". That is exactly what
happened again here — a first pass gated only ``grpc/server.py``, a review
caught three more sites by grepping ``attempt_torch_compile``, and *that* grep
still missed ``benchmark_inference.py``, which calls ``torch.compile`` directly.
Three manual passes, three misses, same file family. So the rule is machine-checked
now, over both spellings.

New entry points should call ``policies.utils.maybe_compile_sample_actions``.
If a site genuinely must compile a sampler itself, add it to
:data:`_GATED_DIRECT_COMPILE` with the guard in place and a reason.
"""

import ast
from pathlib import Path

import pytest

import opentau

_SCRIPTS_ROOT = Path(opentau.__file__).parent / "scripts"

#: Sites that call a compile helper on a sampler directly, each of which must
#: guard on ``carries_rollout_state`` in the same function. Keyed by module path
#: relative to ``scripts/``.
_GATED_DIRECT_COMPILE = {
    # Raises rather than skipping: a benchmark that silently measured an
    # uncompiled sampler would report the wrong number, which is worse here
    # than refusing to run.
    "benchmark_inference.py",
}

#: Names whose application to a ``*sample_actions`` callable counts as compiling it.
_COMPILE_CALLEES = {"attempt_torch_compile", "torch.compile", "compile"}


def _callee_name(call: ast.Call) -> str:
    """Renders a call's callee as a dotted string.

    Args:
        call: The call node.

    Returns:
        Dotted name, e.g. ``torch.compile``, or "" when it is not a plain name.
    """
    node = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _mentions_sample_actions(call: ast.Call) -> bool:
    """Whether any positional argument refers to a ``sample_actions`` attribute.

    Args:
        call: The call node.

    Returns:
        True when the call is applied to something named ``*sample_actions``.
    """
    return any(isinstance(arg, ast.Attribute) and arg.attr == "sample_actions" for arg in call.args)


def _enclosing_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """All function definitions in a module, including nested ones.

    Args:
        tree: Parsed module.

    Returns:
        Every ``FunctionDef`` node.
    """
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _script_files() -> list[Path]:
    """Collects every Python file under ``scripts/``.

    Returns:
        Sorted list of paths.
    """
    return sorted(_SCRIPTS_ROOT.rglob("*.py"))


def test_scripts_root_is_found():
    """Guards the guard: an empty sweep must fail rather than pass."""
    assert _SCRIPTS_ROOT.is_dir(), f"scripts/ not found at {_SCRIPTS_ROOT}"
    assert _script_files(), f"no Python files found under {_SCRIPTS_ROOT}"


@pytest.mark.parametrize("path", _script_files(), ids=lambda p: p.name)
def test_no_script_compiles_a_sampler_without_the_rollout_state_gate(path: Path):
    """A direct compile of ``sample_actions`` must be gated, or routed through the helper.

    Args:
        path: Script under test.
    """
    rel = str(path.relative_to(_SCRIPTS_ROOT))
    tree = ast.parse(path.read_text())

    for func in _enclosing_functions(tree):
        direct = [
            call
            for call in ast.walk(func)
            if isinstance(call, ast.Call)
            and _callee_name(call) in _COMPILE_CALLEES
            and _mentions_sample_actions(call)
        ]
        if not direct:
            continue
        assert rel in _GATED_DIRECT_COMPILE, (
            f"{rel}::{func.name} compiles sample_actions directly. Route it through "
            "`opentau.policies.utils.maybe_compile_sample_actions`, or add the file to "
            "_GATED_DIRECT_COMPILE with an in-function carries_rollout_state guard."
        )
        assert "carries_rollout_state" in ast.dump(func), (
            f"{rel}::{func.name} is allowlisted for a direct compile but does not guard on "
            "carries_rollout_state, so a state-carrying policy would be compiled anyway."
        )


def test_the_allowlist_has_no_stale_entries():
    """An allowlisted file that no longer compiles a sampler must be removed.

    Without this the allowlist grows monotonically and stops meaning anything —
    the "two hand-maintained lists fail open" failure CLAUDE.md rule 5 names.
    """
    still_compiling = set()
    for path in _script_files():
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and _callee_name(call) in _COMPILE_CALLEES
                and _mentions_sample_actions(call)
            ):
                still_compiling.add(str(path.relative_to(_SCRIPTS_ROOT)))
    stale = _GATED_DIRECT_COMPILE - still_compiling
    assert not stale, f"_GATED_DIRECT_COMPILE entries no longer compile a sampler: {sorted(stale)}"


def test_base_policy_declares_the_flag_off_by_default():
    """The contract must live on the base class, not on ``getattr`` defaults.

    Every consumer reads ``policy.carries_rollout_state``; if only the one
    subclass that needs it declared the attribute, each consumer would need a
    ``getattr`` default and a missing one would read as False.
    """
    from opentau.policies.pretrained import PreTrainedPolicy

    assert PreTrainedPolicy.carries_rollout_state is False


def test_pi05_ttt_declares_it_and_opts_out_of_candidates():
    """The one policy that carries state must say so on both axes."""
    from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTPolicy

    assert PI05TTTPolicy.carries_rollout_state is True
    assert PI05TTTPolicy.supports_candidate_sampling is False
