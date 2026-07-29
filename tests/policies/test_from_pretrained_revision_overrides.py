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

"""Structural invariants for the per-policy ``from_pretrained`` overrides.

Seven policies override ``PreTrainedPolicy.from_pretrained`` to remap state-dict
keys. Each used to resolve ``model.safetensors`` itself via
``transformers.utils.cached_file``, which produced two silent failure modes:

1. Every download-control argument was read as ``kwargs.get("<name>")`` even
   though each is a *keyword-only parameter* of the override — so it never lands
   in ``**kwargs`` and the lookup was unconditionally ``None``. ``revision`` in
   particular meant an ``@<step>`` tag was ignored and ``main`` loaded instead.
2. The resolution was wrapped in a bare ``except Exception`` that logged a
   warning and returned an untrained model, so a typo'd repo or tag produced
   randomly-initialized weights that look like a successful load.

These are AST checks rather than behavioral ones so they cover all seven
policies without constructing a multi-billion-parameter model, and so a
regression is caught at the shape level rather than only when someone happens to
exercise that policy with a revision.
"""

import ast
from pathlib import Path

import pytest

import opentau

_POLICIES_ROOT = Path(opentau.__file__).parent / "policies"

#: Every policy that overrides ``from_pretrained`` **and resolves the weights
#: itself**. Policies whose override just decorates ``super().from_pretrained``
#: (pi0) inherit the base implementation and are excluded — see
#: ``test_every_override_either_delegates_or_is_listed``, which pins that split
#: so a delegating override cannot quietly grow its own resolver. Not
#: auto-discovered on purpose: adding a policy here should be a conscious act.
OVERRIDE_FILES = [
    "pi05/modeling_pi05.py",
    "pi05_mem/modeling_pi05.py",
    "pi06/modeling_pi06.py",
    "pi07/low_level/modeling_pi07_low_level.py",
    "pi07/high_level_planner/modeling_pi07_high_level.py",
    "pi07_paligemma/low_level/modeling_pi07_low_level.py",
    "pi07_paligemma/high_level_planner/modeling_pi07_high_level.py",
]


def _from_pretrained_node(relpath: str) -> tuple[ast.FunctionDef, str]:
    path = _POLICIES_ROOT / relpath
    source = path.read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained":
            return node, source
    raise AssertionError(f"{relpath} declares no from_pretrained override")


def _delegates_to_super(node: ast.FunctionDef) -> bool:
    """True when the override's only weight path is ``super().from_pretrained(...)``."""
    return any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "from_pretrained"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


def test_every_override_either_delegates_or_is_listed():
    """No policy may resolve checkpoint weights outside the reviewed set.

    A ``from_pretrained`` override either delegates to the base implementation
    (which handles ``repo@revision`` and Hub errors) or appears in
    ``OVERRIDE_FILES`` and is held to the invariants below. A third option — a
    new bespoke resolver — is exactly the regression this pins.
    """
    unreviewed = []
    for path in sorted(_POLICIES_ROOT.rglob("modeling_*.py")):
        relpath = str(path.relative_to(_POLICIES_ROOT))
        if relpath in OVERRIDE_FILES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "from_pretrained"
                and not _delegates_to_super(node)
            ):
                unreviewed.append(relpath)
    assert not unreviewed, (
        f"{unreviewed} override from_pretrained without delegating to super(). Add them to "
        f"OVERRIDE_FILES and route weight resolution through resolve_pretrained_weights_file."
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_override_never_reads_its_own_parameters_out_of_kwargs(relpath):
    """``kwargs.get("revision")`` on a keyword-only param is always ``None``.

    Python puts named keyword-only parameters in their own binding, never in
    ``**kwargs``, so this pattern silently discards whatever the caller passed.
    """
    node, _ = _from_pretrained_node(relpath)
    declared = {a.arg for a in node.args.args + node.args.kwonlyargs}
    offenders = sorted(
        {
            call.args[0].value
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "kwargs"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value in declared
        }
    )
    assert not offenders, (
        f"{relpath}: from_pretrained reads its own keyword-only parameter(s) "
        f"{offenders} out of **kwargs, where they can never appear. Use the "
        f"parameter directly."
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_override_resolves_weights_through_the_shared_helper(relpath):
    """One resolver for every policy, so ``@<step>`` and Hub errors behave alike.

    A bespoke ``cached_file`` call does not understand the ``repo_id@revision``
    spec and reintroduces the swallowed-failure path this test exists to prevent.
    """
    _, source = _from_pretrained_node(relpath)
    assert "cached_file" not in source, (
        f"{relpath}: resolves checkpoint weights with cached_file. Use "
        f"resolve_pretrained_weights_file so repo@revision specs and Hub errors "
        f"are handled identically across policies."
    )
    assert source.count("resolve_pretrained_weights_file(") == 1, (
        f"{relpath}: expected exactly one resolve_pretrained_weights_file(...) call"
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_override_splits_the_revision_suffix(relpath):
    """The spec must be split before it is used as a repo id or a path."""
    node, _ = _from_pretrained_node(relpath)
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "split_repo_revision"
    ]
    assert len(calls) == 1, (
        f"{relpath}: expected exactly one split_repo_revision(...) call in from_pretrained"
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_override_resolves_checkpoint_provenance(relpath):
    """Every override must key its config to the weights it is about to load.

    ``resolve_checkpoint_provenance`` resolves ``config_version`` from the
    checkpoint (absent tag -> legacy 0) and runs the input-resolution check. The
    base class always did this; none of the overrides did, and because
    ``make_policy`` always passes ``config=``, their
    ``if config is None: PreTrainedConfig.from_pretrained(...)`` branch never
    fires — so every production policy silently applied the *current*
    normalization convention to legacy weights.
    """
    _, source = _from_pretrained_node(relpath)
    assert source.count("resolve_checkpoint_provenance(") == 1, (
        f"{relpath}: expected exactly one resolve_checkpoint_provenance(...) call"
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_provenance_is_resolved_before_the_model_is_constructed(relpath):
    """Normalize/Unnormalize are built from ``config_version`` in ``cls(config)``.

    Resolving the convention after construction would leave the modules built
    against the wrong one, which is silent — the weights load fine and the
    numbers are subtly wrong.
    """
    node, _ = _from_pretrained_node(relpath)
    provenance_line = next(
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_checkpoint_provenance"
    )
    construct_line = min(
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "cls"
    )
    assert provenance_line < construct_line, (
        f"{relpath}: resolve_checkpoint_provenance must run before cls(config, ...) "
        f"(got line {provenance_line} vs {construct_line})"
    )


@pytest.mark.parametrize("relpath", OVERRIDE_FILES)
def test_override_never_passes_the_raw_spec_to_a_loader(relpath):
    """After splitting, downstream calls must use the split repo id.

    Forwarding the raw ``pretrained_name_or_path`` would hand ``repo@6000`` to
    ``hf_hub_download``, which rejects "@" in a repo id.
    """
    node, _ = _from_pretrained_node(relpath)
    leaked = [
        kw.value.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg == "pretrained_name_or_path"
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "pretrained_name_or_path"
    ]
    assert not leaked, (
        f"{relpath}: forwards the unsplit pretrained_name_or_path to a loader; "
        f"pass the repo id returned by split_repo_revision instead."
    )
