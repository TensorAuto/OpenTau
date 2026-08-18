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

"""Every entry point that can reach the sampler must arm or refuse best-of-N.

``n_candidates`` lives on ``PreTrainedConfig``, so it is serialized into every checkpoint's
``config.json`` and travels with it. Any script pointed at such a checkpoint therefore *sees*
the field — and the failure mode worth spending a test on is the one that produces no error at
all: a script that reads the config, ignores the field, and leaves an operator believing
best-of-N is running when it is not.

So the rule is binary, and this module pins it structurally. A script that builds a policy from
``cfg.policy`` **and** reaches that policy's sampler must either

* call :func:`opentau.policies.candidates.configure_candidates` (arm it), or
* call :func:`opentau.policies.candidates.refuse_candidates` (raise, with a reason naming what
  would go wrong).

Scripts that build a policy but never touch ``sample_actions`` / ``select_action`` — ``train``,
``profile_step``, ``find_unused_params``, ``calculate_value``,
``get_advantage_and_percentiles``, ``compute_max_token_length`` — are out of scope rather than
allowlisted: with no sampler call there is no sampling to fan out, so there is nothing for the
field to be silently ignored *by*. That distinction is derived from the tree here, not
hand-maintained, which is what keeps the two lists below from failing open (CLAUDE.md rule 5).

An AST sweep rather than a behavioural test because the thing being checked is *coverage*:
constructing a multi-billion-parameter policy once per entry point would pin the same call
against nine near-identical stubs while still missing the only thing that differs between them
— whether the call is there at all.
"""

import ast
from pathlib import Path

import pytest

import opentau

_SCRIPTS_ROOT = Path(opentau.__file__).parent / "scripts"

#: Factory functions that turn ``cfg.policy`` into a policy (or its class).
_POLICY_BUILDERS = frozenset({"make_policy", "get_policy_class"})

#: Attributes whose presence means the script can reach the flow-matching sampler, and so can
#: be affected by ``n_candidates``. Both spellings count: ``select_action`` delegates to
#: ``sample_actions`` after the action queue drains.
_SAMPLER_ATTRS = frozenset({"sample_actions", "select_action"})

#: Serving entry points that arm best-of-N. Each must call ``configure_candidates`` *before*
#: its first sampler call, so critic loading, the dtype cast, the shape smoke-test and any
#: out-of-memory land at startup rather than on the first robot request.
ARMS_CANDIDATES = [
    "inference.py",
    "benchmark_inference.py",
    "grpc/server.py",
    "robocasa/server.py",
]

#: Entry points that reach the sampler but cannot honour best-of-N, mapped to the reason —
#: recorded here so the list reads as a set of decisions rather than a set of exclusions.
REFUSES_CANDIDATES = {
    "eval.py": "serving-only scope; the eval batch would multiply the candidate memory",
    "export_to_onnx.py": "the ONNX wrapper bypasses the policy layer the critic lives in",
    "actions_mse_loss.py": "compiles the policy-level sampler, trapping the critic in the graph",
    "high_level_planner_inference.py": "the planner families emit text, not an action chunk",
    "diagnose_accel.py": "pins the noise draw that best-of-N exists to fan out",
}


def _reaches_the_sampler_from_cfg_policy(source: str) -> bool:
    """Whether ``source`` builds a policy out of ``cfg.policy`` and can reach its sampler.

    Both halves are required. Building alone is the training path, which never samples;
    a sampler reference alone is ``onnx_inference`` / ``tensorrt_inference``, which run an
    exported artifact and have no policy object to arm.

    Sampler *references* count, not just calls: ``actions_mse_loss`` and every compiling
    server pass ``policy.sample_actions`` to ``torch.compile`` and invoke the result through a
    local name, so a call-only matcher would miss exactly the script whose refusal reason is
    that it compiles the policy-level sampler.
    """
    tree = ast.parse(source)
    builds = False
    reaches = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _POLICY_BUILDERS
        ):
            args = list(node.args) + [kw.value for kw in node.keywords]
            builds = builds or any(
                isinstance(sub, ast.Attribute) and sub.attr == "policy"
                for arg in args
                for sub in ast.walk(arg)
            )
        elif isinstance(node, ast.Attribute) and node.attr in _SAMPLER_ATTRS:
            reaches = True
    return builds and reaches


def _entry_points_reaching_the_sampler() -> set[str]:
    return {
        path.relative_to(_SCRIPTS_ROOT).as_posix()
        for path in _SCRIPTS_ROOT.rglob("*.py")
        if _reaches_the_sampler_from_cfg_policy(path.read_text())
    }


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every ``name(...)`` call, by bare function name."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _first_sampler_call_line(tree: ast.AST) -> int | None:
    """Line of the earliest ``<obj>.sample_actions(...)`` / ``.select_action(...)`` call."""
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SAMPLER_ATTRS
    ]
    return min(lines) if lines else None


def _tree(rel_path: str) -> ast.Module:
    return ast.parse((_SCRIPTS_ROOT / rel_path).read_text())


def test_every_entry_point_that_reaches_the_sampler_arms_or_refuses_candidates():
    """The registry itself: no script may silently ignore ``n_candidates``.

    This is the test that has to fail when somebody adds a tenth entry point. It derives the
    population from the tree and checks it against the two hand-kept lists, so a new server
    that copies an existing one — imports included — still lands here as an unhandled file
    rather than as a script that loads an ``n_candidates=8`` checkpoint and quietly serves one
    candidate.

    The reverse direction matters too: an entry listed but no longer detected is stale, and a
    stale allowlist entry is how a list stops describing the code it guards.
    """
    detected = _entry_points_reaching_the_sampler()
    handled = set(ARMS_CANDIDATES) | set(REFUSES_CANDIDATES)

    unhandled = detected - handled
    assert not unhandled, (
        "These entry points build a policy from `cfg.policy` and reach its sampler, but "
        "neither arm best-of-N (`configure_candidates`) nor refuse it (`refuse_candidates`), "
        "so a checkpoint whose config.json carries n_candidates>1 is served as a single "
        f"candidate with no error: {sorted(unhandled)}"
    )

    stale = handled - detected
    assert not stale, (
        "These are listed in ARMS_CANDIDATES/REFUSES_CANDIDATES but no longer build a policy "
        f"and reach its sampler; drop them so the lists keep describing the tree: {sorted(stale)}"
    )


def test_the_detector_flags_a_newly_added_entry_point():
    """Pin the property the registry test's failure depends on.

    A registry test whose scan silently stops matching passes forever while covering nothing,
    so the conflict is constructed here rather than trusted: a synthetic script in the shape a
    new server would take must come back detected — which is what makes it show up as
    ``unhandled`` above until someone adds it to a list.
    """
    new_entry_point = (
        "from opentau.policies.factory import get_policy_class\n"
        "def serve(cfg):\n"
        "    policy = get_policy_class(cfg.policy.type).from_pretrained(cfg.policy.pretrained_path)\n"
        "    return policy.sample_actions({})\n"
    )
    assert _reaches_the_sampler_from_cfg_policy(new_entry_point), (
        "the scan behind test_every_entry_point_that_reaches_the_sampler_arms_or_refuses_"
        "candidates no longer recognizes a new serving entry point, so that test would pass "
        "over one instead of failing on it"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "from opentau.policies.factory import make_policy\n"
            "def train(cfg):\n"
            "    policy = make_policy(cfg=cfg.policy)\n"
            "    return policy.forward({})\n",
            id="builds-but-never-samples",
        ),
        pytest.param(
            "import onnxruntime\n"
            "def run(session, batch):\n"
            "    return session.run(None, batch)  # exported sample_actions graph\n",
            id="samples-but-builds-no-policy",
        ),
    ],
)
def test_the_detector_ignores_scripts_that_are_out_of_scope(source):
    """The other half of the conflict: a detector that returns ``True`` for everything.

    It would make the registry test pass only because both lists happen to be long enough,
    and would then demand a pointless refusal line in every training script. Both halves of
    the ``builds and reaches`` conjunction are exercised, one per case.
    """
    assert not _reaches_the_sampler_from_cfg_policy(source)


@pytest.mark.parametrize("rel_path", ARMS_CANDIDATES)
def test_arm_listed_entry_points_actually_call_configure_candidates(rel_path):
    """A list entry is a claim about the source; check the source makes it true.

    Without this the allowlist is self-certifying — adding a filename would satisfy the
    registry test whether or not the call was ever written.
    """
    assert _calls(_tree(rel_path), "configure_candidates"), (
        f"{rel_path} is listed in ARMS_CANDIDATES but never calls configure_candidates()"
    )


@pytest.mark.parametrize("rel_path", sorted(REFUSES_CANDIDATES))
def test_refuse_listed_entry_points_actually_call_refuse_candidates(rel_path):
    """Same, for the refusal half."""
    assert _calls(_tree(rel_path), "refuse_candidates"), (
        f"{rel_path} is listed in REFUSES_CANDIDATES but never calls refuse_candidates()"
    )


def test_no_entry_point_both_arms_and_refuses():
    """The two lists are alternatives; membership in both is a contradiction, not a belt-and-braces."""
    both = set(ARMS_CANDIDATES) & set(REFUSES_CANDIDATES)
    assert not both, f"listed as both arming and refusing best-of-N: {sorted(both)}"


@pytest.mark.parametrize("rel_path", ARMS_CANDIDATES)
def test_candidates_are_armed_before_the_first_sampler_call(rel_path):
    """Arming after the warmup would defer every startup failure to the first robot request.

    ``configure_candidates`` is where the critic is loaded, moved to the device, cast and
    smoke-called for its output shape, and where the widened batch first allocates. Every one
    of those is a startup concern: a server that warms up at N=1 and only fans out on the
    first real request pays a recompile and can run out of memory with a robot waiting.
    """
    tree = _tree(rel_path)
    arm = _calls(tree, "configure_candidates")
    assert len(arm) == 1, f"{rel_path}: expected exactly one configure_candidates() call, found {len(arm)}"

    first_sample = _first_sampler_call_line(tree)
    assert first_sample is not None, f"{rel_path}: no sampler call found; the ordering matcher needs updating"
    assert arm[0].lineno < first_sample, (
        f"{rel_path}: configure_candidates() at line {arm[0].lineno} runs after the first "
        f"sample_actions()/select_action() call at line {first_sample}, so critic loading and "
        "the candidate fan-out land on a live request instead of at startup"
    )


@pytest.mark.parametrize("rel_path", ARMS_CANDIDATES)
def test_arming_passes_an_explicit_dtype(rel_path):
    """``attach_critic`` keeps the critic out of ``policy._modules`` on purpose.

    The cost of that is that no later ``policy.to(...)`` reaches it, so the cast has to be
    passed in here or never happen. Every one of these entry points runs its policy in
    bfloat16; a critic silently left in float32 behind it is a dtype mismatch that only
    surfaces on the first request, and only for a critic that has parameters (the built-in
    ``MedoidCritic`` has none, so this cannot be caught by exercising today's default).
    """
    call = _calls(_tree(rel_path), "configure_candidates")[0]
    passed = {kw.arg for kw in call.keywords}
    assert "device" in passed and "dtype" in passed, (
        f"{rel_path}: configure_candidates() must be given explicit `device=` and `dtype=`; "
        f"got {sorted(passed)}"
    )


@pytest.mark.parametrize("rel_path", sorted(REFUSES_CANDIDATES))
def test_refusals_explain_what_would_go_wrong(rel_path):
    """``reason`` is keyword-only and lands verbatim in the operator's traceback.

    A refusal that only says "unsupported" tells someone who set ``n_candidates=4`` nothing
    about where to set it instead, which is the whole difference between this and the silent
    ignore it replaces.
    """
    for call in _calls(_tree(rel_path), "refuse_candidates"):
        reason = next((kw.value for kw in call.keywords if kw.arg == "reason"), None)
        assert reason is not None, f"{rel_path}: refuse_candidates() must be passed a `reason=`"
        # Adjacent string literals are folded into one Constant by the parser, but a reason
        # that interpolates a config value (eval names its own batch size) is a JoinedStr —
        # measure the literal text in either shape rather than demanding the simpler one.
        if isinstance(reason, ast.Constant):
            literal = reason.value
        elif isinstance(reason, ast.JoinedStr):
            literal = "".join(p.value for p in reason.values if isinstance(p, ast.Constant))
        else:
            raise AssertionError(f"{rel_path}: refuse_candidates(reason=...) must be a string literal")
        assert len(literal) > 40, (
            f"{rel_path}: refuse_candidates(reason=...) must be a sentence explaining what "
            "would go wrong, not a placeholder"
        )
