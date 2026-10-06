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

"""Inference entry points hand the policy a float32 ``state``; the model still sees training's.

Two contracts, pinned end to end through the real serving code:

* **Delta-action targets.** The dataset forms delta actions against the float32 state and only
  then casts the sample to bfloat16, so the inverse at inference must add a float32 state back.
  The servers used to build ``state`` in the bfloat16 serving dtype, which put up to 2**-7 rad
  of rounding on a joint in [2, 4) rad into every absolute target of every chunk.
* **Model-input parity.** Training normalizes a bfloat16 state in bfloat16 arithmetic. A float32
  state must not change that — ``Normalize`` computes in its stats' dtype — or a discretized
  state token moves by a bin (a quarter of states across [-4, 4] rad for the stats here).

Everything runs on CPU against a real ``PI05Policy``; only its Hub-loaded pieces (PaliGemma
tokenizer, FAST processor, the flow-matching backbone) are stubbed.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

import opentau
from opentau.configs.deployment import PlannerConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.datasets.action_indexing import subtract_chunk_start_state
from opentau.policies.layers import PerGroupLinear
from opentau.policies.normalize import Normalize
from opentau.policies.pi05 import modeling_pi05
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.utils import cast_to_weight_dtype, to_dtype_preserving_siglip_float32
from opentau.scripts.grpc import robot_inference_pb2
from opentau.scripts.grpc.server import RobotPolicyServicer
from opentau.scripts.robocasa import server as robocasa_server
from opentau.utils.utils import INFERENCE_STATE_DTYPE, create_dummy_observation

STATE_DIM = 8  # 7 arm joints + gripper; already max_state_dim, so the servers' pad is a no-op
ACTION_DIM = 8
CHUNK = 4
DELTA_MAP = {joint: joint for joint in range(7)}  # the gripper (dim 7) stays absolute
CPU = torch.device("cpu")

# Off bfloat16's grid on purpose: in [2, 4) rad bfloat16 can miss by up to 2**-7 ~ 0.0078 rad,
# and 2.0071 / -2.5551 / 3.9001 each lose more than 0.006 rad.
JOINTS = [2.0071, -3.1234, 2.9999, 1.2345, -2.5551, 0.3333, 3.9001, 0.85]
# Not powers of two (dividing by one is exact in any binary format), and wide enough that every
# joint normalizes inside [-1, 1] for the discrete-state path.
STATE_MEAN, STATE_STD = 0.1, 4.2


class _RecordingTokenizer:
    """Stands in for the PaliGemma tokenizer; keeps every prompt it was asked to encode."""

    def __init__(self):
        self.prompts: list[list[str]] = []

    def __call__(self, prompts, *, max_length, **_):
        self.prompts.append(list(prompts))
        ones = torch.ones(len(prompts), max_length, dtype=torch.long)
        return {"input_ids": ones, "attention_mask": ones}


class _StubFlowMatching(nn.Module):
    """Stands in for PaliGemma + the action expert.

    Emits a fixed float32 chunk of normalized deltas — the real sampler integrates float32 noise,
    so its output is float32 as well — and records what the policy fed it.
    """

    def __init__(self, chunk: torch.Tensor):
        super().__init__()
        # A plain attribute, not a buffer, so the serving-time bfloat16 cast leaves it alone.
        self.chunk = chunk
        self.seen: dict = {}

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_prefix,
        delay,
        noise=None,
        state=None,
        n_candidates=1,
        accel=None,
    ):
        self.seen = {"state": state, "action_prefix": action_prefix}
        return self.chunk.expand(lang_tokens.shape[0], -1, -1).clone()


def _delta_chunk() -> torch.Tensor:
    """``(1, CHUNK, ACTION_DIM)`` small displacements, plus an absolute gripper command."""
    steps = torch.arange(1, CHUNK + 1, dtype=torch.float32)
    chunk = torch.outer(steps, torch.linspace(-0.03, 0.03, ACTION_DIM))
    chunk[:, 7] = 0.8
    return chunk.unsqueeze(0)


def _served_policy(monkeypatch, *, state_type: str, action_mode=NormalizationMode.IDENTITY, max_delay=0):
    """A real ``PI05Policy`` cast exactly the way every serving entry point casts it."""
    tokenizer = _RecordingTokenizer()
    stub = _StubFlowMatching(_delta_chunk())
    monkeypatch.setattr(
        modeling_pi05, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    )
    monkeypatch.setattr(
        modeling_pi05,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(vocab_size=8)),
    )
    monkeypatch.setattr(PI05Policy, "_check_discrete_action_tokenizer_convention", lambda self, path: None)
    monkeypatch.setattr(PI05Policy, "_build_flow_matching", lambda self, config, vocab_size: stub)

    config = PI05Config(
        n_obs_steps=1,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        max_delay=max_delay,
        max_state_dim=STATE_DIM,
        max_action_dim=ACTION_DIM,
        state_type=state_type,
        resize_imgs_with_padding=None,
        prompt_max_length=8,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": action_mode,
        },
        delta_action_state_map=dict(DELTA_MAP),
    )
    config.input_features = {
        "state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    stats = {
        "state": {
            "mean": torch.full((STATE_DIM,), STATE_MEAN),
            "std": torch.full((STATE_DIM,), STATE_STD),
        },
        "actions": {
            "mean": torch.full((ACTION_DIM,), 0.01),
            "std": torch.full((ACTION_DIM,), 0.037),
            "min": torch.full((ACTION_DIM,), -0.2),
            "max": torch.full((ACTION_DIM,), 0.2),
        },
    }
    policy = PI05Policy(config, per_dataset_stats=[stats])
    to_dtype_preserving_siglip_float32(policy, dtype=torch.bfloat16)
    policy.eval()
    return policy, stub, tokenizer


@pytest.fixture
def make_servicer():
    """A gRPC servicer serving a given policy on CPU, with policy loading patched out."""
    servicers = []

    def _make(policy) -> RobotPolicyServicer:
        cfg = SimpleNamespace(
            planner=PlannerConfig(enabled=False),
            policy=SimpleNamespace(type="pi05"),
            server=SimpleNamespace(robot_type=None, control_mode=None, dataset_repo_id=None),
            num_cams=1,
            resolution=(8, 8),
            max_state_dim=STATE_DIM,
            max_action_dim=ACTION_DIM,
            action_chunk=CHUNK,
        )
        with patch.object(RobotPolicyServicer, "_load_policy"):
            servicer = RobotPolicyServicer(cfg)
        servicer.policy = policy
        servicer.device = CPU  # `auto_torch_device` may pick CUDA / MPS; the policy lives on CPU
        servicers.append(servicer)
        return servicer

    yield _make
    for servicer in servicers:
        servicer.close()


def _request(state, **fields) -> robot_inference_pb2.ObservationRequest:
    request = robot_inference_pb2.ObservationRequest(prompt="pick", **fields)
    request.robot_state.state.extend(state)
    return request


def _serve(servicer, request) -> torch.Tensor:
    """Run one ``GetActionChunk`` and return the chunk exactly as it went over the wire."""
    context = MagicMock()
    response = servicer.GetActionChunk(request, context)
    context.abort.assert_not_called()
    return torch.tensor([list(row.values) for row in response.action_chunk], dtype=torch.float32)


def _training_normalized_state(policy, state: torch.Tensor) -> torch.Tensor:
    """What training feeds the model: the dataset's bfloat16 state through ``Normalize``."""
    batch = {"state": state.to(torch.bfloat16).unsqueeze(0)}
    return policy.normalize_inputs(batch, torch.zeros(1, dtype=torch.long))["state"]


def _float32_math_normalized_state(policy, state: torch.Tensor) -> torch.Tensor:
    """What ``Normalize`` used to compute for a float32 state: float32 math on bfloat16 stats."""
    buffer = policy.normalize_inputs.buffer_state
    eps = policy.normalize_inputs.eps
    std = buffer["std"][0]
    std = torch.where(std.abs() < eps, torch.ones_like(std), std)
    return ((state - buffer["mean"][0]) / (std + eps)).unsqueeze(0)


def _bins(normalized: torch.Tensor) -> list[int]:
    """``prepare_discrete_state``'s binning."""
    return ((normalized.float().clamp(-1.0, 1.0) + 1.0) * 128.0).long().clamp(0, 255)[0].tolist()


# --------------------------------------------------------------------------------------
# Delta-action targets: the inverse must add the unrounded state back.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("state_type", ["continuous", "discrete"])
def test_grpc_serves_delta_targets_against_the_float32_state(monkeypatch, make_servicer, state_type):
    """Absolute joint targets must equal float32 ``state + delta``, bit for bit."""
    policy, stub, _ = _served_policy(monkeypatch, state_type=state_type)
    state = torch.tensor(JOINTS)  # the robot's wire format is float32 as well

    served = _serve(make_servicer(policy), _request(JOINTS))

    delta = stub.chunk[0]  # IDENTITY action normalization: the stub's output is the delta itself
    expected = delta.clone()
    expected[:, :7] += state[:7]
    stale = delta.clone()
    stale[:, :7] += state[:7].to(torch.bfloat16).float()
    assert (stale - expected).abs().max() > 5e-3, "precondition: bf16 rounding must be visible here"
    assert torch.equal(served, expected)


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_robocasa_serves_delta_targets_against_the_float32_state(monkeypatch, batched):
    """Same contract through the RoboCasa websocket runner, single and batched requests."""
    policy, stub, _ = _served_policy(monkeypatch, state_type="continuous")
    runner = object.__new__(robocasa_server.OpenTauRoboCasaPolicy)  # skip checkpoint loading
    runner.cfg = SimpleNamespace(num_cams=1, resolution=(8, 8), max_state_dim=STATE_DIM)
    runner.device, runner.dtype, runner.policy = CPU, torch.bfloat16, policy
    runner.accel_prefix, runner.last_accel = None, None
    images = {key: np.zeros((8, 8, 3), dtype=np.uint8) for key in robocasa_server.ROBOCASA_CAMERA_ORDER}
    states = [np.asarray(JOINTS, dtype=np.float64), -np.asarray(JOINTS, dtype=np.float64)]  # msgpack is f64

    if batched:
        served = runner.infer_batch([(images, s, "pick") for s in states], ACTION_DIM)
    else:
        states = states[:1]
        served = [runner.infer(images, states[0], "pick", ACTION_DIM)]

    for out, state64 in zip(served, states, strict=True):
        state = torch.tensor(state64, dtype=torch.float32)
        expected = stub.chunk[0].clone()
        expected[:, :7] += state[:7]
        np.testing.assert_array_equal(out, expected.numpy().astype(np.float64))


def test_grpc_rtc_prefix_is_converted_to_deltas_as_training_did(monkeypatch, make_servicer):
    """A served RTC prefix (absolute targets) must reach the model as training's delta prefix.

    Training: ``a - s`` in float32, cast to bfloat16 by the dataset, normalized in bfloat16. The
    server used to round both ``a`` and ``s`` to bfloat16 and subtract there.
    """
    policy, stub, _ = _served_policy(
        monkeypatch, state_type="continuous", action_mode=NormalizationMode.MEAN_STD, max_delay=2
    )
    state = torch.tensor(JOINTS)
    prefix = torch.zeros(CHUNK, ACTION_DIM)
    prefix[:2] = state + torch.tensor([[0.011], [0.023]])  # two committed rows near the current pose
    request = _request(JOINTS, delay=2)
    for row in prefix[:2]:
        request.prefix_action.add(values=row.tolist())

    _serve(make_servicer(policy), request)

    index = torch.zeros(1, dtype=torch.long)
    trained = policy.normalize_targets(
        {"actions": subtract_chunk_start_state(prefix, state, DELTA_MAP).to(torch.bfloat16).unsqueeze(0)},
        index,
    )["actions"]
    stale_deltas = subtract_chunk_start_state(prefix.to(torch.bfloat16), state.to(torch.bfloat16), DELTA_MAP)
    stale = policy.normalize_targets({"actions": stale_deltas.unsqueeze(0)}, index)["actions"]
    assert not torch.equal(stale, trained), "precondition: the bf16 path must differ here"
    assert torch.equal(stub.seen["action_prefix"], trained)


# --------------------------------------------------------------------------------------
# Model-input parity: a float32 state must reach the model exactly as training's bf16 one.
# --------------------------------------------------------------------------------------


def test_served_continuous_state_matches_the_training_input(monkeypatch, make_servicer):
    policy, stub, _ = _served_policy(monkeypatch, state_type="continuous")
    state = torch.tensor(JOINTS)
    trained = _training_normalized_state(policy, state)
    float32_math = _float32_math_normalized_state(policy, state).to(torch.bfloat16)
    assert not torch.equal(float32_math, trained), "precondition: float32 math must differ here"

    _serve(make_servicer(policy), _request(JOINTS))

    assert stub.seen["state"].dtype == torch.bfloat16
    assert torch.equal(stub.seen["state"], trained)


def test_served_discrete_state_tokens_match_the_training_tokens(monkeypatch, make_servicer):
    policy, _, tokenizer = _served_policy(monkeypatch, state_type="discrete")
    state = torch.tensor(JOINTS)
    trained_bins = _bins(_training_normalized_state(policy, state))
    assert _bins(_float32_math_normalized_state(policy, state)) != trained_bins, (
        "precondition: float32 math must move at least one state token here"
    )

    _serve(make_servicer(policy), _request(JOINTS))

    assert tokenizer.prompts[-1] == [f"Task: pick, State: {' '.join(map(str, trained_bins))};\n"]


@pytest.mark.parametrize(
    "mode, stats",
    [
        (NormalizationMode.MEAN_STD, {"mean": STATE_MEAN, "std": STATE_STD}),
        (NormalizationMode.MIN_MAX, {"min": -3.3, "max": 4.1}),
        (NormalizationMode.QUANTILE, {"q01": -3.3, "q99": 4.1}),
    ],
)
def test_normalize_computes_in_its_stats_dtype(mode, stats):
    """bfloat16 stats: a float32 input normalizes exactly like its bfloat16 rounding (training's
    input). float32 stats: a bfloat16 input is upcast, as plain type promotion would."""
    features = {"state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    per_dataset = [{"state": {name: torch.full((STATE_DIM,), value) for name, value in stats.items()}}]
    norm32 = Normalize(features, {"STATE": mode}, per_dataset_stats=per_dataset)
    norm16 = copy.deepcopy(norm32).to(torch.bfloat16)
    index = torch.zeros(1, dtype=torch.long)
    x = torch.tensor([JOINTS])

    served = norm16({"state": x}, index)["state"]
    trained = norm16({"state": x.to(torch.bfloat16)}, index)["state"]
    float32_math = copy.deepcopy(norm16).float()({"state": x}, index)["state"].to(torch.bfloat16)
    assert not torch.equal(float32_math, trained), "precondition: float32 math must differ here"
    assert served.dtype == torch.bfloat16
    assert torch.equal(served, trained)

    upcast = norm32({"state": x.to(torch.bfloat16)}, index)["state"]
    assert upcast.dtype == torch.float32
    assert torch.equal(upcast, norm32({"state": x.to(torch.bfloat16).float()}, index)["state"])


# --------------------------------------------------------------------------------------
# State projections read their input dtype off the layer, not off the caller.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer, fallback, expected",
    [
        (nn.Linear(4, 2).to(torch.bfloat16), None, torch.bfloat16),
        (PerGroupLinear(4, 2, num_groups=2).to(torch.bfloat16), None, torch.bfloat16),
        (nn.Linear(4, 2), torch.bfloat16, torch.float32),  # the weight wins over the fallback
        (nn.Identity(), torch.bfloat16, torch.bfloat16),  # weightless test stub: the fallback
        (lambda x: x, None, torch.float32),  # no weight, no fallback: unchanged
    ],
    ids=["linear", "per-group-linear", "weight-beats-fallback", "identity-stub", "lambda-stub"],
)
def test_cast_to_weight_dtype(layer, fallback, expected):
    assert cast_to_weight_dtype(torch.ones(1, 4), layer, fallback).dtype == expected


def test_pi0_state_projection_accepts_a_float32_state():
    """pi0 fed ``state`` straight into ``state_proj``, so a float32 state crashed a bf16 policy."""
    from opentau.policies.pi0.configuration_pi0 import PI0Config
    from opentau.policies.pi0.modeling_pi0 import PI0FlowMatching

    fm = object.__new__(PI0FlowMatching)
    nn.Module.__init__(fm)
    fm.config = PI0Config(chunk_size=CHUNK, n_action_steps=CHUNK, max_state_dim=STATE_DIM, proj_width=16)
    fm.state_proj = nn.Linear(STATE_DIM, 16)
    fm.action_in_proj = nn.Linear(ACTION_DIM, 16)
    fm.action_time_mlp_in = nn.Linear(32, 16)
    fm.action_time_mlp_out = nn.Linear(16, 16)
    fm.to(torch.bfloat16)
    state = torch.tensor([JOINTS])

    embs, _, _ = PI0FlowMatching.embed_suffix(
        fm, state, torch.zeros(1, CHUNK, ACTION_DIM), torch.full((1,), 0.5)
    )

    assert torch.equal(embs[:, 0], fm.state_proj(state.to(torch.bfloat16)))


# --------------------------------------------------------------------------------------
# Entry points: every observation builder emits `state` at INFERENCE_STATE_DTYPE.
# --------------------------------------------------------------------------------------


def test_grpc_prepare_observation_keeps_state_and_prefix_unrounded(make_servicer):
    servicer = make_servicer(policy=None)  # building the batch never touches the policy
    with_prefix = _request(JOINTS, delay=1)
    with_prefix.prefix_action.add(values=JOINTS)

    for request in (_request(JOINTS), with_prefix):
        batch, prefix, _ = servicer._prepare_observation(request)
        assert batch["state"].dtype == INFERENCE_STATE_DTYPE
        assert torch.equal(batch["state"][0], torch.tensor(JOINTS))
        # Absolute targets, re-anchored on the state for a delta policy; `_load_policy`'s warmup
        # must trace this same dtype or the first real request can recompile.
        assert prefix.dtype == INFERENCE_STATE_DTYPE
    assert torch.equal(prefix[0, 0], torch.tensor(JOINTS))


def test_create_dummy_observation_builds_the_state_at_inference_dtype():
    cfg = SimpleNamespace(num_cams=1, resolution=(8, 8), max_state_dim=STATE_DIM, action_chunk=CHUNK)
    obs = create_dummy_observation(cfg, CPU, dtype=torch.bfloat16)
    assert obs["camera0"].dtype == torch.bfloat16
    assert obs["state"].dtype == INFERENCE_STATE_DTYPE


#: Modules that build inference batches by hand, swept for any `state` built at another dtype.
_BATCH_BUILDERS = (
    *sorted((Path(opentau.__file__).parent / "scripts").rglob("*.py")),
    Path(opentau.__file__).parent / "envs" / "utils.py",
    Path(opentau.__file__).parent / "utils" / "utils.py",
)


def _is_inference_state_dtype(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "INFERENCE_STATE_DTYPE"
    return isinstance(node, ast.Attribute) and node.attr in {"INFERENCE_STATE_DTYPE", "float32"}


def _state_built_at_another_dtype(path: Path) -> list[int]:
    """Lines building ``state`` (a ``"state"`` dict entry or ``x["state"] = ...``) with a
    ``dtype=`` that is neither ``INFERENCE_STATE_DTYPE`` nor ``torch.float32``."""
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            values = [
                v for k, v in zip(node.keys, node.values, strict=True) if getattr(k, "value", None) == "state"
            ]
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and getattr(t.slice, "value", None) == "state" for t in node.targets
        ):
            values = [node.value]
        else:
            continue
        for value in values:
            for call in (n for n in ast.walk(value) if isinstance(n, ast.Call)):
                hits += [
                    call.lineno
                    for kw in call.keywords
                    if kw.arg == "dtype" and not _is_inference_state_dtype(kw.value)
                ]
    return hits


@pytest.mark.parametrize(
    "path", _BATCH_BUILDERS, ids=lambda p: str(p.relative_to(Path(opentau.__file__).parent))
)
def test_no_entry_point_builds_state_at_another_dtype(path):
    """Copying the old ``dtype=self.dtype`` pattern into a new entry point must fail the suite."""
    assert _state_built_at_another_dtype(path) == [], (
        f"{path.name} builds `state` at a dtype other than INFERENCE_STATE_DTYPE; a delta-action "
        "policy would add a rounded state onto its predicted deltas. See INFERENCE_STATE_DTYPE."
    )
