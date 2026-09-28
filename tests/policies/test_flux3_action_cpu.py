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

"""CPU tests for the flux3_action policy.

No weights are downloaded: the DiT is built from a deliberately tiny ``dit_config`` and
the two frozen encoders are injected (``MockTextEncoder`` and a stub VAE), the pattern
``test_cosmos3_cpu.py`` / ``test_xr1_cpu.py`` use.

Most of these are **silent-divergence pins** -- each targets a place where a plausible,
reviewable change is wrong and *nothing crashes*:

* the video content streams quietly dropped from the trunk ("we only want actions"),
  which upstream forbids and which degrades action quality rather than raising;
* OpenTau normalization switched on over a policy that already range-normalizes itself,
  silently double-normalizing every action target;
* the batch-key aliasing clobbering a key upstream also reads, or deep-copying tensors;
* the heads' 5x LR group flattened away by returning bare parameters;
* an upstream re-sync reintroducing absolute ``flux_action`` imports, which resolve in a
  developer's tree only if the upstream repo happens to be importable there.
"""

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import opentau
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.factory import get_policy_class, make_policy_config
from opentau.policies.flux3_action.config import REQUIRED_CONTENT_STREAMS
from opentau.policies.flux3_action.configuration_flux3_action import Flux3ActionConfig
from opentau.policies.flux3_action.modeling_flux3_action import Flux3ActionPolicy
from opentau.policies.flux3_action.models.text_encoder import MockTextEncoder

VENDORED_ROOT = Path(opentau.__file__).parent / "policies" / "flux3_action"

TINY_DIT = {
    "hidden_size": 256,
    "num_heads": 4,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [16, 16, 16, 16],
    "context_in_dim": 512,
    "vec_in_dim": 64,
}


class _StubVAE(nn.Module):
    """Stand-in for the frozen video VAE; never called by these tests."""


def _config(**overrides) -> Flux3ActionConfig:
    kwargs = {
        "camera_keys": (
            "observation.images.wrist",
            "observation.images.left",
            "observation.images.right",
        ),
        "video_vae_id": "stub",
        "text_encoder_id": "stub",
        "dit_config": dict(TINY_DIT),
        "torch_dtype": "float32",
    }
    kwargs.update(overrides)
    cfg = Flux3ActionConfig(**kwargs)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return cfg


def _policy(cfg: Flux3ActionConfig | None = None) -> Flux3ActionPolicy:
    return Flux3ActionPolicy(cfg or _config(), video_vae=_StubVAE(), text_encoder=MockTextEncoder())


# --------------------------------------------------------------------------- registration
def test_policy_is_registered_everywhere():
    """All six registration touchpoints agree; five are easy to update and one is not."""
    assert "flux3_action" in opentau.available_policies
    assert get_policy_class("flux3_action") is Flux3ActionPolicy
    cfg = make_policy_config("flux3_action")
    assert isinstance(cfg, Flux3ActionConfig)
    assert cfg.type == "flux3_action"
    assert Flux3ActionPolicy.name == "flux3_action"
    assert Flux3ActionPolicy.config_class is Flux3ActionConfig


# ------------------------------------------------------- the structural invariant of F3A
def test_trunk_keeps_the_video_streams_alongside_the_action_streams():
    """F3A denoises action *and* video tokens in one sequence; video is not optional.

    Upstream marks both video streams required, and the released DROID settings guide
    video tokens at 4.0 against 1.0 on action tokens -- so a trunk built without them
    still runs and still returns actions, just worse ones. Pin the stream set so that
    silent degradation cannot land as a "we only predict actions" simplification.
    """
    assert REQUIRED_CONTENT_STREAMS == ("video", "video_cond")
    policy = _policy()
    assert set(policy.model.dit.in_channels) == {"action", "action_cond", "video", "video_cond"}


def test_fp8r_is_rejected_with_a_reason_not_a_latent_import_error():
    """The fp8r transformer is deliberately not vendored; its import site is lazy.

    Without an explicit check the failure would surface as an ImportError from inside
    ``from_pretrained``, long after the config that caused it was accepted.
    """
    with pytest.raises(ValueError, match="fp8r"):
        _config(quantization="fp8r")


# --------------------------------------------------------------------------- config rules
def test_cross_field_validation_is_delegated_to_upstream():
    """``to_policy_config`` is what enforces upstream's own constraints.

    Restating them in the OpenTau config would let the two drift; this pins that the
    delegation actually happens at construction time rather than at the first forward.
    """
    # history/profile pairing: n_obs_steps > 1 requires inference_profile="history"
    with pytest.raises(ValueError):
        _config(n_obs_steps=4)
    # a valid pairing still builds
    cfg = _config(n_obs_steps=4, inference_profile="history", history_snapshots=2, gripper_flip_dims=())
    assert cfg.to_policy_config().n_obs_steps == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_action_steps": 64, "chunk_size": 32},  # execution horizon beyond the trained chunk
        {"action_dim": 64, "max_action_dim": 32},  # real width beyond the padded width
    ],
)
def test_horizon_and_width_bounds_are_enforced(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_action_delta_indices_span_the_chunk_and_no_future_frames_are_requested():
    """Actions get a forward horizon; observations deliberately get none.

    Finetuning F3A needs future camera frames, which the dataset layer cannot request
    today. Eval does not -- video tokens are denoised from noise conditioned on the
    observed frame. Pin that this policy asks only for what inference needs, so the
    training-side change is a deliberate follow-up rather than an accident.
    """
    cfg = _config()
    assert cfg.action_delta_indices == list(range(cfg.chunk_size))
    assert cfg.observation_delta_indices is None


def test_opentau_normalization_is_identity_on_every_stream():
    """F3A range-normalizes state and actions itself from its own q01/q99 bounds.

    Letting OpenTau normalize as well would double-normalize the action targets --
    no crash, just a policy trained against the wrong scale.
    """
    cfg = _config()
    assert set(cfg.normalization_mapping.values()) == {NormalizationMode.IDENTITY}


# --------------------------------------------------------------------------- the adapter
def test_batch_keys_are_aliased_without_copying_or_clobbering():
    policy = _policy()
    state, actions = torch.zeros(2, 8), torch.zeros(2, 32, 8)
    out = policy._upstream_batch({"observation.state": state, "actions": actions, "task": "pick"})
    # upstream's names now resolve...
    assert out["state"] is state, "aliasing must share the tensor, not copy it"
    assert out["action"] is actions
    # ...without disturbing OpenTau's, or anything upstream reads opportunistically
    assert out["observation.state"] is state
    assert out["task"] == "pick"


def test_aliasing_does_not_overwrite_an_explicit_upstream_key():
    """``setdefault`` semantics: a caller that already speaks upstream's dialect wins."""
    policy = _policy()
    native, opentau_side = torch.ones(2, 8), torch.zeros(2, 8)
    out = policy._upstream_batch({"observation.state": opentau_side, "state": native})
    assert out["state"] is native


def test_optim_params_keep_the_heads_learning_rate_group():
    """Upstream runs the embodiment heads at 5x the trunk LR -- part of the recipe.

    Returning bare ``parameters()`` would train perfectly well and converge worse.
    """
    policy = _policy()
    groups = policy.get_optim_params()
    assert isinstance(groups, list) and len(groups) > 1
    assert all(isinstance(g, dict) and "params" in g for g in groups)
    scales = {g.get("lr_scale", g.get("lr")) for g in groups}
    assert len(scales) > 1, f"expected distinct trunk/head learning rates, got {groups!r}"


def test_reset_is_safe_before_the_inner_model_exists():
    """``PreTrainedPolicy.__init__`` calls ``reset()`` before ``self.model`` is assigned."""
    policy = _policy()
    policy.reset()  # must not raise after construction either


# ------------------------------------------------------------------- the vendoring contract
def test_no_absolute_upstream_imports_survive_in_vendored_code():
    """Vendored modules must import each other relatively.

    Upstream imports itself absolutely in three places. Those resolve only if the
    ``flux_action`` package also happens to be installed, so a re-sync that reintroduces
    one would pass on the syncer's machine and fail in CI.
    """
    offenders = []
    for path in sorted(VENDORED_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flux_action"):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}:{node.lineno}" for a in node.names if a.name.startswith("flux_action")
                ]
    assert not offenders, f"absolute upstream imports must be rewritten relative: {offenders}"


def test_every_upstream_policy_config_field_is_forwarded_explicitly():
    """``to_policy_config`` must pass every field upstream declares.

    A field introduced by an upstream re-sync and not forwarded would silently fall back
    to upstream's default while the OpenTau config still appears to expose it. Parsing
    the call's keywords (rather than trusting the round-trip) is what makes the
    *omission* detectable, since an omitted field still reads back fine as its default.
    """
    from opentau.policies.flux3_action.config import PolicyConfig

    src = (VENDORED_ROOT / "configuration_flux3_action.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "to_policy_config")
    call = next(
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "PolicyConfig"
    )
    forwarded = {kw.arg for kw in call.keywords if kw.arg is not None}

    # ``conditioning_channels`` is a derived property on the upstream dataclass, not an
    # init field, so it is not forwardable; everything else must be.
    declared = {f.name for f in fields(PolicyConfig) if f.init}
    missing = declared - forwarded
    assert not missing, f"PolicyConfig fields not forwarded by to_policy_config: {sorted(missing)}"

    unknown = forwarded - declared
    assert not unknown, f"to_policy_config forwards fields upstream does not declare: {sorted(unknown)}"

    built = _config().to_policy_config()
    assert built.chunk_size == 32 and built.n_action_steps == 32
    assert built.camera_layout == "droid"
