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

"""config_version gate for the zero-range normalization convention.

Covers the behavioral fix (a zero-range MIN_MAX/QUANTILE band maps to 0.0 under
config_version >= 1 instead of the legacy -1.0), the config-level version
plumbing (default, resolution, save concretization, provenance peek), and the
FAST tokenizer convention guard. All CPU-only, no network, no model backbone.
"""

import json
import types
from pathlib import Path

import pytest
import torch

from opentau.configs.policies import CURRENT_CONFIG_VERSION, PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.factory import make_policy_config
from opentau.policies.normalize import (
    ACTION_NORM_META_FILE,
    LEGACY_EPS,
    OPENPI_EPS,
    Normalize,
    Unnormalize,
)
from opentau.policies.pretrained import (
    PreTrainedPolicy,
    _extract_config_version,
    _peek_config_version,
    _read_action_norm_meta,
)

REAL, MAXD = 6, 16  # real width vs padded max dim


def _padded_stats(lo_name, hi_name):
    """One dataset: real dims [-1, 1], padded tail all-zero (lo == hi == 0)."""
    lo = torch.zeros(MAXD)
    hi = torch.zeros(MAXD)
    lo[:REAL] = -1.0
    hi[:REAL] = 1.0
    return [{"action": {lo_name: lo, hi_name: hi}}]


def _feat():
    return {"action": PolicyFeature(type=FeatureType.ACTION, shape=(MAXD,))}


# --------------------------------------------------------------------------- #
# config method contract                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "config_version, expected_center, expected_resolved",
    [(None, True, CURRENT_CONFIG_VERSION), (0, False, 0), (1, True, 1), (2, True, 2)],
)
def test_config_version_method_contract(config_version, expected_center, expected_resolved):
    cfg = make_policy_config("pi0")
    cfg.config_version = config_version
    assert cfg.zero_range_centers_on_zero() is expected_center
    assert cfg.resolved_config_version() == expected_resolved


@pytest.mark.parametrize(
    "config_version, expected_eps",
    [(None, OPENPI_EPS), (0, LEGACY_EPS), (1, OPENPI_EPS), (2, OPENPI_EPS)],
)
def test_normalization_epsilon_is_version_gated(config_version, expected_eps):
    """The epsilon is bundled with zero-range-centering into the v1 openpi-parity behavior:
    legacy (v0) keeps 1e-8, v1+ uses openpi's 1e-6, and an unresolved config runs current."""
    cfg = make_policy_config("pi0")
    cfg.config_version = config_version
    assert cfg.normalization_epsilon() == expected_eps
    # Epsilon and zero-range-centering flip together — both or nothing.
    assert (cfg.normalization_epsilon() == OPENPI_EPS) == cfg.zero_range_centers_on_zero()


def test_config_version_defaults_to_none_on_fresh_config():
    """A freshly built config carries the unresolved sentinel, which behaves as
    CURRENT for normalization (so a fresh run gets the new convention)."""
    cfg = make_policy_config("pi0")
    assert cfg.config_version is None
    assert cfg.zero_range_centers_on_zero() is (CURRENT_CONFIG_VERSION >= 1)


# --------------------------------------------------------------------------- #
# numeric behavior: MIN_MAX / QUANTILE flip, MEAN_STD unaffected              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mode, lo_name, hi_name",
    [(NormalizationMode.MIN_MAX, "min", "max"), (NormalizationMode.QUANTILE, "q01", "q99")],
)
def test_padded_dim_flips_with_convention(mode, lo_name, hi_name):
    stats = _padded_stats(lo_name, hi_name)
    x = torch.zeros(1, MAXD)
    x[0, :REAL] = 0.5
    idx = torch.tensor([0])

    legacy = Normalize(_feat(), {"ACTION": mode}, per_dataset_stats=stats, zero_range_center=False)
    new = Normalize(_feat(), {"ACTION": mode}, per_dataset_stats=stats, zero_range_center=True)
    z_legacy = legacy({"action": x.clone()}, idx)["action"][0]
    z_new = new({"action": x.clone()}, idx)["action"][0]

    # Padded tail: legacy -> -1.0, config_version >= 1 -> 0.0 (openpi output).
    assert torch.allclose(z_legacy[REAL:], torch.full((MAXD - REAL,), -1.0), atol=1e-6)
    assert torch.allclose(z_new[REAL:], torch.zeros(MAXD - REAL), atol=1e-6)
    # Healthy real dims are bit-identical across conventions.
    assert torch.equal(z_legacy[:REAL], z_new[:REAL])


def test_mean_std_is_unaffected_by_convention():
    """MEAN_STD already emits 0.0 on a zero-std dim (no -1 re-centering), so the
    flag must be a no-op for it."""
    mean = torch.zeros(MAXD)
    std = torch.zeros(MAXD)
    std[:REAL] = 1.0
    stats = [{"action": {"mean": mean, "std": std}}]
    x = torch.zeros(1, MAXD)
    x[0, :REAL] = 0.5
    idx = torch.tensor([0])
    a = Normalize(
        _feat(), {"ACTION": NormalizationMode.MEAN_STD}, per_dataset_stats=stats, zero_range_center=False
    )
    b = Normalize(
        _feat(), {"ACTION": NormalizationMode.MEAN_STD}, per_dataset_stats=stats, zero_range_center=True
    )
    za = a({"action": x.clone()}, idx)["action"]
    zb = b({"action": x.clone()}, idx)["action"]
    assert torch.equal(za, zb)
    assert torch.allclose(za[0, REAL:], torch.zeros(MAXD - REAL), atol=1e-6)


@pytest.mark.parametrize(
    "mode, lo_name, hi_name",
    [(NormalizationMode.MIN_MAX, "min", "max"), (NormalizationMode.QUANTILE, "q01", "q99")],
)
@pytest.mark.parametrize("zero_range_center", [False, True])
def test_unnormalize_round_trips_under_both_conventions(mode, lo_name, hi_name, zero_range_center):
    stats = _padded_stats(lo_name, hi_name)
    x = torch.zeros(1, MAXD)
    x[0, :REAL] = torch.tensor([0.5, -0.3, 0.9, -1.0, 0.1, 0.0])
    idx = torch.tensor([0])
    norm = Normalize(_feat(), {"ACTION": mode}, per_dataset_stats=stats, zero_range_center=zero_range_center)
    unnorm = Unnormalize(
        _feat(), {"ACTION": mode}, per_dataset_stats=stats, zero_range_center=zero_range_center
    )
    z = norm({"action": x.clone()}, idx)["action"]
    rec = unnorm({"action": z}, idx)["action"]
    torch.testing.assert_close(rec, x, atol=1e-5, rtol=1e-4)


def test_constant_real_dim_deviation_is_preserved_not_flattened():
    """A genuinely-constant real dim with a deviating value maps to 2*deviation
    (bounded, invertible) under config_version >= 1 — NOT flattened to 0."""
    lo = torch.tensor([[-1.0, 5.0]])  # dim 1 constant at 5
    hi = torch.tensor([[1.0, 5.0]])
    feat = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    stats = [{"action": {"min": lo[0], "max": hi[0]}}]
    x = torch.tensor([[0.0, 5.3]])  # dim 1 deviates by 0.3
    idx = torch.tensor([0])
    new = Normalize(
        feat, {"ACTION": NormalizationMode.MIN_MAX}, per_dataset_stats=stats, zero_range_center=True
    )
    z = new({"action": x}, idx)["action"][0]
    assert z[1].item() == pytest.approx(0.6, abs=1e-5)  # 2*0.3


# --------------------------------------------------------------------------- #
# config version plumbing: extract / peek / save concretization              #
# --------------------------------------------------------------------------- #


def test_extract_config_version():
    assert _extract_config_version({"config_version": 1}) == 1
    assert _extract_config_version({"policy": {"config_version": 0}}) == 0
    assert _extract_config_version({"config_version": None}) is None
    assert _extract_config_version({}) is None
    assert _extract_config_version({"config_version": True}) is None  # bool rejected
    assert _extract_config_version("not a dict") is None


def test_peek_config_version_prefers_config_json(tmp_path):
    assert _peek_config_version(tmp_path) is None  # empty dir
    (tmp_path / "train_config.json").write_text(json.dumps({"policy": {"config_version": 1}}))
    assert _peek_config_version(tmp_path) == 1  # nested train_config
    (tmp_path / "config.json").write_text(json.dumps({"config_version": 0}))
    assert _peek_config_version(tmp_path) == 0  # config.json checked first


def test_peek_config_version_reads_hf_config_json(tmp_path):
    """Converted checkpoint repos publish the train config as hf_config.json (a
    full passthrough carrying nested .policy.config_version); the peek reads it."""
    (tmp_path / "hf_config.json").write_text(
        json.dumps({"policy": {"config_version": 1, "pretrained_path": "TensorAuto/some-repo"}})
    )
    assert _peek_config_version(tmp_path) == 1


def test_save_pretrained_concretizes_and_stamps(tmp_path):
    """A fresh config (config_version None) saved via _save_pretrained writes a
    concrete CURRENT version and stamps opentau_version, so a later fine-tune
    peeks a real int, not null."""
    cfg = make_policy_config("pi0")
    assert cfg.config_version is None
    cfg._save_pretrained(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["config_version"] == CURRENT_CONFIG_VERSION
    assert "opentau_version" in data


def test_save_pretrained_preserves_explicit_legacy(tmp_path):
    """An explicitly legacy config (a resolved v0 checkpoint) saves v0, not CURRENT."""
    cfg = make_policy_config("pi0")
    cfg.config_version = 0
    cfg._save_pretrained(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["config_version"] == 0


def test_config_version_round_trips_through_from_pretrained(tmp_path):
    """A saved config.json's config_version survives PreTrainedConfig.from_pretrained
    (the config-only load path; missing key -> None sentinel)."""
    cfg = make_policy_config("pi0")
    cfg.config_version = 0
    cfg._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.config_version == 0


# --------------------------------------------------------------------------- #
# FAST tokenizer convention guard                                             #
# --------------------------------------------------------------------------- #


class _CfgStub:
    def __init__(self, version):
        self._v = version

    def resolved_config_version(self):
        return CURRENT_CONFIG_VERSION if self._v is None else self._v

    def zero_range_centers_on_zero(self):
        return self.resolved_config_version() >= 1


def _write_sidecar(d: Path, center: bool):
    (d / ACTION_NORM_META_FILE).write_text(
        json.dumps({"config_version": 1 if center else 0, "zero_range_center": center})
    )


def _run_guard(policy_version, tok_center, tmp_path):
    if tok_center is not None:
        _write_sidecar(tmp_path, tok_center)
    stub = types.SimpleNamespace(config=_CfgStub(policy_version))
    PreTrainedPolicy._check_discrete_action_tokenizer_convention(stub, tmp_path)


def test_read_action_norm_meta_absent_is_none(tmp_path):
    assert _read_action_norm_meta(tmp_path) is None


@pytest.mark.parametrize("policy_version, tok_center", [(1, True), (0, False)])
def test_guard_passes_when_convention_matches(policy_version, tok_center, tmp_path):
    _run_guard(policy_version, tok_center, tmp_path)  # must not raise


@pytest.mark.parametrize("policy_version, tok_center", [(1, False), (0, True)])
def test_guard_raises_on_mismatch(policy_version, tok_center, tmp_path):
    with pytest.raises(ValueError, match="normalization mismatch"):
        _run_guard(policy_version, tok_center, tmp_path)


def test_guard_is_noop_when_sidecar_absent(tmp_path):
    # v1 policy + tokenizer without a sidecar (upstream / pre-versioning): no raise.
    _run_guard(1, None, tmp_path)


def test_guard_local_dir_never_hits_network(tmp_path, monkeypatch):
    """A local tokenizer directory reads the sidecar off disk — no hf_hub_download."""
    import opentau.policies.pretrained as pt

    def _boom(*a, **k):
        raise AssertionError("hf_hub_download must not be called for a local dir")

    monkeypatch.setattr(pt, "hf_hub_download", _boom)
    _write_sidecar(tmp_path, True)
    stub = types.SimpleNamespace(config=_CfgStub(1))
    pt.PreTrainedPolicy._check_discrete_action_tokenizer_convention(stub, tmp_path)  # no raise


def test_guard_remote_repo_attempts_network_and_fails_open(monkeypatch):
    """A non-local (Hub repo) tokenizer path attempts a network sidecar read; the
    reader fails open, so an unreachable repo is a no-op, not a crash."""
    import opentau.policies.pretrained as pt

    calls = {"n": 0, "local_files_only": None}

    def _fake_download(*a, local_files_only=False, **k):
        calls["n"] += 1
        calls["local_files_only"] = local_files_only
        raise OSError("simulated offline")  # fail-open path

    monkeypatch.setattr(pt, "hf_hub_download", _fake_download)
    stub = types.SimpleNamespace(config=_CfgStub(1))
    # "TensorAuto/some-fitted-tok" is not a local dir -> network attempt.
    pt.PreTrainedPolicy._check_discrete_action_tokenizer_convention(stub, "TensorAuto/some-fitted-tok")
    assert calls["n"] >= 1, "a remote repo path must attempt the network sidecar read"
    assert calls["local_files_only"] is False
