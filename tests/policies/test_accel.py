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

"""Tests for the denoising-acceleration (``accel``) uncertainty proxy.

The estimator's whole value rests on two properties from the paper — a hard zero on a
straight (maximally certain) trajectory, and growth with the amount of bending — so those
are pinned numerically against hand-computed values rather than against the implementation.
The masking tests construct the cases that would silently corrupt the score in production:
padded action dims, frozen real-time-chunking rows, and the un-executed chunk tail.
"""

import math

import pytest
import torch
from torch import nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.accel import (
    ACCEL_PREFIX_ENV,
    MIN_PREFIX,
    AccelMeter,
    AccelProvenance,
    assert_comparable,
    build_provenance,
    configure_accel,
    default_prefix,
    executed_row_mask,
    make_meter,
    resolve_action_dim_mask,
    resolve_prefix,
)
from opentau.policies.normalize import Unnormalize

CPU = torch.device("cpu")


def _meter(prefix=3, batch_size=1, dim_mask=None):
    return AccelMeter(prefix=prefix, batch_size=batch_size, device=CPU, dim_mask=dim_mask)


def _feed(meter, velocities):
    for v in velocities:
        meter.update(v)
    return meter


# --------------------------------------------------------------------------------------
# The two theoretical anchors: zero baseline (Prop A.2) and positivity (Prop A.3).
# --------------------------------------------------------------------------------------


def test_constant_velocity_scores_exactly_zero():
    """A straight, constant-velocity path is the paper's maximally-certain field (Theorem 1),
    for which ``accel`` must be exactly 0 — not merely small."""
    v = torch.randn(2, 4, 3)
    meter = _feed(_meter(prefix=5, batch_size=2), [v.clone() for _ in range(5)])
    assert torch.equal(meter.value(), torch.zeros(2))


def test_any_bend_scores_strictly_positive():
    """Prop A.3: a bend at any interior step betrays a non-degenerate posterior."""
    velocities = [torch.ones(1, 2, 2), torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)]
    meter = _feed(_meter(prefix=3), velocities)
    assert meter.value().item() > 0.0


def test_matches_hand_computed_value():
    """Pin the exact estimator, not just its sign.

    Velocities (single row, 2 dims): (1,0) -> (0,1) -> (1,0). Each consecutive difference
    has norm sqrt(2), each velocity has norm 1, so accel_3 = 3 * (2*sqrt(2)) / 3 = 2*sqrt(2).
    """
    velocities = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    ]
    meter = _feed(_meter(prefix=3), velocities)
    assert meter.value().item() == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-6)


def test_score_grows_with_bend_magnitude():
    """Local monotonicity (Prop A.4): more curvature => strictly larger score.

    The paper's guarantee is in expectation over a small-spread family, so this pins the
    weaker but testable direction — a strictly sharper bend scores strictly higher.
    """
    scores = []
    for bend in (0.1, 0.5, 2.0):
        velocities = [
            torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[[1.0, bend]]]),
            torch.tensor([[[1.0, 0.0]]]),
        ]
        scores.append(_feed(_meter(prefix=3), velocities).value().item())
    assert scores[0] < scores[1] < scores[2]


# --------------------------------------------------------------------------------------
# Prefix semantics — the detail most likely to be silently reversed by a refactor.
# --------------------------------------------------------------------------------------


def test_prefix_truncates_the_tail_not_the_head():
    """The prefix must integrate the FIRST p steps (the noise end).

    Constructed so the two ends disagree: the first two velocities are identical (no bend)
    while the last two differ sharply. A prefix of 2 must therefore score 0. An
    implementation that took the *last* p steps — the singular, discretization-dominated
    end the paper explicitly discards — would score positive here.
    """
    velocities = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 5.0]]]),
    ]
    assert _feed(_meter(prefix=2), velocities).value().item() == 0.0
    assert _feed(_meter(prefix=3), velocities).value().item() > 0.0


def test_updates_past_the_prefix_are_ignored():
    """Feeding a longer schedule than the prefix must not change the score."""
    velocities = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
    ]
    short = _feed(_meter(prefix=2), velocities).value().item()
    long = _feed(_meter(prefix=2), velocities + [torch.tensor([[[9.0, 9.0]]])] * 5).value().item()
    assert short == long
    assert _feed(_meter(prefix=2), velocities + [torch.zeros(1, 1, 2)] * 5).steps == 2


def test_multiplier_follows_the_steps_actually_accumulated():
    """Algorithm 1's running form is ``J / (S / (t+1))``, so a schedule shorter than the
    requested prefix degrades gracefully instead of scaling by a step count it never ran."""
    velocities = [torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]])]
    meter = _feed(_meter(prefix=10), velocities)
    assert meter.steps == 2
    assert meter.value().item() == pytest.approx(2 * math.sqrt(2.0) / 2.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# Undefined-score handling. 0.0 would read as "maximally certain" — the opposite of
# "no measurement" — so these must be NaN.
# --------------------------------------------------------------------------------------


def test_single_velocity_is_nan_not_zero():
    meter = _feed(_meter(prefix=5), [torch.randn(1, 2, 2)])
    assert math.isnan(meter.value().item())


def test_all_zero_velocities_are_nan_not_zero():
    """An empty denominator means every scored element was masked out, which is an absence
    of measurement rather than evidence of certainty."""
    meter = _feed(_meter(prefix=3), [torch.zeros(1, 2, 2)] * 3)
    assert math.isnan(meter.value().item())


def test_prefix_below_two_is_rejected():
    """A 1-step prefix has an empty numerator and would report 0.0 for every field."""
    with pytest.raises(ValueError, match=f">= {MIN_PREFIX}"):
        _meter(prefix=1)
    with pytest.raises(ValueError, match=f">= {MIN_PREFIX}"):
        resolve_prefix(1, 10)


def test_prefix_helpers():
    assert default_prefix(10) == 9
    assert default_prefix(5) == 4
    assert default_prefix(2) == 2
    assert resolve_prefix(None, 10) is None
    assert resolve_prefix(9, 5) == 5, "prefix must clamp to the schedule length"
    with pytest.raises(ValueError, match="at least"):
        default_prefix(1)


# --------------------------------------------------------------------------------------
# Batching — these samplers run 16-env vectorized rollouts.
# --------------------------------------------------------------------------------------


def test_per_sample_scores_are_independent():
    """A flat norm over the batch would average independent, concurrently-running episodes
    into one number. Sample 0 travels straight; sample 1 bends."""
    velocities = [
        torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
    ]
    value = _feed(_meter(prefix=3, batch_size=2), velocities).value()
    assert value.shape == (2,)
    assert value[0].item() == 0.0
    assert value[1].item() == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-6)


# --------------------------------------------------------------------------------------
# Masking. Both of these are silent in production: the score keeps coming out, just wrong.
# --------------------------------------------------------------------------------------


def test_dim_mask_excludes_padded_action_dims():
    """Padded dims are unsupervised network output (masked out of the training loss), so
    their noise must not enter either sum.

    Dim 0 is real and travels straight; dim 1 is 'padding' carrying pure bend. Masked, the
    score must be exactly 0; unmasked it is positive — that gap is the contamination the
    mask exists to remove.
    """
    velocities = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 7.0]]]),
        torch.tensor([[[1.0, -7.0]]]),
    ]
    unmasked = _feed(_meter(prefix=3), velocities).value().item()
    masked = _feed(_meter(prefix=3, dim_mask=torch.tensor([[True, False]])), velocities).value().item()
    assert unmasked > 0.0
    assert masked == 0.0


def test_row_mask_excludes_frozen_and_unexecuted_rows():
    """Row 0 is a frozen RTC prefix row, row 2 is outside the executed window; only row 1
    should be scored. The excluded rows carry all the bend here."""
    velocities = [
        torch.tensor([[[1.0], [1.0], [1.0]]]),
        torch.tensor([[[9.0], [1.0], [9.0]]]),
    ]
    meter = _meter(prefix=2)
    meter.set_row_mask(torch.tensor([[False, True, False]]))
    assert _feed(meter, velocities).value().item() == 0.0


def test_row_and_dim_masks_compose():
    """`set_row_mask` must multiply into an existing dim mask, not replace it."""
    meter = _meter(prefix=2, dim_mask=torch.tensor([[True, False]]))
    meter.set_row_mask(torch.tensor([[True, False]]))
    velocities = [
        torch.tensor([[[1.0, 3.0], [4.0, 5.0]]]),
        torch.tensor([[[1.0, -3.0], [-4.0, -5.0]]]),
    ]
    # Only element [row 0, dim 0] survives, and it is constant across both steps.
    assert _feed(meter, velocities).value().item() == 0.0


# --------------------------------------------------------------------------------------
# `executed_row_mask` — the slice that is wrong whenever delay > 0.
# --------------------------------------------------------------------------------------


def test_executed_row_mask_without_delay():
    mask = executed_row_mask(
        prefix_mask=torch.zeros(1, 5, dtype=torch.bool),
        delay=torch.tensor(0),
        chunk_size=5,
        n_action_steps=3,
        device=CPU,
    )
    assert mask.tolist() == [[True, True, True, False, False]]


def test_executed_row_mask_offsets_the_window_by_delay():
    """``select_action`` executes ``actions[delay : delay + n_action_steps]``. The naive
    ``[:n_action_steps]`` slice would score the frozen rows and miss the executed tail —
    a mistake that only manifests under real-time chunking."""
    mask = executed_row_mask(
        prefix_mask=torch.tensor([[True, True, False, False, False]]),
        delay=torch.tensor(2),
        chunk_size=5,
        n_action_steps=2,
        device=CPU,
    )
    assert mask.tolist() == [[False, False, True, True, False]]


def test_executed_row_mask_accepts_per_sample_delay():
    mask = executed_row_mask(
        prefix_mask=torch.tensor([[False, False, False], [True, False, False]]),
        delay=torch.tensor([0, 1]),
        chunk_size=3,
        n_action_steps=1,
        device=CPU,
    )
    assert mask.tolist() == [[True, False, False], [False, True, False]]


# --------------------------------------------------------------------------------------
# Action-dim mask derivation from the normalization buffers.
# --------------------------------------------------------------------------------------


def _unnormalize(mode, stats, action_dim=4, eps=1e-6):
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return Unnormalize(
        features,
        {"ACTION": mode},
        per_dataset_stats=stats,
        dataset_names=[f"ds{i}" for i in range(len(stats))],
        eps=eps,
    )


def test_dim_mask_recovers_the_pad_tail_from_mean_std_buffers():
    """Two real dims, two zero-variance pad dims — the LIBERO-in-32 shape in miniature."""
    stats = [{"actions": {"mean": torch.zeros(4), "std": torch.tensor([1.0, 2.0, 0.0, 0.0])}}]
    mask = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.MEAN_STD, stats),
        max_action_dim=4,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    assert mask.tolist() == [[True, True, False, False]]


def test_dim_mask_recovers_the_pad_tail_from_quantile_buffers():
    stats = [
        {
            "actions": {
                "q01": torch.tensor([-1.0, -1.0, 0.0, 0.0]),
                "q99": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            }
        }
    ]
    mask = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.QUANTILE, stats),
        max_action_dim=4,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    assert mask.tolist() == [[True, True, False, False]]


def test_dim_mask_is_gathered_per_norm_head():
    """Buffers are ``(num_datasets, action_dim)``. On a co-trained mixture two samples in
    one batch legitimately score a different number of dims, so a single shared constant
    mask would be wrong for one of them."""
    stats = [
        {"actions": {"mean": torch.zeros(4), "std": torch.tensor([1.0, 1.0, 0.0, 0.0])}},
        {"actions": {"mean": torch.zeros(4), "std": torch.tensor([1.0, 1.0, 1.0, 0.0])}},
    ]
    mask = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.MEAN_STD, stats),
        max_action_dim=4,
        dataset_index=torch.tensor([0, 1]),
    )
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_dim_mask_pads_when_the_sampler_is_wider_than_the_buffer():
    stats = [{"actions": {"mean": torch.zeros(2), "std": torch.tensor([1.0, 1.0])}}]
    mask = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.MEAN_STD, stats, action_dim=2),
        max_action_dim=4,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    assert mask.tolist() == [[True, True, False, False]]


def test_dim_mask_reads_eps_off_the_live_module():
    """``eps`` is ``config_version``-dependent (1e-6 at v1, 1e-8 at v0), so reading the
    module-level default instead of the live attribute would classify a dim whose std sits
    between the two conventions under the wrong checkpoint's rule."""
    stats = [{"actions": {"mean": torch.zeros(2), "std": torch.tensor([1.0, 1e-7])}}]
    strict = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.MEAN_STD, stats, action_dim=2, eps=1e-6),
        max_action_dim=2,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    loose = resolve_action_dim_mask(
        _unnormalize(NormalizationMode.MEAN_STD, stats, action_dim=2, eps=1e-8),
        max_action_dim=2,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    assert strict.tolist() == [[True, False]]
    assert loose.tolist() == [[True, True]]


def test_dim_mask_refuses_identity_normalization():
    """IDENTITY has no stats buffer, so the pad tail is underivable — and IDENTITY also
    leaves per-dim scale heterogeneous, violating the estimator's premise outright.
    Refusing is the point: silently passing an all-ones mask would emit a plausible number
    computed over 25 columns of noise."""
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    unnorm = Unnormalize(features, {"ACTION": NormalizationMode.IDENTITY}, num_datasets=1)
    with pytest.raises(ValueError, match="IDENTITY"):
        resolve_action_dim_mask(unnorm, max_action_dim=4, dataset_index=torch.zeros(1, dtype=torch.long))


def test_dim_mask_refuses_an_all_degenerate_head():
    stats = [{"actions": {"mean": torch.zeros(2), "std": torch.zeros(2)}}]
    with pytest.raises(ValueError, match="degenerate"):
        resolve_action_dim_mask(
            _unnormalize(NormalizationMode.MEAN_STD, stats, action_dim=2),
            max_action_dim=2,
            dataset_index=torch.zeros(1, dtype=torch.long),
        )


def test_degenerate_head_error_points_at_the_routing_not_the_stats():
    """The realistic cause is a wrongly-routed sample, so the message must say so.

    A co-trained mixture can carry a placeholder head with all-zero stats (OpenTau's own
    `ci_config` has one), and an observation with no dataset provenance falls back to head 0
    — which may be exactly that placeholder. A bare "the mask is empty" sends the reader off
    to audit their action statistics when the real fix is to tag the observation. So the
    error must name the selected head, the per-head scorable counts, and the usable heads.
    """
    stats = [
        {"actions": {"mean": torch.zeros(4), "std": torch.zeros(4)}},  # placeholder head
        {"actions": {"mean": torch.zeros(4), "std": torch.tensor([1.0, 2.0, 3.0, 0.0])}},
    ]
    unnorm = _unnormalize(NormalizationMode.MEAN_STD, stats)
    with pytest.raises(ValueError) as excinfo:
        resolve_action_dim_mask(unnorm, max_action_dim=4, dataset_index=torch.zeros(1, dtype=torch.long))
    message = str(excinfo.value)
    assert "[0]" in message, "must name the head that was actually selected"
    assert "[0, 3]" in message, "must report scorable dims per head so the contrast is visible"
    assert "[1]" in message and "dataset_repo_id" in message, (
        "must point at the routing fix when another head does have usable stats"
    )

    # And head 1 really is usable — otherwise the advice would be wrong.
    assert resolve_action_dim_mask(
        unnorm, max_action_dim=4, dataset_index=torch.ones(1, dtype=torch.long)
    ).tolist() == [[True, True, True, False]]


def test_degenerate_error_says_so_when_no_head_is_usable():
    stats = [{"actions": {"mean": torch.zeros(2), "std": torch.zeros(2)}}]
    with pytest.raises(ValueError, match="No head has usable action stats"):
        resolve_action_dim_mask(
            _unnormalize(NormalizationMode.MEAN_STD, stats, action_dim=2),
            max_action_dim=2,
            dataset_index=torch.zeros(1, dtype=torch.long),
        )


# --------------------------------------------------------------------------------------
# `make_meter` / provenance.
# --------------------------------------------------------------------------------------


class _FakeConfig:
    type = "pi05"
    num_steps = 10
    chunk_size = 4
    n_action_steps = 4
    max_delay = 0
    max_action_dim = 4
    delta_action_state_map = None
    normalization_mapping = {"ACTION": NormalizationMode.MEAN_STD}


class _FakePolicy(nn.Module):
    def __init__(self, accel_prefix=None):
        super().__init__()
        self.config = _FakeConfig()
        self.accel_prefix = accel_prefix
        stats = [{"actions": {"mean": torch.zeros(4), "std": torch.tensor([1.0, 2.0, 0.0, 0.0])}}]
        self.unnormalize_outputs = _unnormalize(NormalizationMode.MEAN_STD, stats)


def test_make_meter_returns_none_when_disabled():
    assert make_meter(_FakePolicy(accel_prefix=None), batch_size=1, device=CPU) is None


def test_make_meter_builds_the_dim_mask_and_clamps_the_prefix():
    meter = make_meter(_FakePolicy(accel_prefix=99), batch_size=1, device=CPU)
    assert meter is not None
    assert meter.prefix == 10, "prefix must clamp to num_steps"
    assert meter.dim_mask.tolist() == [[True, True, False, False]]


def test_provenance_records_the_scoring_configuration():
    policy = _FakePolicy(accel_prefix=9)
    meter = make_meter(policy, batch_size=2, device=CPU, dataset_index=torch.zeros(2, dtype=torch.long))
    prov = build_provenance(policy, meter, dataset_index=torch.zeros(2, dtype=torch.long))
    assert prov.policy_type == "pi05"
    assert prov.num_steps == 10
    assert prov.prefix == 9
    assert prov.action_norm_mode == "MEAN_STD"
    assert prov.num_scored_dims == (2, 2)
    assert AccelProvenance.from_dict(prov.to_dict()) == prov


def _prov(**overrides):
    base = {
        "policy_type": "pi05",
        "num_steps": 10,
        "prefix": 9,
        "chunk_size": 10,
        "n_action_steps": 10,
        "max_delay": 0,
        "action_norm_mode": "MEAN_STD",
        "has_delta_action_map": False,
        "velocity_dtype": "bfloat16",
    }
    base.update(overrides)
    return AccelProvenance(**base)


# --------------------------------------------------------------------------------------
# `configure_accel` — the single enable knob shared by every serving entry point.
# --------------------------------------------------------------------------------------


class _FakeCfg:
    def __init__(self, accel_prefix=None):
        self.policy = _FakeConfig()
        if accel_prefix is not None:
            self.policy.accel_prefix = accel_prefix


def test_configure_accel_is_off_by_default(monkeypatch):
    """Nothing requested => the attribute is untouched and every line in the sampler stays
    dead. A free score is still a score nothing should enable implicitly."""
    monkeypatch.delenv(ACCEL_PREFIX_ENV, raising=False)
    policy = _FakePolicy()
    assert configure_accel(policy, _FakeCfg()) is None
    assert policy.accel_prefix is None


def test_configure_accel_auto_resolves_the_papers_prefix(monkeypatch):
    monkeypatch.setenv(ACCEL_PREFIX_ENV, "auto")
    policy = _FakePolicy()
    assert configure_accel(policy, _FakeCfg()) == 9  # default_prefix(10)
    assert policy.accel_prefix == 9


def test_configure_accel_precedence_override_beats_config_beats_env(monkeypatch):
    monkeypatch.setenv(ACCEL_PREFIX_ENV, "3")
    assert configure_accel(_FakePolicy(), _FakeCfg()) == 3
    assert configure_accel(_FakePolicy(), _FakeCfg(accel_prefix=5)) == 5
    assert configure_accel(_FakePolicy(), _FakeCfg(accel_prefix=5), override=7) == 7


def test_configure_accel_ignores_a_blank_env_value(monkeypatch):
    monkeypatch.setenv(ACCEL_PREFIX_ENV, "   ")
    assert configure_accel(_FakePolicy(), _FakeCfg()) is None


def test_configure_accel_reads_the_real_policy_config_field(monkeypatch):
    """`accel_prefix` is a real field on ``PreTrainedConfig``, not just an env var.

    Draccus JSON configs are the repo's primary interface, so a feature reachable only
    through the environment is effectively unreachable. Uses a genuine ``PI05Config`` rather
    than the fake above, so the test fails if the field is ever dropped from the dataclass.
    """
    from opentau.policies.pi05.configuration_pi05 import PI05Config

    monkeypatch.delenv(ACCEL_PREFIX_ENV, raising=False)
    cfg = _FakeCfg()
    cfg.policy = PI05Config()
    assert cfg.policy.accel_prefix is None, "accel must be off by default in the config"
    assert configure_accel(_FakePolicy(), cfg) is None

    cfg.policy.accel_prefix = "auto"
    assert configure_accel(_FakePolicy(), cfg) == 9  # default_prefix(num_steps=10)
    cfg.policy.accel_prefix = 4
    assert configure_accel(_FakePolicy(), cfg) == 4


def test_configure_accel_refuses_an_unwired_policy(monkeypatch):
    """A family whose sampler was never wired would accept the attribute and then never read
    it — a silent no-op an operator would misread as 'the score is always missing'."""
    monkeypatch.delenv(ACCEL_PREFIX_ENV, raising=False)
    policy = _FakePolicy()
    del policy.accel_prefix
    with pytest.raises(ValueError, match="not wired"):
        configure_accel(policy, _FakeCfg(), override=4)


def test_configure_accel_rejects_a_nonsense_prefix(monkeypatch):
    monkeypatch.delenv(ACCEL_PREFIX_ENV, raising=False)
    with pytest.raises(ValueError, match="integer"):
        configure_accel(_FakePolicy(), _FakeCfg(), override="nonsense")
    with pytest.raises(ValueError, match=f">= {MIN_PREFIX}"):
        configure_accel(_FakePolicy(), _FakeCfg(), override=1)


def test_comparable_provenance_passes():
    assert_comparable(_prov(), _prov(dataset_index=(3,), num_scored_dims=(7,)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_steps", 5),
        ("prefix", 4),
        ("action_norm_mode", "QUANTILE"),
        ("has_delta_action_map", True),
        ("velocity_dtype", "float32"),
        ("max_delay", 2),
        ("policy_type", "pi07"),
    ],
)
def test_mismatched_provenance_is_refused(field, value):
    """Every one of these shifts the score's distribution, so a threshold calibrated under
    one must not be silently applied under another."""
    with pytest.raises(ValueError, match="not applicable"):
        assert_comparable(_prov(), _prov(**{field: value}))
