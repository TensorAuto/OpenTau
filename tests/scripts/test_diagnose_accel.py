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

"""Tests for the ``accel`` prefix study.

The claim under test is narrow and easy to fake: that this diagnostic *selects* a prefix
rather than merely describing one. The distinction matters because the obvious summary —
mean ``accel_p`` against ``p`` — cannot select anything. Both of its sums accumulate over
the prefix, so the curve rises with ``p`` whether or not the extra Euler steps carry
posterior information, and reading its peak as "the best prefix" is a measurement that
would return the same answer on data with no signal at all.

So the central test here (`test_the_study_follows_rank_correlation_not_the_rising_mean`)
constructs a policy where the two criteria give *different* answers, and pins that the study
follows the correlation. Without that disagreement the test would pass under either
implementation, which is the failure mode CLAUDE.md rule 8 exists to prevent.
"""

import numpy as np
import pytest
import torch
from torch import nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.accel import (
    AccelMeter,
    action_dim_scale,
    make_meter,
    record_traces,
)
from opentau.policies.normalize import Unnormalize
from opentau.scripts.diagnose_accel import (
    AccelDiagnosticReport,
    _allocate_draws,
    _posterior_spread,
    _signal_to_floor,
    _spearman,
    format_report,
    measure_prefix_quality,
)

CPU = torch.device("cpu")
ACTION_DIM = 4
CHUNK = 3
NUM_STEPS = 10


def _unnormalize(mode, stats, action_dim=ACTION_DIM):
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return Unnormalize(features, {FeatureType.ACTION: mode}, stats)


# --------------------------------------------------------------------------------------
# `AccelMeter.value_at` — every prefix out of one pass.
# --------------------------------------------------------------------------------------


def test_value_at_equals_a_meter_independently_built_at_that_prefix():
    """The whole trace mechanism is only sound if it is *exactly* the shorter-prefix score.

    Bit-identical, not merely close: `value_at` reads snapshots of the same running sums the
    real meter accumulates, so any drift would mean the snapshot is taken at the wrong point
    in `update` (e.g. before the numerator is folded in) and would shift every prefix by one
    step — a silent off-by-one that a tolerance-based assert would wave through.
    """
    generator = torch.Generator().manual_seed(0)
    velocities = [torch.randn(2, CHUNK, ACTION_DIM, generator=generator) for _ in range(6)]

    traced = AccelMeter(prefix=6, batch_size=2, device=CPU, record_trace=True)
    for velocity in velocities:
        traced.update(velocity)

    for prefix in range(2, 7):
        reference = AccelMeter(prefix=prefix, batch_size=2, device=CPU)
        for velocity in velocities:
            reference.update(velocity)
        assert torch.equal(traced.value_at(prefix), reference.value()), (
            f"value_at({prefix}) must reproduce a meter built at prefix={prefix} exactly"
        )


def test_value_at_is_unavailable_without_tracing():
    """Off by default: the serving hot path must not pay for a diagnostic feature."""
    meter = AccelMeter(prefix=3, batch_size=1, device=CPU)
    for _ in range(3):
        meter.update(torch.ones(1, CHUNK, ACTION_DIM))
    assert meter.record_trace is False
    with pytest.raises(RuntimeError, match="record_trace"):
        meter.value_at(2)


def test_value_at_rejects_a_prefix_the_trace_cannot_support():
    meter = AccelMeter(prefix=5, batch_size=1, device=CPU, record_trace=True)
    for _ in range(3):
        meter.update(torch.ones(1, CHUNK, ACTION_DIM))
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        meter.value_at(4)
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        meter.value_at(1)


# --------------------------------------------------------------------------------------
# `record_traces` — reaching the meter without widening a policy signature.
# --------------------------------------------------------------------------------------


class _Config:
    type = "pi05"
    num_steps = NUM_STEPS
    chunk_size = CHUNK
    n_action_steps = CHUNK
    max_delay = 0
    max_action_dim = ACTION_DIM
    delta_action_state_map = None
    normalization_mapping = {"ACTION": NormalizationMode.MEAN_STD}


def _policy_stats(std=(1.0, 1.0, 1.0, 1.0)):
    return [{"actions": {"mean": torch.zeros(ACTION_DIM), "std": torch.tensor(std)}}]


class _MeterOnlyPolicy(nn.Module):
    def __init__(self, accel_prefix=NUM_STEPS):
        super().__init__()
        self.config = _Config()
        self.accel_prefix = accel_prefix
        self.unnormalize_outputs = _unnormalize(NormalizationMode.MEAN_STD, _policy_stats())


def test_meters_built_outside_the_context_neither_trace_nor_leak():
    meter = make_meter(_MeterOnlyPolicy(), batch_size=1, device=CPU)
    assert meter.record_trace is False


def test_the_context_hands_back_every_meter_built_inside_it():
    policy = _MeterOnlyPolicy()
    with record_traces() as meters:
        first = make_meter(policy, batch_size=1, device=CPU)
        second = make_meter(policy, batch_size=1, device=CPU)
    assert meters == [first, second]
    assert all(m.record_trace for m in meters)
    # And the switch is scoped: the next meter is back to the serving configuration.
    assert make_meter(policy, batch_size=1, device=CPU).record_trace is False


def test_the_context_restores_tracing_when_the_body_raises():
    policy = _MeterOnlyPolicy()
    with pytest.raises(RuntimeError, match="boom"), record_traces():
        raise RuntimeError("boom")
    assert make_meter(policy, batch_size=1, device=CPU).record_trace is False


# --------------------------------------------------------------------------------------
# `action_dim_scale` — putting a raw-unit spread into the units accel is computed in.
# --------------------------------------------------------------------------------------


def test_scale_is_the_std_under_mean_std():
    stats = _policy_stats(std=(2.0, 4.0, 1.0, 0.0))
    scale = action_dim_scale(
        _unnormalize(NormalizationMode.MEAN_STD, stats),
        max_action_dim=ACTION_DIM,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    # The degenerate dim becomes 1.0 so callers can divide unconditionally; it is masked out.
    assert scale.tolist() == [[2.0, 4.0, 1.0, 1.0]]


def test_scale_is_half_the_range_under_min_max():
    """`[lo, hi]` maps onto `[-1, 1]`, so the divisor is half the range, not the range.

    Getting this factor wrong would not break anything visibly — it rescales every
    observation by the same constant, and the study's rank correlation is invariant to that.
    It matters because the spread is also reported as an absolute number in the JSON, where a
    2x error is indistinguishable from a real one.
    """
    stats = [{"actions": {"min": torch.zeros(ACTION_DIM), "max": torch.tensor([2.0, 6.0, 1.0, 0.0])}}]
    scale = action_dim_scale(
        _unnormalize(NormalizationMode.MIN_MAX, stats),
        max_action_dim=ACTION_DIM,
        dataset_index=torch.zeros(1, dtype=torch.long),
    )
    assert scale.tolist() == [[1.0, 3.0, 0.5, 1.0]]


# --------------------------------------------------------------------------------------
# `_posterior_spread` — the reference the prefix study scores against.
# --------------------------------------------------------------------------------------


def _spread_policy(std=(1.0, 1.0, 1.0, 1.0), n_action_steps=CHUNK):
    policy = _MeterOnlyPolicy()
    policy.unnormalize_outputs = _unnormalize(NormalizationMode.MEAN_STD, _policy_stats(std))
    policy.config = type("C", (_Config,), {"n_action_steps": n_action_steps})()
    return policy


def test_spread_is_invariant_to_a_constant_offset_across_resamples():
    """This is what lets a delta-action policy be measured without inverting the transform.

    `sample_actions` re-adds the chunk-start state before returning, but that offset is the
    same for every resample of one observation, so it cancels in the deviation. If it did
    not, the spread would be contaminated by the state and the study would be correlating
    `accel` against the robot's pose.
    """
    generator = torch.Generator().manual_seed(3)
    chunks = [torch.randn(1, CHUNK, ACTION_DIM, generator=generator) for _ in range(8)]
    offset = torch.full((1, CHUNK, ACTION_DIM), 17.0)
    policy = _spread_policy()
    index = torch.zeros(1, dtype=torch.long)

    plain = _posterior_spread(policy, chunks, index)
    shifted = _posterior_spread(policy, [c + offset for c in chunks], index)
    assert plain == pytest.approx(shifted, rel=1e-5)


def test_spread_standardizes_by_the_per_dim_norm_scale():
    """A dim with 10x the raw scale must not contribute 10x the spread."""
    generator = torch.Generator().manual_seed(4)
    base = [torch.randn(1, CHUNK, ACTION_DIM, generator=generator) for _ in range(8)]
    # Blow up dim 1 tenfold in raw space, and tell the norm stats about it.
    stretched = []
    for chunk in base:
        scaled = chunk.clone()
        scaled[:, :, 1] *= 10.0
        stretched.append(scaled)

    plain = _posterior_spread(_spread_policy(), base, torch.zeros(1, dtype=torch.long))
    matched = _posterior_spread(
        _spread_policy(std=(1.0, 10.0, 1.0, 1.0)), stretched, torch.zeros(1, dtype=torch.long)
    )
    assert plain == pytest.approx(matched, rel=1e-5), (
        "standardizing by the norm scale must undo a per-dim unit change"
    )


def test_spread_ignores_padded_dims_and_rows_past_the_executed_window():
    """Both exclusions mirror the meter: unsupervised output, and actions never applied."""
    chunks = []
    for value in range(6):
        chunk = torch.zeros(1, CHUNK, ACTION_DIM)
        chunk[:, :, 3] = value * 100.0  # padded dim (std 0) — must not register
        chunk[:, 2, 0] = value * 100.0  # row past n_action_steps=2 — must not register
        chunks.append(chunk)

    policy = _spread_policy(std=(1.0, 1.0, 1.0, 0.0), n_action_steps=2)
    assert _posterior_spread(policy, chunks, torch.zeros(1, dtype=torch.long)) == [0.0]


# --------------------------------------------------------------------------------------
# `_spearman`.
# --------------------------------------------------------------------------------------


def test_spearman_is_monotone_not_linear():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert _spearman(xs, [1.0, 8.0, 27.0, 64.0]) == pytest.approx(1.0)
    assert _spearman(xs, [-1.0, -8.0, -27.0, -64.0]) == pytest.approx(-1.0)


def test_spearman_declines_to_answer_rather_than_warning():
    assert np.isnan(_spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])), "constant input has no ranking"
    assert np.isnan(_spearman([1.0, 2.0], [1.0, 2.0])), "two points always correlate perfectly"
    assert _spearman([1.0, 2.0, 3.0, float("nan")], [1.0, 2.0, 3.0, 0.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# The study itself.
# --------------------------------------------------------------------------------------


class _ScriptedFlowPolicy(nn.Module):
    """A policy whose curvature and posterior spread are dialled independently.

    ``state[0]`` carries an uncertainty level ``u``. The returned chunk's spread across
    resamples is proportional to ``u`` (so ``u`` *is* the ground truth the study is trying to
    recover), while the scripted velocity sequence makes the *early* prefix track ``u`` and
    the *late* prefix anti-track it. Velocities lie along one axis so their norms are exactly
    the scripted magnitudes and the arithmetic stays inspectable.
    """

    def __init__(self, late_slope: float):
        super().__init__()
        self.config = _Config()
        self.accel_prefix = None
        self.unnormalize_outputs = _unnormalize(NormalizationMode.MEAN_STD, _policy_stats())
        self.register_parameter("_anchor", nn.Parameter(torch.zeros(1)))
        self.late_slope = late_slope

    def _resolve_dataset_index(self, batch):
        return torch.zeros(len(batch["state"]), dtype=torch.long)

    def sample_actions(self, batch, noise=None, **kwargs):
        uncertainty = float(batch["state"][0, 0])
        meter = make_meter(self, batch_size=1, device=CPU, dataset_index=self._resolve_dataset_index(batch))

        # Magnitudes along one axis: |v_t - v_{t-1}| is exactly the step in this list.
        # The late step must stay POSITIVE as `late_slope` bites: accel's numerator sums
        # norms, so a step driven negative comes back as a large magnitude and restores the
        # correlation it was meant to destroy. 0.2 keeps it positive across the `u` range.
        magnitude = 1.0
        for step in range(self.config.num_steps):
            if step == 1:
                magnitude += uncertainty  # early curvature grows with the true spread
            elif step >= 2:
                magnitude += 0.2 - self.late_slope * uncertainty  # late curvature fights it
            velocity = torch.zeros(1, CHUNK, ACTION_DIM)
            velocity[:, 0, 0] = magnitude
            if meter is not None:
                meter.update(velocity)

        # Posterior spread proportional to u, so the reference measure recovers it.
        return uncertainty * noise[:, :CHUNK, :ACTION_DIM]


def _observations(levels):
    return [{"state": torch.tensor([[level] + [0.0] * 3])} for level in levels]


def test_the_study_follows_rank_correlation_not_the_rising_mean():
    """The point of the change: the criterion is rho, and rho can disagree with the mean.

    This policy is built so the two answers differ. Early curvature tracks the true posterior
    spread; late curvature is scripted to fight it, so `accel_T` *anti*-correlates. But the
    per-prefix mean still rises monotonically with `p`, because both of accel's sums
    accumulate. An implementation that picked the largest mean would therefore choose the
    worst available prefix, and this test would fail — which is exactly what makes it a pin
    on the criterion rather than on the plumbing.
    """
    policy = _ScriptedFlowPolicy(late_slope=1.0)
    study = measure_prefix_quality(
        policy, _observations([0.01, 0.02, 0.03, 0.05, 0.08, 0.13]), num_resamples=8, seed=0
    )

    assert study.best_prefix == 2, "the informative prefix here is the earliest one"
    assert study.rho[2] == pytest.approx(1.0), "early accel tracks the true spread exactly"
    assert study.rho[NUM_STEPS] < 0.0, "late accel was scripted to anti-track it"

    means = {p: float(np.mean(v)) for p, v in study.accel_by_prefix.items()}
    assert max(means, key=lambda p: means[p]) == NUM_STEPS, (
        "the mean sweep must peak at the WORST prefix — otherwise the two criteria agree "
        "and this test would pass under either implementation"
    )


def test_the_study_endorses_the_default_when_the_default_is_right():
    """The mirror image: no manufactured disagreement, so `T - 1` should win on its merits."""
    policy = _ScriptedFlowPolicy(late_slope=0.0)
    study = measure_prefix_quality(
        policy, _observations([0.01, 0.02, 0.03, 0.05, 0.08, 0.13]), num_resamples=8, seed=0
    )
    assert study.rho_at_default == pytest.approx(1.0)
    assert study.rho[study.best_prefix] == pytest.approx(study.rho_at_default)


def test_the_reference_spread_recovers_the_scripted_uncertainty():
    """If the reference did not track `u`, every rho above would be meaningless."""
    levels = [0.01, 0.02, 0.03, 0.05, 0.08, 0.13]
    study = measure_prefix_quality(
        _ScriptedFlowPolicy(late_slope=0.0), _observations(levels), num_resamples=8, seed=0
    )
    assert _spearman(study.divergences, levels) == pytest.approx(1.0)


def test_the_study_refuses_a_sample_too_small_to_rank():
    with pytest.raises(ValueError, match="rank correlation"):
        measure_prefix_quality(
            _ScriptedFlowPolicy(late_slope=0.0), _observations([0.01, 0.02]), num_resamples=4
        )


# --------------------------------------------------------------------------------------
# The go/no-go verdict — "not measured" must never render as "OK".
# --------------------------------------------------------------------------------------


def _report(**overrides):
    payload = {
        "num_steps": NUM_STEPS,
        "prefix": 9,
        "num_scored_dims": [ACTION_DIM],
        "max_action_dim": ACTION_DIM,
        "float32_scores": [],
        "serving_scores": [0.4, 0.5],
        "serving_dtype": "bfloat16",
        "dtype_floor": float("nan"),
        "noise_spread": 0.01,
        "observation_spread": 0.09,
        "signal_to_floor": float("nan"),
    }
    payload.update(overrides)
    return AccelDiagnosticReport(**payload)


def test_an_unmeasured_floor_is_not_an_infinite_ratio():
    """NaN is truthy and non-finite, so the obvious guard silently yields `inf`.

    That is the whole bug: skipping the float32 leg leaves nothing to compare against, but
    an `inf` ratio is indistinguishable from the best possible result.
    """
    assert np.isnan(_signal_to_floor(0.09, float("nan")))


def test_a_floor_measured_at_exactly_zero_is_still_infinite():
    """The correction must not overreach — a real zero floor is a real infinite ratio.

    float32 and the serving dtype agreeing on every observation is a legitimate (if
    unlikely) measurement, and it genuinely means the rounding floor does not bind.
    """
    assert _signal_to_floor(0.09, 0.0) == float("inf")
    assert _signal_to_floor(0.09, 0.009) == pytest.approx(10.0)


def test_the_report_refuses_to_certify_a_floor_it_never_measured():
    """Every `<` comparison against NaN is False, so the thresholds fall through to "OK".

    Without an explicit NaN branch ahead of them, a skipped run prints the single most
    reassuring line the script has — about a quantity it did not compute. This is the
    regression test for the path that shipped without one.
    """
    rendered = format_report(_report())
    assert "NOT MEASURED" in rendered
    assert "OK — real signal dominates" not in rendered
    assert "MEASURE_DTYPE_FLOOR" in rendered, "must say how to get the missing measurement"


def test_the_report_still_certifies_a_genuinely_measured_run():
    rendered = format_report(_report(dtype_floor=0.0056, signal_to_floor=17.2))
    assert "OK — real signal dominates" in rendered
    assert "NOT MEASURED" not in rendered


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(1.0, "STOP"), (5.0, "MARGINAL"), (20.0, "OK")],
)
def test_the_measured_verdict_bands_are_unchanged(ratio, expected):
    assert expected in format_report(_report(dtype_floor=0.01, signal_to_floor=ratio))


# --------------------------------------------------------------------------------------
# Observation allocation across a mixture.
# --------------------------------------------------------------------------------------


def test_a_short_dataset_is_made_up_by_the_others():
    """`count` is the study's sample size, so a shortfall silently weakens every rho.

    A mixture member smaller than its even slice must not lower the total — the datasets
    after it have frames to spare.
    """
    assert sum(_allocate_draws([5, 100], 24)) == 24
    assert sum(_allocate_draws([100, 5], 24)) == 24
    assert sum(_allocate_draws([1, 1, 1, 500], 24)) == 24


def test_asking_for_fewer_observations_than_datasets_does_not_round_up():
    """Each observation costs K denoise passes, so overshooting the request is not free."""
    quotas = _allocate_draws([100] * 5, 2)
    assert sum(quotas) == 2
    assert all(q >= 0 for q in quotas)


def test_allocation_is_capped_by_what_the_mixture_actually_holds():
    assert sum(_allocate_draws([3, 4], 24)) == 7
    assert _allocate_draws([3, 4], 24) == [3, 4]


def test_allocation_spreads_across_members_rather_than_draining_the_first():
    """Drawing everything from one dataset is the narrowed-range failure this replaced."""
    quotas = _allocate_draws([1000, 1000, 1000], 24)
    assert sum(quotas) == 24
    assert min(quotas) > 0, "every dataset must contribute"


def test_the_study_restores_the_policy_prefix():
    """It runs at the full schedule to fill the trace; leaving it there would silently change
    the score every later call produces."""
    policy = _ScriptedFlowPolicy(late_slope=0.0)
    policy.accel_prefix = 4
    measure_prefix_quality(policy, _observations([0.01, 0.02, 0.03]), num_resamples=4)
    assert policy.accel_prefix == 4
