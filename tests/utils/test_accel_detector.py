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

"""Tests for the offline CUSUM + split-conformal detector over an ``accel`` stream."""

import json
import math

import numpy as np
import pytest

from opentau.policies.accel import AccelProvenance
from opentau.utils.accel_detector import (
    CALIBRATION_FILENAME,
    CusumCalibration,
    calibrate,
    conformal_rank,
    cusum_stream,
    detect,
    episode_peak,
    evaluate,
)


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
# The CUSUM recursion.
# --------------------------------------------------------------------------------------


def test_cusum_is_floored_at_zero():
    """A quiet stream must not bank negative credit — otherwise a long calm stretch would
    buy immunity from a later genuine excursion."""
    stream = cusum_stream([0.0] * 10, mu0=1.0, k=0.0)
    assert stream.tolist() == [0.0] * 10


def test_cusum_accumulates_persistent_excess():
    stream = cusum_stream([2.0, 2.0, 2.0], mu0=1.0, k=0.0)
    assert stream.tolist() == [1.0, 2.0, 3.0]


def test_slack_absorbs_transient_spikes_but_not_a_sustained_shift():
    """The paper's slack ``k`` is what makes a single noisy chunk harmless while a
    persistent elevation still integrates to an alarm."""
    transient = cusum_stream([0.0, 0.0, 0.4, 0.0, 0.0], mu0=0.0, k=0.5)
    assert transient.max() == 0.0

    sustained = cusum_stream([0.6] * 10, mu0=0.0, k=0.5)
    assert sustained[-1] == pytest.approx(1.0)


def test_non_finite_scores_carry_the_statistic_forward():
    """The meter emits NaN when a chunk has no valid measurement. That is an absence of
    evidence, not evidence of drift, so it must neither bump nor reset the statistic."""
    with_nan = cusum_stream([2.0, float("nan"), 2.0], mu0=1.0, k=0.0)
    assert with_nan.tolist() == [1.0, 1.0, 2.0]


def test_episode_peak_reduces_any_length_to_one_scalar():
    assert episode_peak([], mu0=0.0, k=0.0) == 0.0
    assert episode_peak([1.0, 3.0, 0.0], mu0=1.0, k=0.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# Split-conformal rank selection.
# --------------------------------------------------------------------------------------


def test_conformal_rank_matches_the_papers_operating_point():
    """M=50, alpha=0.1 -> the 46th smallest of 50 calibration peaks."""
    assert conformal_rank(50, 0.1) == 46


def test_conformal_rank_refuses_an_undersized_calibration_set():
    """With too few episodes no finite threshold can certify the rate; silently clamping to
    the max would report a guarantee that does not hold."""
    with pytest.raises(ValueError, match="at least"):
        conformal_rank(5, 0.01)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_rank_rejects_out_of_range_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        conformal_rank(50, alpha)


# --------------------------------------------------------------------------------------
# Calibration.
# --------------------------------------------------------------------------------------


def _quiet_episodes(n=50, length=30, seed=0):
    rng = np.random.default_rng(seed)
    return [(0.4 + 0.02 * rng.standard_normal(length)).tolist() for _ in range(n)]


def test_calibration_holds_its_nominal_false_alarm_rate_in_sample():
    """The conformal construction guarantees at most ``ceil((M+1)(1-alpha))``-th order
    statistic coverage, so at most ``alpha`` of the calibration episodes may exceed it."""
    episodes = _quiet_episodes()
    cal = calibrate(episodes, alpha=0.1)
    alarms = sum(detect(ep, cal).alarmed for ep in episodes)
    assert alarms / len(episodes) <= 0.1


def test_calibration_records_its_inputs():
    cal = calibrate(_quiet_episodes(n=50, length=20), alpha=0.1, provenance=_prov())
    assert cal.num_calibration_episodes == 50
    assert cal.num_calibration_chunks == 1000
    assert cal.k == pytest.approx(cal.slack_c * cal.sigma)
    assert cal.provenance == _prov()


def test_calibration_rejects_an_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        calibrate([])


def test_calibration_rejects_an_all_nan_set():
    """All-NaN means the meter never had a valid measurement — a misconfiguration that
    must surface loudly rather than produce a threshold of NaN."""
    with pytest.raises(ValueError, match="no finite scores"):
        calibrate([[float("nan")] * 5 for _ in range(50)])


# --------------------------------------------------------------------------------------
# Detection.
# --------------------------------------------------------------------------------------


def test_a_sustained_excursion_alarms_with_lead():
    """The failure signature from the paper's Figure 4: a quiet stretch, then a step change
    that persists to the episode's end. The alarm must land inside the excursion, with lead
    counted in remaining chunks."""
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    failing = [0.4] * 20 + [1.5] * 20
    result = detect(failing, cal)
    assert result.alarmed
    assert 20 <= result.alarm_index < 30, "alarm should fire shortly after the excursion starts"
    assert result.lead == len(failing) - 1 - result.alarm_index
    assert result.lead > 0


def test_a_quiet_episode_does_not_alarm():
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    result = detect([0.4] * 40, cal)
    assert not result.alarmed
    assert result.alarm_index is None and result.lead is None


def test_an_isolated_spike_is_absorbed_but_the_same_level_sustained_alarms():
    """The property that distinguishes a CUSUM from a per-chunk threshold, and the paper's
    successful-rollout case (Figure 8): one excursion above the reference level is absorbed,
    while *the same level* held for several chunks integrates past the threshold.

    Both streams contain the identical maximum score, so a detector that merely compared
    each chunk against a level would be unable to separate them.
    """
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    level = 0.5

    assert not detect([0.4] * 15 + [level] + [0.4] * 15, cal).alarmed
    assert detect([0.4] * 15 + [level] * 10 + [0.4] * 5, cal).alarmed


def test_detection_refuses_a_provenance_mismatch():
    """A threshold fitted at T=10 says nothing about a stream produced at T=5."""
    cal = calibrate(_quiet_episodes(), alpha=0.1, provenance=_prov())
    with pytest.raises(ValueError, match="not applicable"):
        detect([0.4] * 10, cal, provenance=_prov(num_steps=5, prefix=4))


def test_detection_allows_a_matching_provenance():
    cal = calibrate(_quiet_episodes(), alpha=0.1, provenance=_prov())
    detect([0.4] * 10, cal, provenance=_prov(dataset_index=(2,)))


def test_detection_skips_the_check_when_either_side_is_unlabeled():
    cal = calibrate(_quiet_episodes(), alpha=0.1, provenance=None)
    detect([0.4] * 10, cal, provenance=_prov())


# --------------------------------------------------------------------------------------
# Evaluation metrics.
# --------------------------------------------------------------------------------------


def test_evaluate_reports_detection_rate_false_alarm_rate_and_lead():
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    episodes = [[0.4] * 40] * 10 + [[0.4] * 10 + [1.5] * 30] * 10
    successes = [True] * 10 + [False] * 10
    metrics = evaluate(episodes, successes, cal)
    assert metrics["true_positive_rate"] == 1.0
    assert metrics["false_alarm_rate"] == 0.0
    assert metrics["n_failed"] == 10.0 and metrics["n_successful"] == 10.0
    assert metrics["median_lead"] > 0


def test_evaluate_rejects_mismatched_lengths():
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    with pytest.raises(ValueError, match="differ in length"):
        evaluate([[0.4]], [True, False], cal)


def test_evaluate_returns_nan_for_an_absent_class():
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    metrics = evaluate([[0.4] * 10], [True], cal)
    assert math.isnan(metrics["true_positive_rate"])
    assert metrics["false_alarm_rate"] == 0.0


# --------------------------------------------------------------------------------------
# Persistence.
# --------------------------------------------------------------------------------------


def test_calibration_round_trips_through_json(tmp_path):
    cal = calibrate(_quiet_episodes(), alpha=0.1, provenance=_prov())
    path = cal.save(tmp_path)
    assert path.name == CALIBRATION_FILENAME
    assert json.loads(path.read_text())["provenance"]["policy_type"] == "pi05"

    loaded = CusumCalibration.load(tmp_path)
    assert loaded == cal
    assert detect([0.4] * 10, loaded, provenance=_prov()).peak == pytest.approx(detect([0.4] * 10, cal).peak)


def test_calibration_saves_to_an_explicit_filename(tmp_path):
    cal = calibrate(_quiet_episodes(), alpha=0.1)
    path = cal.save(tmp_path / "nested" / "mine.json")
    assert path.name == "mine.json"
    assert CusumCalibration.load(path) == cal
