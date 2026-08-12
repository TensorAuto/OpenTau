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

"""Tests for the ``accel`` calibration driver.

The failure mode this file exists to pin is a **guard that deletes itself**. A calibration
carries an :class:`~opentau.policies.accel.AccelProvenance` label, and
:func:`~opentau.utils.accel_detector.detect` refuses a stream whose provenance disagrees
with it. But an *unlabelled* calibration skips that check entirely — so any bug that
wrongly drops the label converts a hard refusal into a silent pass, which is strictly worse
than the mismatch it was meant to catch.

Comparing the whole provenance dict does exactly that: ``dataset_index`` and
``num_scored_dims`` are per-sample and vary with batch composition, so two tasks agreeing on
every distribution-shifting field get declared "disagreeing" and the label is dropped.
"""

import json

import numpy as np
import pytest

from opentau.scripts.calibrate_accel import (
    _resolve_provenance,
    fit,
    load_eval_info,
    split_by_outcome,
)

COMPARABLE = {
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


def _provenance(**overrides):
    payload = dict(COMPARABLE)
    payload.setdefault("dataset_index", (0,))
    payload.setdefault("num_scored_dims", (7,))
    payload.update(overrides)
    return payload


def _task(*, accels, successes, provenance=None, task_id=0):
    return {
        "task_group": "libero",
        "task_id": task_id,
        "metrics": {
            "accels": accels,
            "successes": successes,
            "accel_provenance": _provenance() if provenance is None else provenance,
        },
    }


def _quiet(n=12, length=20, seed=0):
    rng = np.random.default_rng(seed)
    return [(0.4 + 0.02 * rng.standard_normal(length)).tolist() for _ in range(n)]


# --------------------------------------------------------------------------------------
# Provenance resolution — the fail-open path.
# --------------------------------------------------------------------------------------


def test_tasks_differing_only_in_per_sample_fields_still_share_a_label():
    """Batch composition must not look like a configuration disagreement.

    ``dataset_index`` and ``num_scored_dims`` are per-sample; two tasks routed through
    different norm heads, or with different numbers of episodes, legitimately differ there
    while agreeing on everything that shifts the score's distribution. Treating that as a
    disagreement drops the label and disarms the apply-time comparability check.
    """
    tasks = [
        _task(
            accels=[[0.4]],
            successes=[True],
            provenance=_provenance(dataset_index=(0, 0), num_scored_dims=(7, 7)),
        ),
        _task(
            accels=[[0.4]],
            successes=[True],
            provenance=_provenance(dataset_index=(1,), num_scored_dims=(6,)),
            task_id=1,
        ),
    ]
    resolved = _resolve_provenance(tasks)
    assert resolved is not None, "per-sample differences must not drop the label"
    assert resolved.num_steps == 10 and resolved.action_norm_mode == "MEAN_STD"


def test_the_shared_label_clears_the_per_sample_fields():
    """Carrying one arbitrary task's per-sample values on a pooled calibration is a fiction.

    They are cleared rather than kept, and `assert_comparable` ignores them anyway, so the
    cleared tuples are inert at apply time.
    """
    tasks = [
        _task(accels=[[0.4]], successes=[True], provenance=_provenance(dataset_index=(3,))),
        _task(
            accels=[[0.4]],
            successes=[True],
            provenance=_provenance(dataset_index=(9,)),
            task_id=1,
        ),
    ]
    resolved = _resolve_provenance(tasks)
    assert resolved.dataset_index == ()
    assert resolved.num_scored_dims == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_steps", 5),
        ("prefix", 4),
        ("action_norm_mode", "QUANTILE"),
        ("velocity_dtype", "float32"),
        ("has_delta_action_map", True),
        ("max_delay", 4),
        ("policy_type", "pi07"),
    ],
)
def test_a_genuine_disagreement_still_drops_the_label(field, value):
    """The relaxation must not go too far — a real configuration split must stay unlabelled.

    Without this, the fix for the per-sample false positive could silently degrade into
    "never disagree", which would restore the fail-open behaviour by another route.
    """
    tasks = [
        _task(accels=[[0.4]], successes=[True]),
        _task(accels=[[0.4]], successes=[True], provenance=_provenance(**{field: value}), task_id=1),
    ]
    assert _resolve_provenance(tasks) is None


def test_no_recorded_provenance_leaves_the_calibration_unlabelled():
    assert _resolve_provenance([]) is None


# --------------------------------------------------------------------------------------
# Outcome pairing — `accels[i]` must belong to `successes[i]`.
# --------------------------------------------------------------------------------------


def test_streams_are_split_by_their_own_episode_outcome():
    good, bad, successes = split_by_outcome(
        _task(accels=[[0.1], [0.9], [0.2]], successes=[True, False, True])
    )
    assert good == [[0.1], [0.2]]
    assert bad == [[0.9]]
    assert successes == [True, False, True]


def test_a_length_mismatch_drops_the_task_rather_than_pairing_it():
    """One stream short would shift every label by one — silently fitting on the wrong set."""
    good, bad, successes = split_by_outcome(_task(accels=[[0.1]], successes=[True, False]))
    assert (good, bad, successes) == ([], [], [])


def test_a_task_without_accel_is_simply_absent():
    assert split_by_outcome(_task(accels=[], successes=[True])) == ([], [], [])


# --------------------------------------------------------------------------------------
# Fitting.
# --------------------------------------------------------------------------------------


def test_fit_returns_none_when_there_are_too_few_successes_to_certify_alpha():
    """A threshold that cannot certify the requested rate must not be produced at all."""
    tasks = [_task(accels=_quiet(n=3), successes=[True] * 3)]
    assert fit(tasks, alpha=0.1, slack_c=0.25, label="pooled") is None


def test_fit_produces_a_labelled_calibration_from_enough_successes():
    tasks = [_task(accels=_quiet(n=12), successes=[True] * 12)]
    result = fit(tasks, alpha=0.2, slack_c=0.25, label="pooled")
    assert result is not None
    assert result["num_successful"] == 12
    assert result["calibration"].provenance is not None, "a single-config fit must stay labelled"
    assert result["calibration"].eta >= 0.0


def test_fit_reports_detection_against_failing_episodes():
    """A held-in evaluation is optimistic but must still separate the two classes."""
    good = _quiet(n=12, length=20)
    bad = [[0.4] * 5 + [1.6] * 15 for _ in range(4)]
    tasks = [_task(accels=good + bad, successes=[True] * 12 + [False] * 4)]
    result = fit(tasks, alpha=0.2, slack_c=0.25, label="pooled")
    assert result["metrics"]["true_positive_rate"] == 1.0
    assert result["metrics"]["false_alarm_rate"] <= 0.2


# --------------------------------------------------------------------------------------
# Input loading.
# --------------------------------------------------------------------------------------


def test_load_eval_info_accepts_a_file_or_its_directory(tmp_path):
    payload = {"per_task": [_task(accels=[[0.4]], successes=[True])]}
    path = tmp_path / "eval_info.json"
    path.write_text(json.dumps(payload))
    assert len(load_eval_info(path)) == 1
    assert len(load_eval_info(tmp_path)) == 1


def test_load_eval_info_rejects_a_missing_or_unrelated_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_info(tmp_path / "nope.json")

    other = tmp_path / "eval_info.json"
    other.write_text(json.dumps({"overall": {}}))
    with pytest.raises(ValueError, match="per_task"):
        load_eval_info(other)
