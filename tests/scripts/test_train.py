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

"""Unit tests for pure helpers in train.

The actual ``train()`` entrypoint wraps a full training loop and is
exercised end-to-end by ``.github/workflows/regression_test.yml``;
this file only covers the small parsing helpers that benefit from
isolated coverage.
"""

from __future__ import annotations

import pytest

from opentau.scripts.train import _find_unused_params_from_env, _mixture_weighted_aggregate
from opentau.utils.logging_utils import AverageMeter, MetricsTracker


class TestFindUnusedParamsFromEnv:
    """``FIND_UNUSED_PARAMS`` env var → bool."""

    def test_default_is_true_when_unset(self, monkeypatch):
        monkeypatch.delenv("FIND_UNUSED_PARAMS", raising=False)
        assert _find_unused_params_from_env() is True

    def test_explicit_true(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "true")
        assert _find_unused_params_from_env() is True

    def test_explicit_false(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "false")
        assert _find_unused_params_from_env() is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "TRUE")
        assert _find_unused_params_from_env() is True

        monkeypatch.setenv("FIND_UNUSED_PARAMS", "False")
        assert _find_unused_params_from_env() is False

        monkeypatch.setenv("FIND_UNUSED_PARAMS", "FALSE")
        assert _find_unused_params_from_env() is False

    def test_empty_string_is_treated_as_false(self, monkeypatch):
        # Empty string != "true" so we default to False. Matches the
        # strict "anything-non-true is false" semantics of the inline
        # expression we extracted; flipping to False on empty is the
        # safe choice because a misspelled env var should not silently
        # preserve the old default behavior.
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "")
        assert _find_unused_params_from_env() is False

    def test_unknown_values_parse_as_false(self, monkeypatch):
        # "yes" / "1" / anything other than a case-insensitive "true"
        # counts as False. Keeps the parser strict.
        for value in ("yes", "1", "enabled", "Y", "on"):
            monkeypatch.setenv("FIND_UNUSED_PARAMS", value)
            assert _find_unused_params_from_env() is False, f"Expected {value!r} to parse as False, got True"


def _make_tracker(
    loss: float, mse: float = 0.0, ce: float = 0.0, l1: float = 0.0, acc: float = 0.0
) -> MetricsTracker:
    """Build a ``MetricsTracker`` with one update per metric.

    A single assignment per attribute means each ``AverageMeter`` ends up with
    ``avg == val``, which is exactly what we need to exercise the weighted
    aggregation in isolation.
    """
    tracker = MetricsTracker(
        batch_size=8,
        metrics={
            "loss": AverageMeter("val_total_loss", ":.3f"),
            "mse_loss": AverageMeter("val_mse_loss", ":.3f"),
            "ce_loss": AverageMeter("val_ce_loss", ":.3f"),
            "l1_loss": AverageMeter("val_l1_loss", ":.3f"),
            "accuracy": AverageMeter("val_accuracy", ":.3f"),
        },
    )
    tracker.loss = loss
    tracker.mse_loss = mse
    tracker.ce_loss = ce
    tracker.l1_loss = l1
    tracker.accuracy = acc
    return tracker


class TestMixtureWeightedAggregate:
    """``_mixture_weighted_aggregate`` collapses per-dataset trackers using mixture weights."""

    def test_equal_weights_is_simple_mean(self):
        trackers = {"a": _make_tracker(loss=1.0), "b": _make_tracker(loss=3.0)}
        weights = {"a": 1.0, "b": 1.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        assert agg["loss"] == pytest.approx(2.0)

    def test_unequal_weights_renormalize(self):
        # weights [3, 1] -> 0.75 * 1.0 + 0.25 * 5.0 = 2.0
        trackers = {"a": _make_tracker(loss=1.0), "b": _make_tracker(loss=5.0)}
        weights = {"a": 3.0, "b": 1.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        assert agg["loss"] == pytest.approx(2.0)

    def test_renormalizes_over_present_keys_only(self):
        # ``name_to_weight`` includes a name that is missing from
        # ``per_dataset_trackers`` (e.g. an empty val subset). The aggregate
        # should ignore it and renormalize over the present keys.
        trackers = {"a": _make_tracker(loss=1.0), "b": _make_tracker(loss=3.0)}
        weights = {"a": 1.0, "b": 1.0, "c_empty": 100.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        assert agg["loss"] == pytest.approx(2.0)

    def test_aggregates_all_metric_keys(self):
        trackers = {
            "a": _make_tracker(loss=1.0, mse=2.0, ce=3.0, l1=4.0, acc=0.1),
            "b": _make_tracker(loss=5.0, mse=6.0, ce=7.0, l1=8.0, acc=0.5),
        }
        weights = {"a": 1.0, "b": 3.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        # 0.25 * a + 0.75 * b
        assert agg["loss"] == pytest.approx(0.25 * 1.0 + 0.75 * 5.0)
        assert agg["mse_loss"] == pytest.approx(0.25 * 2.0 + 0.75 * 6.0)
        assert agg["ce_loss"] == pytest.approx(0.25 * 3.0 + 0.75 * 7.0)
        assert agg["l1_loss"] == pytest.approx(0.25 * 4.0 + 0.75 * 8.0)
        assert agg["accuracy"] == pytest.approx(0.25 * 0.1 + 0.75 * 0.5)

    def test_empty_trackers_returns_zeros(self):
        agg = _mixture_weighted_aggregate({}, {})
        assert agg == {"loss": 0.0, "mse_loss": 0.0, "ce_loss": 0.0, "l1_loss": 0.0, "accuracy": 0.0}

    def test_all_zero_weights_returns_zeros(self):
        trackers = {"a": _make_tracker(loss=1.0), "b": _make_tracker(loss=2.0)}
        weights = {"a": 0.0, "b": 0.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        # Avoids div-by-zero; behaviour matches "no signal to average".
        assert agg["loss"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
