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

import logging
from types import SimpleNamespace

import accelerate
import pytest
import torch

from opentau.scripts.train import (
    _bucket_per_sample,
    _commit_wandb_step,
    _find_unused_params_from_env,
    _mixture_weighted_aggregate,
    _sync_deepspeed_gradient_accumulation_steps,
)
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


def _make_accelerator(distributed_type, ds_grad_acc, is_main_process=True):
    plugin = SimpleNamespace(
        hf_ds_config=SimpleNamespace(config={"gradient_accumulation_steps": ds_grad_acc}),
        gradient_accumulation_steps=ds_grad_acc,
    )
    return SimpleNamespace(
        distributed_type=distributed_type,
        is_main_process=is_main_process,
        deepspeed_plugin=plugin,
    )


def _make_cfg(grad_acc):
    return SimpleNamespace(gradient_accumulation_steps=grad_acc)


def test_non_deepspeed_is_noop(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.MULTI_GPU, ds_grad_acc=2)
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 2
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 2
    assert not caplog.records


def test_deepspeed_matching_value_no_warning(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2)
    cfg = _make_cfg(grad_acc=2)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 2
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_deepspeed_mismatch_overrides_and_warns_on_main(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2, is_main_process=True)
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 4
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 4
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2" in message
    assert "4" in message


def test_deepspeed_mismatch_non_main_overrides_without_warning(caplog):
    accelerator = _make_accelerator(
        accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2, is_main_process=False
    )
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 4
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 4
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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

    def test_empty_trackers_returns_empty_dict(self):
        # Without any trackers the function has no way to know which metric
        # keys the policy emits, so it returns an empty dict rather than a
        # zeroed-out hardcoded key set. Caller (``train.py``) never invokes
        # this with empty trackers in practice; the empty-input path is here
        # only as a documented degenerate case.
        assert _mixture_weighted_aggregate({}, {}) == {}

    def test_all_zero_weights_returns_zeros(self):
        trackers = {"a": _make_tracker(loss=1.0), "b": _make_tracker(loss=2.0)}
        weights = {"a": 0.0, "b": 0.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        # Avoids div-by-zero; behaviour matches "no signal to average".
        assert agg["loss"] == 0.0

    def test_aggregates_only_keys_present_on_trackers(self):
        # Mirrors the production case where the policy returns only MSE+CE
        # (every VLA policy except the value head). ``l1_loss``/``accuracy``
        # meters are absent from the tracker, so the aggregate must not
        # invent them.
        def _make_partial_tracker(loss: float, mse: float, ce: float) -> MetricsTracker:
            tracker = MetricsTracker(
                batch_size=8,
                metrics={
                    "loss": AverageMeter("val_total_loss", ":.3f"),
                    "mse_loss": AverageMeter("val_mse_loss", ":.3f"),
                    "ce_loss": AverageMeter("val_ce_loss", ":.3f"),
                },
            )
            tracker.loss = loss
            tracker.mse_loss = mse
            tracker.ce_loss = ce
            return tracker

        trackers = {
            "a": _make_partial_tracker(loss=1.0, mse=2.0, ce=3.0),
            "b": _make_partial_tracker(loss=5.0, mse=6.0, ce=7.0),
        }
        weights = {"a": 1.0, "b": 3.0}
        agg = _mixture_weighted_aggregate(trackers, weights)
        assert set(agg.keys()) == {"loss", "mse_loss", "ce_loss"}
        assert agg["loss"] == pytest.approx(0.25 * 1.0 + 0.75 * 5.0)
        assert agg["mse_loss"] == pytest.approx(0.25 * 2.0 + 0.75 * 6.0)
        assert agg["ce_loss"] == pytest.approx(0.25 * 3.0 + 0.75 * 7.0)


class TestBucketPerSample:
    """``_bucket_per_sample`` disaggregates a *mixed* batch by integer group key —
    the core of the per-(dataset, control_mode) validation breakdown (issue #373).
    Grouping on the dropout-immune ``dataset_index`` is what makes a mixed batch
    decomposable; this is pure CPU arithmetic with no accelerator."""

    def test_disaggregates_mixed_batch_without_cross_contamination(self):
        group = torch.tensor([0, 1, 0, 1, 2])
        mse_sum = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mse_count = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        ce_sum = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        ce_count = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        buckets = _bucket_per_sample(group, mse_sum, mse_count, ce_sum, ce_count)
        assert set(buckets) == {0, 1, 2}
        # group 0 = samples {0, 2}; groups 1/2 must not leak in.
        assert buckets[0]["mse_sum"] == pytest.approx(4.0)
        assert buckets[0]["mse_count"] == pytest.approx(2.0)
        assert buckets[0]["ce_sum"] == pytest.approx(40.0)
        assert buckets[0]["ce_count"] == pytest.approx(4.0)
        assert buckets[1]["mse_sum"] == pytest.approx(6.0)
        assert buckets[2]["mse_sum"] == pytest.approx(5.0)

    def test_group_mean_is_sum_over_count_not_mean_of_means(self):
        # Heterogeneous counts: the regrouped mean must be Σsum/Σcount, so a
        # 3-slot sample outweighs a 1-slot sample (mean-of-means would not).
        group = torch.tensor([0, 0])
        s = torch.tensor([2.0, 9.0])
        c = torch.tensor([1.0, 3.0])
        buckets = _bucket_per_sample(group, s, c, s, c)
        mean = buckets[0]["mse_sum"] / buckets[0]["mse_count"]
        assert mean == pytest.approx(11.0 / 4.0)

    def test_zero_count_group_carries_zeros_for_caller_guard(self):
        # pi0 emits zero-count CE; the bucket keeps 0/0 and the caller's +1e-8
        # turns it into a 0 mean (no NaN).
        group = torch.tensor([0, 1])
        mse = torch.tensor([1.0, 2.0])
        cnt = torch.tensor([1.0, 1.0])
        zeros = torch.tensor([0.0, 0.0])
        buckets = _bucket_per_sample(group, mse, cnt, zeros, zeros)
        assert buckets[0]["ce_sum"] == 0.0
        assert buckets[0]["ce_count"] == 0.0
        assert buckets[0]["ce_sum"] / (buckets[0]["ce_count"] + 1e-8) == pytest.approx(0.0)


class TestCommitWandbStep:
    """``_commit_wandb_step`` issues exactly one empty, commit-tagged log."""

    def test_emits_empty_commit_log_at_step(self):
        # The fix relies on three properties of the emitted call: an empty
        # ``values`` dict (flush the accumulated row, add no keys), the explicit
        # ``step`` (seal that row, not some other), and ``commit=True`` routed
        # only under the ``"wandb"`` key (tracker-safe; other trackers never see
        # it). Assert the exact call shape to pin all three.
        calls = []

        def _log(values, step=None, log_kwargs=None):
            calls.append((values, step, log_kwargs))

        accelerator = SimpleNamespace(log=_log)
        _commit_wandb_step(accelerator, 1234)

        assert calls == [({}, 1234, {"wandb": {"commit": True}})]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
