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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import accelerate
import pytest
import torch
import wandb

from opentau.scripts.train import (
    _bucket_per_sample,
    _commit_wandb_step,
    _deepspeed_zero_stage,
    _eval_with_fresh_envs,
    _find_unused_params_from_env,
    _init_wandb_trackers,
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


def _make_stage_accelerator(distributed_type, zero_optimization=None):
    """Accelerator stub exposing the deepspeed plugin config ``_deepspeed_zero_stage`` reads.

    ``zero_optimization`` is the dict placed under the DeepSpeed config's
    ``"zero_optimization"`` key; pass ``None`` to model a config that never
    declared it.
    """
    config = {}
    if zero_optimization is not None:
        config["zero_optimization"] = zero_optimization
    plugin = SimpleNamespace(hf_ds_config=SimpleNamespace(config=config))
    return SimpleNamespace(distributed_type=distributed_type, deepspeed_plugin=plugin)


class TestDeepspeedZeroStage:
    """``_deepspeed_zero_stage`` extracts the active ZeRO stage; 0 for non-DeepSpeed."""

    @pytest.mark.parametrize(
        "distributed_type",
        [
            accelerate.DistributedType.NO,
            accelerate.DistributedType.MULTI_GPU,
            accelerate.DistributedType.FSDP,
        ],
    )
    def test_non_deepspeed_is_zero(self, distributed_type):
        # Non-DeepSpeed backends must short-circuit to 0 *without* touching the
        # deepspeed plugin (which is None / unconfigured here) -- that is what
        # lets the ZeRO-3 guards in train() test ``>= 3`` directly.
        acc = SimpleNamespace(distributed_type=distributed_type, deepspeed_plugin=None)
        assert _deepspeed_zero_stage(acc) == 0

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_deepspeed_returns_configured_stage(self, stage):
        acc = _make_stage_accelerator(
            accelerate.DistributedType.DEEPSPEED, zero_optimization={"stage": stage}
        )
        assert _deepspeed_zero_stage(acc) == stage

    def test_deepspeed_without_zero_optimization_is_zero(self):
        # A DeepSpeed config that never declared zero_optimization (or declared
        # it without a stage) means no parameter sharding -> treat as stage 0 so
        # the ZeRO-3-only guards stay off.
        acc_missing = _make_stage_accelerator(accelerate.DistributedType.DEEPSPEED, zero_optimization=None)
        assert _deepspeed_zero_stage(acc_missing) == 0

        acc_no_stage = _make_stage_accelerator(accelerate.DistributedType.DEEPSPEED, zero_optimization={})
        assert _deepspeed_zero_stage(acc_no_stage) == 0


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


class TestInitWandbTrackers:
    """``_init_wandb_trackers`` forks on resume and degrades safely on a fork failure."""

    @staticmethod
    def _cfg(**wandb_kwargs):
        from opentau.configs.default import WandBConfig

        return SimpleNamespace(wandb=WandBConfig(enable=False, project="proj", **wandb_kwargs))

    def test_fork_passes_fork_from_in_single_init(self):
        """A successful resume-fork issues exactly one init carrying ``fork_from``
        and returns the new tracker."""
        cfg = self._cfg(run_id="parent", notes="user notes")
        init = Mock()
        accelerator = SimpleNamespace(
            init_trackers=init, get_tracker=lambda name, unwrap: SimpleNamespace(id="forked")
        )

        run = _init_wandb_trackers(accelerator, cfg, {"cfg": 1}, step=50)

        assert run.id == "forked"
        assert init.call_count == 1
        assert init.call_args.kwargs["init_kwargs"]["wandb"]["fork_from"] == "parent?_step=50"

    def test_fork_failure_falls_back_to_in_place_resume(self):
        """A ``wandb.Error`` on the fork init drops ``fork_from``, restores the user's
        notes (no stale "Forked from ..." annotation), and retries as an in-place
        resume (``id`` + ``resume='allow'``)."""
        cfg = self._cfg(run_id="parent", notes="user notes")
        init = Mock(side_effect=[wandb.errors.CommError("run not found"), None])
        accelerator = SimpleNamespace(
            init_trackers=init, get_tracker=lambda name, unwrap: SimpleNamespace(id="resumed")
        )

        run = _init_wandb_trackers(accelerator, cfg, {"cfg": 1}, step=50)

        assert run.id == "resumed"
        assert init.call_count == 2
        first = init.call_args_list[0].kwargs["init_kwargs"]["wandb"]
        assert first["fork_from"] == "parent?_step=50"
        retry = init.call_args_list[1].kwargs["init_kwargs"]["wandb"]
        assert "fork_from" not in retry
        assert retry["id"] == "parent" and retry["resume"] == "allow"
        assert retry["notes"] == "user notes"

    def test_non_fork_wandb_error_propagates(self):
        """A ``wandb.Error`` on a non-fork init (no ``run_id`` -> no ``fork_from``) is a
        real failure and must not be swallowed by the fork fallback."""
        cfg = self._cfg()  # no run_id -> to_wandb_kwargs emits no fork_from
        init = Mock(side_effect=wandb.errors.CommError("boom"))
        accelerator = SimpleNamespace(init_trackers=init, get_tracker=lambda name, unwrap: None)

        with pytest.raises(wandb.errors.CommError):
            _init_wandb_trackers(accelerator, cfg, {"cfg": 1}, step=None)


class TestEvalWithFreshEnvs:
    """``_eval_with_fresh_envs`` builds eval envs, runs eval, and ALWAYS tears them down.

    Per-eval teardown frees the sim renderer's non-PyTorch GPU memory before
    training resumes (otherwise it stacks with the regrown training footprint
    and OOMs the next backward). Even when the eval itself raises, the ``finally``
    must (1) close the envs and (2) ``reset()`` the policy so its per-eval
    observation buffers (sized to ``eval.batch_size``) do not persist into the
    next training step (the ``retained_alloc`` leak).
    """

    @staticmethod
    def _cfg():
        return SimpleNamespace(
            env=SimpleNamespace(max_parallel_tasks=1),
            eval=SimpleNamespace(
                batch_size=2,
                use_async_envs=True,
                n_episodes=2,
                max_episodes_rendered=0,
                grid_size=None,
            ),
            policy=SimpleNamespace(use_amp=False),
            output_dir=Path("/tmp/opentau_eval_test"),
            seed=0,
        )

    @staticmethod
    def _acc():
        # ``unwrap_model`` returns the underlying module so the helper can call
        # ``reset()`` on it (mirrors ``accelerate.Accelerator.unwrap_model``).
        return SimpleNamespace(device=SimpleNamespace(type="cpu"), unwrap_model=lambda p: p)

    def test_builds_and_closes_each_call(self, monkeypatch):
        envs1 = {"robocasa": {0: object()}}
        envs2 = {"robocasa": {0: object()}}
        make = Mock(side_effect=[envs1, envs2])
        close = Mock()
        ep_all = Mock(return_value={"overall": {}})
        monkeypatch.setattr("opentau.scripts.train.make_envs", make)
        monkeypatch.setattr("opentau.scripts.train.close_envs", close)
        monkeypatch.setattr("opentau.scripts.train.eval_policy_all", ep_all)

        acc = self._acc()
        cfg = self._cfg()
        out1 = _eval_with_fresh_envs(cfg, Mock(), acc, None, "000010")
        out2 = _eval_with_fresh_envs(cfg, Mock(), acc, None, "000020")

        assert make.call_count == 2
        assert close.call_count == 2
        # The exact env dict built each call is the one handed to close (lifecycle pairing).
        assert [c.args[0] for c in close.call_args_list] == [envs1, envs2]
        assert out1 is ep_all.return_value
        assert out2 is ep_all.return_value

    def test_closes_even_when_eval_raises(self, monkeypatch):
        envs = {"robocasa": {0: object()}}
        make = Mock(return_value=envs)
        close = Mock()
        boom = Mock(side_effect=RuntimeError("eval blew up"))
        monkeypatch.setattr("opentau.scripts.train.make_envs", make)
        monkeypatch.setattr("opentau.scripts.train.close_envs", close)
        monkeypatch.setattr("opentau.scripts.train.eval_policy_all", boom)

        with pytest.raises(RuntimeError, match="eval blew up"):
            _eval_with_fresh_envs(self._cfg(), Mock(), self._acc(), None, "000010")
        close.assert_called_once_with(envs)

    def test_resets_policy_after_successful_eval(self, monkeypatch):
        # The policy accumulates eval-batch-sized observation history in its internal
        # buffers; reset() must run after the eval so referenced GPU memory (which
        # empty_cache cannot reclaim) does not persist into the resumed training step.
        make = Mock(return_value={"robocasa": {0: object()}})
        ep_all = Mock(return_value={"overall": {}})
        monkeypatch.setattr("opentau.scripts.train.make_envs", make)
        monkeypatch.setattr("opentau.scripts.train.close_envs", Mock())
        monkeypatch.setattr("opentau.scripts.train.eval_policy_all", ep_all)

        policy = Mock()
        _eval_with_fresh_envs(self._cfg(), policy, self._acc(), None, "000010")
        policy.reset.assert_called_once_with()

    def test_resets_policy_even_when_eval_raises(self, monkeypatch):
        # reset() shares the ``finally`` with ``close_envs``: a failed eval must not
        # leak the policy's eval-batch observation buffers into the next training step.
        make = Mock(return_value={"robocasa": {0: object()}})
        boom = Mock(side_effect=RuntimeError("eval blew up"))
        monkeypatch.setattr("opentau.scripts.train.make_envs", make)
        monkeypatch.setattr("opentau.scripts.train.close_envs", Mock())
        monkeypatch.setattr("opentau.scripts.train.eval_policy_all", boom)

        policy = Mock()
        with pytest.raises(RuntimeError, match="eval blew up"):
            _eval_with_fresh_envs(self._cfg(), policy, self._acc(), None, "000010")
        policy.reset.assert_called_once_with()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
