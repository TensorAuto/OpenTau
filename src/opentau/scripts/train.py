#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import gc
import inspect
import json
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import contextmanager, nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import accelerate
import torch
import wandb
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list, gather_object
from termcolor import colored

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import cycle
from opentau.envs.factory import make_envs
from opentau.envs.utils import close_envs
from opentau.optim.factory import make_optimizer_and_scheduler
from opentau.optim.master_weights import MasterWeightOptimizer
from opentau.policies.factory import make_policy
from opentau.policies.outlier_utils import OutlierRecord
from opentau.policies.pretrained import PreTrainedPolicy, is_norm_buffer_key
from opentau.scripts.eval import (
    collect_grid_summary_videos,
    consolidate_eval_info,
    eval_policy_all,
    make_subgoal_generator,
)
from opentau.utils.accelerate_utils import set_proc_accelerator
from opentau.utils.logging_utils import AverageMeter, MetricsTracker
from opentau.utils.random_utils import set_seed
from opentau.utils.train_utils import (
    find_missing_rng_state_ranks,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    load_training_step,
    prune_old_checkpoints,
    reseed_new_ranks_on_resume,
    save_checkpoint,
)
from opentau.utils.utils import (
    encode_accelerator_state_dict,
    format_big_number,
    init_logging,
    is_launched_with_accelerate,
)


def update_policy(
    train_config: TrainPipelineConfig,
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: AcceleratedOptimizer,
    grad_clip_norm: float,
    accelerator: accelerate.Accelerator,
    lr_scheduler: AcceleratedScheduler | None = None,
) -> tuple[MetricsTracker, dict]:
    policy.train()
    losses = policy.forward(batch)
    outlier_records = losses.pop("outlier_records", [])
    loss = (
        train_config.loss_weighting["MSE"] * losses["MSE"] + train_config.loss_weighting["CE"] * losses["CE"]
    )

    accelerator.backward(loss)
    accelerator.unscale_gradients(optimizer=optimizer)

    if accelerator.sync_gradients:
        # When the optimizer is wrapped in ``MasterWeightOptimizer`` (the
        # DDP / single / FSDP path; see issue #181), gradients live in bf16
        # on the live params and have not yet been upcast to the fp32 masters.
        # Calling ``MasterWeightOptimizer.clip_grad_norm_`` performs the
        # bf16 -> fp32 upcast and clips on the fp32 master grads, so the
        # subsequent ``optimizer.step`` reads from the clipped fp32 grads.
        # Under DeepSpeed the inner BF16_Optimizer clips internally, so we
        # use ``accelerator.clip_grad_norm_`` there.
        inner_opt = getattr(optimizer, "optimizer", optimizer)
        if isinstance(inner_opt, MasterWeightOptimizer):
            grad_norm = inner_opt.clip_grad_norm_(grad_clip_norm)
        else:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        if accelerator.is_main_process:
            train_metrics.grad_norm = grad_norm

    optimizer.step()
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # This calls `torch.distributed.all_gather_into_tensor` under the hood, which is not so efficient.
    # We don't actually want to broadcast the gathered tensors to all processes, but only to the main process.
    # Nonetheless, we still do this for correctness, safety, and simplicity.
    # ``L1`` and ``Accuracy`` are optional — only the value head currently returns
    # them; the VLA policies (pi0/pi05/pi06/pi07) return only ``MSE`` and ``CE``.
    # Gating the gather on ``key in losses`` keeps the collective count aligned
    # across ranks because the keys returned by ``forward`` are determined by
    # the policy class, which is identical on every rank.
    loss = accelerator.gather_for_metrics(loss).mean().item()
    mse_loss = accelerator.gather_for_metrics(losses["MSE"]).to(dtype=torch.float32).mean().item()
    ce_loss = accelerator.gather_for_metrics(losses["CE"]).to(dtype=torch.float32).mean().item()
    l1_loss = (
        accelerator.gather_for_metrics(losses["L1"]).to(dtype=torch.float32).mean().item()
        if "L1" in losses
        else None
    )
    accuracy = (
        accelerator.gather_for_metrics(losses["Accuracy"]).to(dtype=torch.float32).mean().item()
        if "Accuracy" in losses
        else None
    )
    # Surface the state/action outlier warning on rank 0 (and thus wandb) regardless of which rank
    # held the offending sample. Runs on every rank (it gathers internally); gated on a rank-uniform
    # config check so policies without the field (pi0/pi05/pi06) skip it without an NCCL mismatch.
    outlier_threshold = getattr(accelerator.unwrap_model(policy).config, "warn_outlier_threshold", None)
    if outlier_threshold is not None and outlier_threshold > 0:
        log_outlier_records_distributed(accelerator, outlier_records, outlier_threshold)
    # This actually calls `.update` method of the `AverageMeter` class. This operation is not idempotent.
    # See MetricsTracker.__setattr__ for more details.
    # In other words, setting `train_metrics.loss = 1` and `train_metrics.loss = 2` consecutively results in
    #   an average of 1.5 when formatted as a string, not just 2.
    if accelerator.is_main_process:
        train_metrics.loss = loss
        train_metrics.mse_loss = mse_loss
        train_metrics.ce_loss = ce_loss
        _observe_optional(train_metrics, "l1_loss", "l1_loss", ":.6f", l1_loss)
        _observe_optional(train_metrics, "accuracy", "accuracy", ":.3f", accuracy)
        train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics


def _observe_optional(
    tracker: MetricsTracker,
    key: str,
    display_name: str,
    fmt: str,
    value: float | None,
) -> None:
    """Lazily allocate-and-update an optional ``AverageMeter`` on ``tracker``.

    Centralizes the lazy-allocation pattern used for optional loss metrics
    (currently ``l1_loss`` and ``accuracy``) that only some policies — e.g.
    the value head — emit. Allocating the meter on first observation rather
    than at tracker construction keeps the Python log line and the wandb
    dashboard clean for policies that never produce the metric.

    Callers must already be on the main process: this function mutates
    ``tracker.metrics`` and that mutation should not be performed on
    non-main ranks (mirroring the surrounding ``if accelerator.is_main_process:``
    guards on regular metric assignments).

    Args:
        tracker: The ``MetricsTracker`` to update.
        key: The metric attribute name (used both as the ``metrics`` dict key
            and as the attribute on ``tracker``).
        display_name: Display name passed to the ``AverageMeter`` constructor
            when first allocating it (e.g. ``"l1_loss"`` for train,
            ``"val_l1_loss"`` for validation).
        fmt: Format string passed to the ``AverageMeter`` constructor.
        value: Value to record; if ``None`` (the metric was absent from this
            batch's ``losses``) the function is a no-op.
    """
    if value is None:
        return
    if key not in tracker.metrics:
        tracker.metrics[key] = AverageMeter(display_name, fmt)
    setattr(tracker, key, value)


# Global cross-step dedup for the state/action outlier warning. The logger below runs the actual
# ``logging.warning`` on rank 0 only, so this is a single GLOBAL map (``(source, key, dim)`` ->
# largest ``|value|`` already warned) rather than the per-rank map the detector used to keep —
# strictly better: a dim that bounces between ranks across epochs is warned once, not once per
# rank. A pair is warned the first time it exceeds the threshold and again only when a strictly
# larger magnitude appears.
_WARNED_OUTLIER_KEYS: dict[tuple[object, str, int], float] = {}


def log_outlier_records_distributed(
    accelerator: accelerate.Accelerator,
    local_records: list[OutlierRecord],
    threshold: float,
) -> None:
    """Gather per-rank outlier records, dedup globally, and warn on rank 0.

    The detector (``detect_state_action_outliers``) runs inside ``forward`` on
    every rank but only finds offenders in that rank's local data shard. Because
    ``wandb.init`` runs on rank 0 only and wandb captures just rank-0's console, a
    warning logged on any other rank never reaches wandb. This collects every
    rank's records onto rank 0 and logs there, so the warning always reaches
    wandb regardless of which rank held the offending sample.

    Rule-5 safe: the only collectives are (1) a SUM reduction of the per-rank
    record *count* and (2) a ``gather_object`` of the records, BOTH gated on the
    globally-reduced count so every rank takes the same branch. They run OUTSIDE
    the ``forward`` graph (here in the training / validation loop), mirroring the
    existing ``gather_for_metrics`` / ``gather_object`` calls.

    Args:
        accelerator: the run's ``Accelerator`` (provides the gathers + rank info).
        local_records: this rank's outlier records for the current batch.
        threshold: the ``|value|`` ceiling, printed in the warning message.
    """
    # (1) Cheap, always-run all-reduce so every rank agrees whether to gather. ``reduce`` returns
    # the summed value to every rank, so ``n_total`` is identical across ranks -> the branch below
    # is rank-aligned (no NCCL mismatch). ``reduce`` rather than ``gather_for_metrics`` on purpose:
    # the latter trims dataloader-padding samples at end-of-loop, which could shave a per-step
    # scalar count (harmless for alignment, but would drop the warning on the last val batch).
    n_total = int(
        accelerator.reduce(
            torch.tensor(len(local_records), device=accelerator.device), reduction="sum"
        ).item()
    )
    if n_total == 0:
        return  # all ranks agree -> nobody calls gather_object -> rank-aligned

    # (2) Heavier gather, only on the rare steps that actually carry an outlier. ``gather_object``
    # concatenates every rank's list into one flat list (and returns the input unchanged when not
    # distributed), so ``gathered`` is the union of all ranks' records.
    gathered: list[OutlierRecord] = gather_object(local_records)
    if not accelerator.is_main_process:
        return

    # Merge to the worst |value| per (source, key, dim) (a sample can surface on several ranks),
    # then apply the cross-step dedup so a steady offender doesn't spam.
    merged: dict[tuple[object, str, int], OutlierRecord] = {}
    for rec in gathered:
        tup = (rec["source"], rec["key"], rec["dim"])
        if tup not in merged or rec["value"] > merged[tup]["value"]:
            merged[tup] = rec
    fresh_by_key: dict[str, list[OutlierRecord]] = {}
    for tup, rec in merged.items():
        prev = _WARNED_OUTLIER_KEYS.get(tup)
        if prev is None or rec["value"] > prev:
            _WARNED_OUTLIER_KEYS[tup] = rec["value"]
            fresh_by_key.setdefault(rec["key"], []).append(rec)
    for key, fresh in fresh_by_key.items():
        # Report the worst (largest |value|) among the new/worsened offenders for this key.
        worst = max(fresh, key=lambda r: r["value"])
        fresh_dims = sorted({r["dim"] for r in fresh})
        logging.warning(
            "Outlier normalized %s: %d new/worsened (source,dim) offender(s) exceed |%.1f|; "
            "dims=%s; worst dim=%d max=%.2f (source=%s episode=%s frame=%s). "
            "Re-warned only when a (source,dim) pair's magnitude exceeds its last warned value.",
            key,
            len(fresh),
            threshold,
            fresh_dims,
            worst["dim"],
            worst["value"],
            worst["source"],
            worst["episode"],
            worst["frame"],
        )


def _commit_wandb_step(accelerator: accelerate.Accelerator, step: int) -> None:
    """Seal + flush the pending wandb row for ``step`` immediately.

    OpenTau logs every metric with an explicit ``step=``, so wandb keeps row N
    open as the "pending" row (letting the many per-step ``accelerator.log(...)``
    calls accumulate into it) and only seals + uploads it when it next sees a
    ``log(step > N)`` -- i.e. when step N+1 first logs. That makes the newest
    point (especially a just-finished eval and its videos) lag one logging
    interval on the dashboard, and risks losing the last pending row on an
    ungraceful kill.

    Calling this once, after every training / validation / eval log for
    ``step``, forces the seal now. ``accelerator.log(values, step, log_kwargs)``
    forwards ``log_kwargs["wandb"]`` straight to ``wandb.run.log`` (other
    trackers only see kwargs keyed to their own name), so ``commit=True`` is
    tracker-safe; an empty ``values`` dict flushes the accumulated row without
    adding keys.

    Must be the LAST wandb write for ``step``: wandb requires monotonically
    increasing steps and warns-and-drops any ``log(step=N)`` issued after N is
    sealed.

    Args:
        accelerator: The active accelerate ``Accelerator`` (caller must already
            be on the main process, where the trackers live).
        step: The training step whose pending wandb row should be sealed.
    """
    accelerator.log({}, step=step, log_kwargs={"wandb": {"commit": True}})


def _mixture_weighted_aggregate(
    per_dataset_trackers: dict[str, MetricsTracker],
    name_to_weight: dict[str, float],
) -> dict[str, float]:
    """Mixture-weighted average of per-dataset validation metrics.

    Weights are taken from ``name_to_weight`` and renormalized over only the
    keys present in ``per_dataset_trackers`` (groups with no validation samples
    in the combined pass are simply absent from the trackers). When the
    renormalization total is 0 -- empty trackers, or all selected groups have
    weight 0 -- every metric is returned as ``0.0``. Keys are generic: the
    validation path keys both trackers and weights by norm-head ``norm_key``.

    The aggregated metric keys are derived from the first tracker's meters.
    All per-dataset trackers share the same meter set because they're all
    populated by the same policy's ``forward`` (which deterministically
    returns the same loss keys for every batch and every rank), so any
    tracker's keys are representative.

    Args:
        per_dataset_trackers: One ``MetricsTracker`` per group present in the
            validation pass, keyed by group identifier (the validation path keys
            by norm-head ``norm_key``).
        name_to_weight: Mapping from that same group identifier to its mixture
            weight (need not be normalized; need not be a strict subset/superset
            of the tracker keys, but must contain every tracker key).

    Returns:
        Dict mapping each metric name found on the trackers to its weighted
        average. Empty when ``per_dataset_trackers`` is empty.
    """
    if not per_dataset_trackers:
        return {}
    metric_keys = tuple(next(iter(per_dataset_trackers.values())).metrics.keys())

    weights = {name: name_to_weight[name] for name in per_dataset_trackers}
    total = sum(weights.values())
    if total <= 0:
        return dict.fromkeys(metric_keys, 0.0)

    per_dataset_dicts = {
        name: tracker.to_dict(use_avg=True) for name, tracker in per_dataset_trackers.items()
    }
    return {
        k: sum((w / total) * per_dataset_dicts[name][k] for name, w in weights.items()) for k in metric_keys
    }


def _bucket_per_sample(
    group_index: torch.Tensor,
    mse_sum: torch.Tensor,
    mse_count: torch.Tensor,
    ce_sum: torch.Tensor,
    ce_count: torch.Tensor,
) -> dict[int, dict[str, float]]:
    """Bucket per-sample MSE/CE ``(sum, count)`` by an integer group key.

    All inputs are 1-D tensors of equal length: ``group_index`` is the per-sample
    integer group (norm-head row, or per-dataset enumerate index), and the others
    are the per-sample numerator/denominator from the policies' ``PerSampleLoss``
    outputs. Returns ``{group: {"mse_sum", "mse_count", "ce_sum", "ce_count"}}``
    summed over the samples in each group; the masked mean for a group is then
    ``Σsum / Σcount``.

    Pure / CPU-only (no collectives), so the validation loop can call it on rank 0
    after gathering, and it is unit-testable without an accelerator. Disjoint groups
    never cross-contaminate, which is exactly what lets a *mixed* batch (the combined
    val pass) be disaggregated by provenance.
    """
    buckets: dict[int, dict[str, float]] = {}
    for g in torch.unique(group_index).tolist():
        mask = group_index == g
        buckets[int(g)] = {
            "mse_sum": float(mse_sum[mask].sum()),
            "mse_count": float(mse_count[mask].sum()),
            "ce_sum": float(ce_sum[mask].sum()),
            "ce_count": float(ce_count[mask].sum()),
        }
    return buckets


def _find_unused_params_from_env() -> bool:
    """Parse the ``FIND_UNUSED_PARAMS`` env var into a bool.

    Under DDP this controls whether the reducer walks the autograd graph
    after each backward to discover parameters that did not receive a
    gradient. It is silently ignored under DeepSpeed. Default is True
    for safety (policies with config-gated heads can produce unused
    params); set ``FIND_UNUSED_PARAMS=false`` once a run has been
    audited with ``scripts/find_unused_params.py`` to reclaim the
    per-step graph-walk cost (~10-15% of step time on pi05).

    Returns:
        bool: True when the env var is unset or equals ``"true"``
        (case-insensitive); False for any other value.
    """
    return os.environ.get("FIND_UNUSED_PARAMS", "true").lower() == "true"


@contextmanager
def _zero3_disabled_init_context(accelerator: accelerate.Accelerator):
    """Context manager that suppresses ZeRO-3 per-construction param partitioning.

    Under DeepSpeed ZeRO-3, transformers' `is_deepspeed_zero3_enabled()` flag
    routes module construction through DeepSpeed's `partition_parameters`
    wrapper, which shards each parameter as it's created. Several init
    paths in our model graph need the full tensor shape and crash on the
    per-rank shard — concretely, transformers' SigLIP `_init_weights`
    calls `lecun_normal_` → `_calculate_fan_in_and_fan_out`, which raises
    on a 0-D shard.

    The accelerate yaml field `zero3_init_flag: false` only controls
    transformers' `from_pretrained` integration, NOT the partitioning
    wrapper that fires on plain `__init__`. To suppress the wrapper for a
    bounded section we have to unset the active `HfDeepSpeedConfig` directly.

    On entry: if the active backend is DeepSpeed ZeRO-3, save and clear the
    HF-side DeepSpeed config so `is_deepspeed_zero3_enabled()` returns False.
    On exit: restore it. ZeRO-3 itself still partitions the params correctly
    when `accelerator.prepare` later wraps the model.

    Args:
        accelerator: The active accelerate ``Accelerator``.
    """
    if accelerator.distributed_type != accelerate.DistributedType.DEEPSPEED:
        yield
        return
    zero_stage = accelerator.deepspeed_plugin.hf_ds_config.config.get("zero_optimization", {}).get("stage", 0)
    if zero_stage < 3:
        yield
        return

    from transformers.integrations.deepspeed import (
        is_deepspeed_zero3_enabled,
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )

    saved_hf_ds_config = accelerator.deepspeed_plugin.hf_ds_config if is_deepspeed_zero3_enabled() else None
    try:
        unset_hf_deepspeed_config()
        yield
    finally:
        if saved_hf_ds_config is not None:
            set_hf_deepspeed_config(saved_hf_ds_config)


def _sync_deepspeed_gradient_accumulation_steps(
    accelerator: accelerate.Accelerator, cfg: TrainPipelineConfig
) -> None:
    """Make TrainPipelineConfig the single source of truth for gradient_accumulation_steps.

    When DeepSpeed is the distributed backend, the value declared in the Accelerate YAML's
    `deepspeed_config.gradient_accumulation_steps` is forcibly overridden to match
    `cfg.gradient_accumulation_steps`. Must be called on all ranks and before
    `accelerator.prepare(...)`, because `_prepare_deepspeed` reads this value from
    `hf_ds_config.config` on every rank at prepare time.
    """
    if accelerator.distributed_type != accelerate.DistributedType.DEEPSPEED:
        return

    ds_config = accelerator.deepspeed_plugin.hf_ds_config.config
    current = ds_config.get("gradient_accumulation_steps", 1)
    target = cfg.gradient_accumulation_steps
    if current != target and accelerator.is_main_process:
        logging.warning(
            "Overriding DeepSpeed `gradient_accumulation_steps` (%s) with the value from "
            "TrainPipelineConfig (%s). TrainPipelineConfig is the single source of truth; "
            "the value in the Accelerate YAML is ignored.",
            current,
            target,
        )
    ds_config["gradient_accumulation_steps"] = target
    accelerator.deepspeed_plugin.gradient_accumulation_steps = target


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    find_unused = _find_unused_params_from_env()
    accelerator_kwargs = {
        "step_scheduler_with_optimizer": False,
        "split_batches": False,  # split_batches == True is not working anyways
        "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=find_unused)],
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
    }
    if cfg.wandb.enable:
        accelerator_kwargs["log_with"] = "wandb"

    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    # ``cfg.output_dir`` embeds a per-process wall-clock timestamp (assigned in
    # ``TrainPipelineConfig`` as ``f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}"``) and every
    # rank parses the config independently, so a launch whose startup straddles a one-second
    # tick leaves ranks disagreeing on ``output_dir`` (e.g. one rank lands on ``HH-MM-49``
    # while the rest use ``HH-MM-50``). Each rank then writes its checkpoint shard / RNG state
    # into a *sibling* directory, silently yielding a checkpoint with fewer than ``world_size``
    # shards that cannot be consolidated or resumed. Broadcast rank 0's value so all ranks
    # share one output tree before ``output_dir`` is first used (the Accelerator does not
    # consume ``output_dir``, so doing this immediately after construction is safe).
    _output_dir = [str(cfg.output_dir)]
    broadcast_object_list(_output_dir, from_process=0)
    cfg.output_dir = Path(_output_dir[0])
    init_logging(accelerator, level=logging.DEBUG if cfg.debug else logging.INFO)
    # Register accelerator globally for use in other modules, (e.g., detect current rank, etc.)
    set_proc_accelerator(accelerator)

    # Must run before `encode_accelerator_state_dict` + `init_trackers` below so the
    # wandb-logged accelerator config and the value DeepSpeed consumes at prepare()
    # time both reflect TrainPipelineConfig.
    _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    # Strict guard for gradient_checkpointing: the pi05 custom forward loop
    # wraps layer bodies in torch.utils.checkpoint.checkpoint. With
    # use_reentrant=False this uses saved_tensors_hooks, which co-exist
    # with FSDP's all-gather hooks in PyTorch ≥2.4 — so FSDP is supported.
    # DeepSpeed ZeRO-3, however, prefetches/re-shards through DeepSpeed-specific
    # hooks that torch.utils.checkpoint does not fire, so it can silently
    # produce wrong gradients; keep that case rejected.
    if getattr(cfg.policy, "gradient_checkpointing", False):
        grad_ckpt_allowed = {
            accelerate.DistributedType.MULTI_GPU,
            accelerate.DistributedType.NO,
            accelerate.DistributedType.DEEPSPEED,
            accelerate.DistributedType.FSDP,
        }
        if accelerator.distributed_type not in grad_ckpt_allowed:
            raise ValueError(
                f"gradient_checkpointing=True is not supported under "
                f"distributed_type={accelerator.distributed_type}. Supported: "
                "MULTI_GPU (DDP), NO (single process), FSDP, DEEPSPEED (ZeRO-1/2 only). "
                "DeepSpeed ZeRO-3 needs deepspeed.checkpointing.checkpoint hooks "
                "which pi05's custom per-layer forward does not wire up. "
                "Either set gradient_checkpointing=False or switch to a "
                "supported backend."
            )
        if accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED:
            zero_stage = accelerator.deepspeed_plugin.hf_ds_config.config.get("zero_optimization", {}).get(
                "stage", 0
            )
            if zero_stage >= 3:
                raise ValueError(
                    f"gradient_checkpointing=True is not supported under "
                    f"DeepSpeed ZeRO stage {zero_stage}. ZeRO-3 re-shards parameters "
                    "during forward and needs deepspeed.checkpointing.checkpoint "
                    "rather than torch.utils.checkpoint. Either set "
                    "gradient_checkpointing=False, use zero_stage: 1 or 2, or "
                    "switch to FSDP (which has equivalent param sharding and "
                    "is compatible with torch.utils.checkpoint)."
                )

    logging.info(pformat(cfg.to_dict()))

    if accelerator.is_main_process:
        accelerator_config = encode_accelerator_state_dict(accelerator.state.__dict__)
        logging.info(pformat(accelerator_config))

        if cfg.wandb.enable:
            step = load_training_step(cfg.checkpoint_path) if cfg.resume else None
            slurm_dict = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
            accelerator.init_trackers(
                cfg.wandb.project,
                config={**cfg.to_dict(), "accelerator": accelerator_config, "slurm": slurm_dict},
                init_kwargs={"wandb": cfg.wandb.to_wandb_kwargs(step=step)},
            )
            tracker = accelerator.get_tracker("wandb", unwrap=True)
            cfg.wandb.run_id = tracker.id
            logging.info(f"tracker initialized with wandb job id: {tracker.id}")

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Enable anomaly detection for debugging NaN/Inf values
    # (warning: large computational overhead)
    torch.autograd.set_detect_anomaly(cfg.trace_nans)
    if cfg.trace_nans:
        logging.warning("Anomaly detection is enabled. This may significantly slow down training.")
    else:
        logging.info("Anomaly detection is disabled.")

    logging.info("Creating dataset")
    if cfg.val_freq > 0:
        train_dataset, val_dataset = make_dataset_mixture(cfg)
    else:
        train_dataset = make_dataset_mixture(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    eval_envs = None
    eval_subgoal_generator = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_envs = make_envs(
            cfg.env, cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )
        eval_subgoal_generator = make_subgoal_generator(cfg)

    logging.info("Creating policy")
    # FSDP needs the policy built in fp32 so its ``MixedPrecision(
    # param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)`` policy can
    # downcast on the fly while keeping the fp32 outer (master) params for
    # AdamW to step on. Without this gate, ``Gemma3WithExpertModel.__init__``
    # would call ``to_bfloat16_like_physical_intelligence`` and the params
    # would already be bf16 by the time FSDP wraps them — MixedPrecision
    # becomes a storage no-op and the optimizer ends up stepping on bf16
    # sharded params with bf16 Adam state, which underflows on small late-
    # training updates. Set the flag where it exists (currently only on
    # ``Gemma3WithExpertConfig``); other policies that don't yet support
    # FSDP simply lack the attribute.
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        vlm_config = getattr(cfg.policy, "vlm_config", None)
        if vlm_config is not None and hasattr(vlm_config, "disable_internal_bf16_cast"):
            vlm_config.disable_internal_bf16_cast = True

    # DeepSpeed ZeRO-3 installs a `partition_parameters` wrapper that shards
    # parameters at construction time (`zero3_init_flag` only controls the
    # transformers `from_pretrained` integration, not this wrapper). Some
    # init paths — e.g. SigLIP's `lecun_normal_` → `_calculate_fan_in_and_fan_out`
    # — need the full tensor shape and crash on a per-rank shard. Build the
    # model with partitioning disabled; ZeRO will re-shard properly when
    # `accelerator.prepare` wraps it. No-op for any non-ZeRO-3 backend.
    with _zero3_disabled_init_context(accelerator):
        policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    # Per-backend precision regime:
    #   * DDP / single-process: outer ``policy.to(bfloat16)`` cast here makes
    #     live params bf16, then ``MasterWeightOptimizer.from_existing``
    #     (issue #181) layers fp32 master + fp32 Adam state on top.
    #   * DeepSpeed ZeRO-1/2: outer ``policy.to(bfloat16)`` cast here, then
    #     ``BF16_Optimizer`` keeps fp32 master + fp32 Adam state, reduces
    #     gradients in bf16.
    #   * FSDP-FULL_SHARD: skip the outer cast (and the model's inner
    #     ``to_bfloat16_like_physical_intelligence`` was already gated above)
    #     so the policy stays fp32 going into ``accelerator.prepare``. FSDP's
    #     ``MixedPrecision`` then provides bf16 compute (forward/backward),
    #     bf16 reduce-scatter (matches DeepSpeed), and the optimizer steps
    #     on the fp32 outer (master) params with fp32 Adam state. Live
    #     params consume HBM in fp32 (sharded across 8 ranks); compute
    #     params materialize in bf16 transiently during the all-gather.
    if accelerator.distributed_type != accelerate.DistributedType.FSDP:
        policy.to(torch.bfloat16)
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # Outside DeepSpeed *and* FSDP, wrap the optimizer so it carries fp32
    # master weights and fp32 Adam state. DeepSpeed provides this via
    # ``BF16_Optimizer``. FSDP provides it via ``MixedPrecision`` over the
    # fp32-built policy — the optimizer was constructed over fp32 outer
    # params and AdamW's exp_avg / exp_avg_sq are allocated to match → fp32.
    # Wrapping with ``MasterWeightOptimizer`` on top of FSDP's flat-param
    # handles also misaligns and triggers a NCCL desync during the first
    # backward (observed empirically), so skip there too.
    if accelerator.distributed_type not in (
        accelerate.DistributedType.DEEPSPEED,
        accelerate.DistributedType.FSDP,
    ):
        optimizer = MasterWeightOptimizer.from_existing(optimizer)
        # ``make_optimizer_and_scheduler`` bound the LR scheduler to the
        # ORIGINAL ``torch.optim.AdamW`` (now discarded in favour of the
        # wrapper's new inner). Without this rebind, ``lr_scheduler.step()``
        # would mutate the orphaned optimizer's ``param_groups[i]['lr']``
        # and the inner AdamW's lr would silently stay at the initial
        # value forever — schedule applied to nothing.
        if lr_scheduler is not None:
            lr_scheduler.optimizer = optimizer

    step = 0  # number of policy updates (forward + backward + optim)

    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if cfg.val_freq > 0:
        train_dataloader = train_dataset.get_dataloader()
        # A single combined validation pass over the whole mixture: under
        # ``accelerator.prepare`` every rank's shard stays full even when individual
        # subsets have fewer frames than ``world_size`` (the old one-loader-per-dataset
        # path left most ranks idle on tiny subsets and stacked that idle time). The
        # validation loop disaggregates metrics per (dataset, control_mode) from each
        # sample's ``dataset_index`` / ``dataset_repo_id`` provenance instead.
        val_dataloader = val_dataset.get_combined_val_dataloader()
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
        if val_dataloader is not None:
            val_dataloader = accelerator.prepare(val_dataloader)
    else:
        train_dataloader = train_dataset.get_dataloader()
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
    # ``accelerator.prepare`` may have moved the policy from CPU to GPU.
    # When ``MasterWeightOptimizer`` is in use, the fp32 masters were cloned
    # from the bf16 live params at wrap-time (still on CPU) and would
    # otherwise stay there — Adam would then run on CPU master tensors,
    # paying a CPU<->GPU copy on every grad upcast and param downcast and
    # never letting the masters appear in nvidia-smi accounting. Rebuild
    # masters from the now-migrated live params so they live on the same
    # device as the model. No-op when devices already agree.
    inner_opt_for_migrate = getattr(optimizer, "optimizer", optimizer)
    if isinstance(inner_opt_for_migrate, MasterWeightOptimizer):
        inner_opt_for_migrate.rebuild_masters_from_live(policy.parameters())
    train_dl_iter = cycle(train_dataloader)

    # Register the LR scheduler for checkpointing
    accelerator.register_for_checkpointing(lr_scheduler)

    # When `save_normalization_stats=False`, strip the per-feature Normalize /
    # Unnormalize buffers from each model state_dict that Accelerate is about
    # to write under `accelerator.save_state(...)`. The hook runs on every
    # rank before the actual safetensors write, so the filtered keys never
    # land on disk. The reload side relies on `make_policy(..., ds_meta=...)`
    # to call `_inject_stats(...)` and repopulate the buffers (see
    # `policies/factory.py::make_policy`).
    if not cfg.policy.save_normalization_stats:

        def _strip_norm_buffers_pre_save(models, weights, input_dir):
            del models, input_dir
            for sd in weights:
                for k in list(sd):
                    if is_norm_buffer_key(k):
                        del sd[k]

        accelerator.register_save_state_pre_hook(_strip_norm_buffers_pre_save)

    if cfg.resume:
        # Inspect the on-disk safetensors header (rather than trusting the
        # *current* `cfg.policy.save_normalization_stats` flag) to decide
        # whether `strict=True` would crash on missing norm-buffer keys.
        # This makes resume robust to the user toggling the flag between the
        # initial run and the resume — the file on disk is the only source
        # of truth for what's actually missing. Reading the header is also
        # cheap; safetensors parses just the index.
        load_kwargs: dict = {}
        try:
            from safetensors import safe_open

            ckpt_safetensors = Path(cfg.checkpoint_path) / "model.safetensors"
            if ckpt_safetensors.exists():
                with safe_open(str(ckpt_safetensors), framework="pt") as f:
                    saved_keys = set(f.keys())
                if not any(is_norm_buffer_key(k) for k in saved_keys):
                    # The disk checkpoint omits every norm buffer — strict
                    # load would raise. The buffers were already repopulated
                    # by `_inject_stats` inside `make_policy(..., ds_meta=...)`
                    # above, so falling back to `strict=False` leaves them at
                    # the right values rather than silently overwriting.
                    load_kwargs["strict"] = False
        except Exception as e:
            logging.warning(
                "Could not inspect %s/model.safetensors to decide strict mode "
                "for resume (%s). Falling back to the cfg flag.",
                cfg.checkpoint_path,
                e,
            )
            if not cfg.policy.save_normalization_stats:
                load_kwargs["strict"] = False
        accelerator.load_state(cfg.checkpoint_path, **load_kwargs)

        # When the master-weights wrapper is in use, the live bf16 weights
        # have just been overwritten by ``accelerator.load_state``. Rebuild
        # the fp32 masters from those weights so the inner optimizer's
        # subsequent steps operate on consistent fp32 master copies.
        # (Under DeepSpeed this is unnecessary; ZeRO restores its own fp32
        # masters from its checkpoint.)
        inner_opt_for_resume = getattr(optimizer, "optimizer", optimizer)
        if isinstance(inner_opt_for_resume, MasterWeightOptimizer):
            inner_opt_for_resume.rebuild_masters_from_live(policy.parameters())

        # all processes should load the step & rng states
        step = load_training_state(cfg.checkpoint_path)

        # load_training_state above applies the consolidated rng_state.safetensors
        # to every rank, which would clobber any per-rank reseed. Run the per-rank
        # reseed AFTER the consolidated load so the new ranks end up decorrelated.
        # Per-rank random_states_<rank>.pkl is loaded by accelerator.load_state, but
        # missing files (when resuming on more GPUs than were saved) are silently
        # skipped, leaving new ranks with whichever RNG the consolidated load gave
        # them — identical across all new ranks. Re-seed those new ranks
        # deterministically here.
        reseed_new_ranks_on_resume(cfg.checkpoint_path, accelerator, cfg.seed)

        logging.info(f"Resuming training from checkpoint {cfg.checkpoint_path}")

    policy.train()

    # setup metrics tracker to average metrics over the logging interval.
    # ``l1_loss`` and ``accuracy`` are populated lazily in ``update_policy``
    # iff the policy's ``forward`` returns ``"L1"`` / ``"Accuracy"`` (only the
    # value head currently does); omitting them here keeps logs clean for VLA
    # policies that don't emit those losses.
    train_metrics = {
        "loss": AverageMeter("total_loss", ":.6f"),
        "mse_loss": AverageMeter("mse_loss", ":.6f"),
        "ce_loss": AverageMeter("ce_loss", ":.6f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "grad_norm": AverageMeter("grad_norm", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes,  # split_batches are not working
        train_metrics,
        initial_step=step,
    )

    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        for _ in range(cfg.gradient_accumulation_steps):
            with accelerator.accumulate(policy) if cfg.gradient_accumulation_steps > 1 else nullcontext():
                logging.debug(f"{step=}, {accelerator.sync_gradients=}")
                batch = next(train_dl_iter)

                train_tracker = update_policy(
                    cfg,
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = (step % cfg.save_freq == 0 or step == cfg.steps) and cfg.save_checkpoint
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = cfg.val_freq > 0 and step % cfg.val_freq == 0

        # Only `train_tracker` on the main process keeps useful statistics,
        #  because we guarded it with if accelerator.is_main_process in the `update_policy` function.
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            log_dict = train_tracker.to_dict(use_avg=True)
            accelerator.log({"Training/Loss": log_dict["loss"]}, step=step)
            accelerator.log({"Training/MSE Loss": log_dict["mse_loss"]}, step=step)
            accelerator.log({"Training/CE Loss": log_dict["ce_loss"]}, step=step)
            if "l1_loss" in train_tracker.metrics:
                accelerator.log({"Training/L1 Loss": log_dict["l1_loss"]}, step=step)
            if "accuracy" in train_tracker.metrics:
                accelerator.log({"Training/Accuracy": log_dict["accuracy"]}, step=step)
            accelerator.log({"Training/Learning Rate": log_dict["lr"]}, step=step)
            accelerator.log({"Training/Grad Norm": log_dict["grad_norm"]}, step=step)
            accelerator.log({"Training/Num Samples": log_dict["samples"]}, step=step)
            train_tracker.reset_averages()

        if is_saving_step:
            # TODO: investigate whether this barrier is needed
            accelerator.wait_for_everyone()
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            # save the accelerator state
            # This will save the model, optimizer, and lr_scheduler state
            accelerator.save_state(checkpoint_dir)
            if accelerator.is_main_process:
                # Mirrors the load-side "All dataloader sampler states loaded successfully"
                # summary, since init_logging suppresses the per-dataloader save lines.
                logging.info(f"Saved all dataloader sampler states to {checkpoint_dir}")

            # save axillary objects such as configs, training step, and rng state
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                cfg.policy.pretrained_path = checkpoint_dir
                save_checkpoint(checkpoint_dir, step, cfg)
                if cfg.last_checkpoint_only:
                    prune_old_checkpoints(checkpoint_dir)

            accelerator.wait_for_everyone()

            # Defense-in-depth tripwire for the partial-save / output_dir-divergence failure
            # mode: accelerate writes one ``random_states_<process_index>.pkl`` per process on
            # every backend, so a complete checkpoint has one per rank. Surface a missing rank
            # loudly here (the hard failure otherwise only appears at resume/consolidation time).
            if accelerator.is_main_process:
                missing_ranks = find_missing_rng_state_ranks(checkpoint_dir, accelerator.num_processes)
                if missing_ranks:
                    logging.error(
                        "CHECKPOINT INCOMPLETE: %s is missing per-rank RNG state for rank(s) %s "
                        "(have %d/%d). A rank likely wrote to a different output_dir or failed to "
                        "save; this checkpoint is not resumable.",
                        checkpoint_dir,
                        missing_ranks,
                        accelerator.num_processes - len(missing_ranks),
                        accelerator.num_processes,
                    )

        if is_val_step and val_dataloader is not None:
            policy.eval()

            def _make_val_tracker(current_step: int = step) -> MetricsTracker:
                return MetricsTracker(
                    cfg.batch_size * accelerator.num_processes,
                    {
                        "loss": AverageMeter("val_total_loss", ":.6f"),
                        "mse_loss": AverageMeter("val_mse_loss", ":.6f"),
                        "ce_loss": AverageMeter("val_ce_loss", ":.6f"),
                    },
                    initial_step=current_step,
                )

            # Per-(dataset, control_mode) breakdown needs per-sample losses. Action
            # policies expose them via ``forward(..., return_per_sample=True)``; other
            # policies (e.g. high-level planners) don't, so fall back to an
            # aggregate-only pass for them. The single combined loader — and its
            # full-shard parallelization — applies to every policy regardless.
            unwrapped_policy = accelerator.unwrap_model(policy)
            supports_per_sample = (
                "return_per_sample" in inspect.signature(unwrapped_policy.forward).parameters
            )
            mse_w = cfg.loss_weighting["MSE"]
            ce_w = cfg.loss_weighting["CE"]
            outlier_threshold = getattr(unwrapped_policy.config, "warn_outlier_threshold", None)
            name_to_index = val_dataset.meta.dataset_name_to_index

            # Rank-0 accumulators across the whole pass. Per-sample tensors are gathered
            # every batch (a collective on all ranks) but only retained on the main
            # process, then bucketed once at the end.
            ps_chunks: dict[str, list[torch.Tensor]] = {
                "mse_sum": [],
                "mse_count": [],
                "ce_sum": [],
                "ce_count": [],
                "norm_index": [],
                "source_index": [],
            }
            agg_tracker = _make_val_tracker()  # only used on the fallback (no per-sample) path
            optional_vals: dict[str, list[float]] = {"l1_loss": [], "accuracy": []}

            logging.info(f"Validation at step {step}...")

            with torch.no_grad():
                for batch in val_dataloader:
                    losses = (
                        policy.forward(batch, return_per_sample=True)
                        if supports_per_sample
                        else policy.forward(batch)
                    )
                    outlier_records = losses.pop("outlier_records", [])

                    # Optional value-head metrics (scalars); aggregate-level only. Gather
                    # outside the per-sample branch so both paths handle them identically.
                    l1_val = (
                        accelerator.gather_for_metrics(losses["L1"]).to(dtype=torch.float32).mean().item()
                        if "L1" in losses
                        else None
                    )
                    accuracy_val = (
                        accelerator.gather_for_metrics(losses["Accuracy"])
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                        if "Accuracy" in losses
                        else None
                    )

                    if supports_per_sample:
                        mse_ps = losses["MSE_per_sample"]
                        ce_ps = losses["CE_per_sample"]
                        # Per-sample provenance keys: ``dataset_index`` is the dropout-immune
                        # norm-head row; ``source_index`` is the per-dataset enumerate index
                        # recovered from the (also dropout-immune) ``dataset_repo_id``. Both
                        # gathered as int tensors so the ragged last-batch de-pad stays
                        # row-aligned with the loss tensors (``gather_object`` would not).
                        norm_index = batch["dataset_index"].to(device=accelerator.device, dtype=torch.long)
                        source_index = torch.tensor(
                            [name_to_index[name] for name in batch["dataset_repo_id"]],
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                        # Gather all per-sample tensors in a SINGLE call so accelerate applies
                        # one identical ragged-last-batch de-pad to every entry: the per-sample
                        # losses and their provenance indices stay row-aligned by construction,
                        # rather than relying on six separate de-pads landing on the same trim.
                        gathered = accelerator.gather_for_metrics(
                            {
                                "mse_sum": mse_ps.sum.to(dtype=torch.float32),
                                "mse_count": mse_ps.count.to(dtype=torch.float32),
                                "ce_sum": ce_ps.sum.to(dtype=torch.float32),
                                "ce_count": ce_ps.count.to(dtype=torch.float32),
                                "norm_index": norm_index,
                                "source_index": source_index,
                            }
                        )
                    else:
                        loss = mse_w * losses["MSE"] + ce_w * losses["CE"]
                        loss_val = accelerator.gather_for_metrics(loss).to(dtype=torch.float32).mean().item()
                        mse_val = (
                            accelerator.gather_for_metrics(losses["MSE"])
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        )
                        ce_val = (
                            accelerator.gather_for_metrics(losses["CE"]).to(dtype=torch.float32).mean().item()
                        )

                    # Warn about outliers on rank 0 too during eval (parity with training); see
                    # ``update_policy`` for the rank-uniform gating rationale.
                    if outlier_threshold is not None and outlier_threshold > 0:
                        log_outlier_records_distributed(accelerator, outlier_records, outlier_threshold)

                    if accelerator.is_main_process:
                        if supports_per_sample:
                            for key, value in gathered.items():
                                ps_chunks[key].append(value.cpu())
                        else:
                            agg_tracker.loss = loss_val
                            agg_tracker.mse_loss = mse_val
                            agg_tracker.ce_loss = ce_val
                        if l1_val is not None:
                            optional_vals["l1_loss"].append(l1_val)
                        if accuracy_val is not None:
                            optional_vals["accuracy"].append(accuracy_val)

            if accelerator.is_main_process:
                meta = val_dataset.meta
                name_to_weight = dict(
                    zip(val_dataset.dataset_names, val_dataset.dataset_weights, strict=True)
                )

                if supports_per_sample and ps_chunks["norm_index"]:
                    norm_index = torch.cat(ps_chunks["norm_index"])
                    source_index = torch.cat(ps_chunks["source_index"])
                    mse_sum = torch.cat(ps_chunks["mse_sum"])
                    mse_count = torch.cat(ps_chunks["mse_count"])
                    ce_sum = torch.cat(ps_chunks["ce_sum"])
                    ce_count = torch.cat(ps_chunks["ce_count"])
                    norm_groups = _bucket_per_sample(norm_index, mse_sum, mse_count, ce_sum, ce_count)
                    source_groups = _bucket_per_sample(source_index, mse_sum, mse_count, ce_sum, ce_count)

                    # Headline breakdown: one tracker per norm head (``robot::control_mode``
                    # or a fallback dataset name), keyed by ``norm_key`` for the aggregate.
                    per_normkey_trackers: dict[str, MetricsTracker] = {}
                    for norm_row, group in norm_groups.items():
                        norm_key = meta.norm_keys[norm_row]
                        mse_mean = group["mse_sum"] / (group["mse_count"] + 1e-8)
                        ce_mean = group["ce_sum"] / (group["ce_count"] + 1e-8)
                        loss_mean = mse_w * mse_mean + ce_w * ce_mean
                        tracker = _make_val_tracker()
                        tracker.mse_loss = mse_mean
                        tracker.ce_loss = ce_mean
                        tracker.loss = loss_mean
                        per_normkey_trackers[norm_key] = tracker
                        logging.info(f"Validation/{norm_key} {tracker}")
                        accelerator.log({f"Validation/{norm_key}/Loss": loss_mean}, step=step)
                        accelerator.log({f"Validation/{norm_key}/MSE Loss": mse_mean}, step=step)
                        accelerator.log({f"Validation/{norm_key}/CE Loss": ce_mean}, step=step)

                    # Finer per-source-dataset breakdown (no info loss when datasets share
                    # a norm head). Logged-only; the aggregate stays per-norm-head.
                    for source_row, group in source_groups.items():
                        ds_name = meta.dataset_names[source_row]
                        mse_mean = group["mse_sum"] / (group["mse_count"] + 1e-8)
                        ce_mean = group["ce_sum"] / (group["ce_count"] + 1e-8)
                        loss_mean = mse_w * mse_mean + ce_w * ce_mean
                        accelerator.log({f"Validation/by_dataset/{ds_name}/Loss": loss_mean}, step=step)
                        accelerator.log({f"Validation/by_dataset/{ds_name}/MSE Loss": mse_mean}, step=step)
                        accelerator.log({f"Validation/by_dataset/{ds_name}/CE Loss": ce_mean}, step=step)

                    # Mixture-weighted aggregate over norm heads, so the overall scalar
                    # reflects the training mixture rather than being dominated by whichever
                    # head has the most val frames. Sum each head's member-dataset weights.
                    norm_key_weight: dict[str, float] = {}
                    for name, weight in name_to_weight.items():
                        nk = meta.norm_keys[meta.dataset_to_norm_index[name]]
                        norm_key_weight[nk] = norm_key_weight.get(nk, 0.0) + weight
                    agg = _mixture_weighted_aggregate(per_normkey_trackers, norm_key_weight)
                elif not supports_per_sample:
                    # Fallback: the combined-pass batch-mean aggregate; no per-group breakdown.
                    agg = agg_tracker.to_dict(use_avg=True)
                else:
                    agg = {}

                if agg:
                    logging.info(f"Validation/aggregate {agg}")
                    accelerator.log({"Validation/Loss": agg["loss"]}, step=step)
                    accelerator.log({"Validation/MSE Loss": agg["mse_loss"]}, step=step)
                    accelerator.log({"Validation/CE Loss": agg["ce_loss"]}, step=step)
                if optional_vals["l1_loss"]:
                    accelerator.log(
                        {"Validation/L1 Loss": sum(optional_vals["l1_loss"]) / len(optional_vals["l1_loss"])},
                        step=step,
                    )
                if optional_vals["accuracy"]:
                    accelerator.log(
                        {
                            "Validation/Accuracy": sum(optional_vals["accuracy"])
                            / len(optional_vals["accuracy"])
                        },
                        step=step,
                    )

            # This barrier is probably necessary to ensure
            # other processes wait for the main process to finish saving
            accelerator.wait_for_everyone()

        if is_eval_step and eval_envs:
            # Return the allocator's cached-but-unused blocks (freed training
            # activations) to the CUDA driver before eval, so eval runs at a lower
            # peak and any non-PyTorch GPU consumer (e.g. an EGL render context for
            # sim eval) has room. Training re-grows the cache on the next step.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                pre_eval_alloc_gib = torch.cuda.memory_allocated() / 1024**3
                pre_eval_resv_gib = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.reset_peak_memory_stats()
            else:
                pre_eval_alloc_gib = pre_eval_resv_gib = 0.0
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=accelerator.device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy_all(
                    eval_envs,
                    policy,
                    cfg.eval.n_episodes,
                    cfg,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=cfg.eval.max_episodes_rendered,
                    grid_size=cfg.eval.grid_size,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                    subgoal_generator=eval_subgoal_generator,
                )

            eval_info = gather_object([eval_info])  # gather across all accelerator processes
            if accelerator.is_main_process:
                eval_info = consolidate_eval_info(eval_info)
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # compact per-group + overall summary (drop verbose video_paths)
                def _fmt(m: dict) -> str:
                    return (
                        f"success={m['pc_success']:.1f}% ∑rwrd={m['avg_sum_reward']:.3f} n={m['n_episodes']}"
                    )

                for group, v in eval_info["per_group"].items():
                    logging.info("eval[%s]: %s", group, _fmt(v))
                logging.info("eval[overall]: %s", _fmt(aggregated))

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_per_gpu_s": AverageMeter("eval_per_gpu_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    eval_metrics,
                    initial_step=step,
                )
                eval_tracker.eval_per_gpu_s = aggregated.get("eval_per_gpu_s", float("nan"))
                eval_tracker.avg_sum_reward = aggregated.get("avg_sum_reward", float("nan"))
                eval_tracker.pc_success = aggregated.get("pc_success", float("nan"))
                logging.info(eval_tracker)
                eval_dict = eval_tracker.to_dict(use_avg=True)
                accelerator.log({"Success Rate": eval_dict["pc_success"]}, step=step)
                accelerator.log({"Evaluation Time": eval_dict["eval_per_gpu_s"]}, step=step)
                for group, v in eval_info["per_group"].items():
                    accelerator.log({f"Success/{group}": v["pc_success"]}, step=step)

                # Save eval_info to the same directory as videos
                videos_dir = cfg.output_dir / "eval" / f"videos_step_{step_id}"
                with open(videos_dir / "eval_info.json", "w") as f:
                    json.dump(eval_info, f, indent=2)

                # Log grid-summary eval videos to wandb (skip individual clips).
                # Main-process-only, and globs the shared videos_dir, so it
                # assumes every rank wrote to a filesystem the main process can
                # read (true for the single-node multi-GPU setup; non-rank-0
                # videos on a multi-node run without a shared FS won't be seen).
                if cfg.wandb.enable and not cfg.wandb.disable_video and cfg.eval.max_episodes_rendered > 0:
                    grid_videos = {
                        f"Eval Videos/{task_name}": wandb.Video(grid_path, format="mp4")
                        for task_name, grid_path in collect_grid_summary_videos(videos_dir)
                    }
                    if grid_videos:
                        accelerator.log(grid_videos, step=step)

            # This barrier is to ensure all processes finishes evaluation before the next training step
            # Some processes might be slower than others
            accelerator.wait_for_everyone()

            if torch.cuda.is_available():
                eval_hbm_probe = {
                    "rank": accelerator.process_index,
                    "pre_alloc": pre_eval_alloc_gib,
                    "pre_resv": pre_eval_resv_gib,
                    "peak_alloc": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_resv": torch.cuda.max_memory_reserved() / 1024**3,
                    "post_alloc": torch.cuda.memory_allocated() / 1024**3,
                    "post_resv": torch.cuda.memory_reserved() / 1024**3,
                }
                gathered_eval_hbm = gather_object([eval_hbm_probe])
                if accelerator.is_main_process:
                    for s in sorted(gathered_eval_hbm, key=lambda x: x["rank"]):
                        logging.info(
                            "Eval HBM probe step=%d rank=%d pre=%.2f/%.2f peak=%.2f/%.2f post=%.2f/%.2f retained_alloc=%+.2f GiB",
                            step,
                            s["rank"],
                            s["pre_alloc"],
                            s["pre_resv"],
                            s["peak_alloc"],
                            s["peak_resv"],
                            s["post_alloc"],
                            s["post_resv"],
                            s["post_alloc"] - s["pre_alloc"],
                        )

        # Seal this step's wandb row as soon as all of its logging (training /
        # validation / eval scalars + eval videos) is done, instead of waiting
        # for the next logging step to auto-commit it. Guarded on the same
        # conditions that gate the logging blocks above, so we only commit on
        # steps that actually logged, and only on main where the trackers live.
        # See issue #353.
        if accelerator.is_main_process and (is_log_step or is_val_step or (is_eval_step and eval_envs)):
            _commit_wandb_step(accelerator, step)

    if cfg.eval_freq > 0 and eval_envs:
        close_envs(eval_envs)

    accelerator.end_training()
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    if not is_launched_with_accelerate():
        raise Exception(
            "This script should be launched with accelerate. Please use `accelerate launch` to run this script."
        )

    train()
