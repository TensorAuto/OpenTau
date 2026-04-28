#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-step timing breakdown for the training loop.

Mirrors ``opentau/scripts/train.py`` setup (Accelerator under DDP or
DeepSpeed depending on the supplied accelerate config; same config
parser, same dataset/policy/optimizer construction), then runs a short
loop that splits wall-clock per step into phases:

  1. ``dataload_wait``    — time blocked on ``next(train_dl_iter)``
  2. ``forward``          — ``policy.forward(batch)`` + loss combine
  3. ``bwd``              — just ``accelerator.backward(loss)`` (includes
                             DeepSpeed/DDP gradient reduce)
  4. ``unscale_clip``     — ``unscale_gradients`` + ``clip_grad_norm_``
  5. ``optim_step``       — ``optimizer.step()`` (includes ZeRO partition
                             update + parameter all-gather)
  6. ``zero_grad_sched``  — ``optimizer.zero_grad()`` + scheduler step
  7. ``backward_step``    — sum of phases 3–6 (kept as an aggregate row
                             for easy comparison with earlier runs that
                             only reported the combined number)
  8. ``sync_gather``      — the 5 ``gather_for_metrics(...).item()`` calls

All ranks call ``torch.cuda.synchronize()`` at phase boundaries so the
reported times include collective + H2D sync costs. Only rank 0 prints.

Usage (same launch incantation as train.py):

    accelerate launch \
        --config_file configs/libero/reproduce_pi05_libero_accelerate_config.yaml \
        src/opentau/scripts/profile_step.py \
        --config_path=configs/libero/reproduce_pi05_libero.json

Environment variables (all optional):

  - ``PROFILE_STEPS=N``       (default 200)  number of measured steps.
    ``cfg.steps`` from the training config is intentionally ignored
    (production configs set it to ~1M).
  - ``PROFILE_NO_OPTIM=1``    (default 0)    skip optimizer creation and
    ``optimizer.step`` / ``zero_grad`` entirely. Useful for isolating
    raw forward+backward compute on a single GPU (no Adam state
    allocated, so a large bf16 model fits without ZeRO partitioning).
  - ``FIND_UNUSED_PARAMS``    (default true) toggles DDP's
    ``find_unused_parameters`` kwarg. Ignored under DeepSpeed. Set to
    ``false`` after auditing with ``find_unused_params.py``.
  - ``FUSED_ADAMW``           (unset = use factory default) force-toggle
    ``torch.optim.AdamW(fused=True|false)`` for an A/B without touching
    the optimizer config JSON.
  - ``PROFILE_STEP_JSON=PATH``  (optional)  dump a JSON summary of
    phase means/medians to ``PATH`` on rank 0 for easy scripting.
"""

import json
import logging
import os
import time
from collections import defaultdict
from statistics import mean, median
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import accelerate
import torch
from accelerate.utils import DistributedDataParallelKwargs

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import cycle
from opentau.optim.factory import make_optimizer_and_scheduler
from opentau.policies.factory import make_policy
from opentau.utils.accelerate_utils import set_proc_accelerator
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import init_logging, is_launched_with_accelerate

WARMUP_STEPS = 20
DEFAULT_MEASURE_STEPS = 200


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _fmt_ms(seconds_list):
    if not seconds_list:
        return "n/a"
    mean_ms = mean(seconds_list) * 1000.0
    median_ms = median(seconds_list) * 1000.0
    p95_ms = sorted(seconds_list)[int(0.95 * (len(seconds_list) - 1))] * 1000.0
    return f"mean={mean_ms:7.2f}ms  median={median_ms:7.2f}ms  p95={p95_ms:7.2f}ms"


@parser.wrap()
def profile(cfg: TrainPipelineConfig):
    cfg.validate()

    # Match train.py's default of find_unused_parameters=True, but allow opting
    # out via env var. The kwarg is silently ignored under DeepSpeed, but under
    # bare DDP it is active and adds its own per-step graph-walk cost.
    find_unused = os.environ.get("FIND_UNUSED_PARAMS", "true").lower() == "true"
    accelerator_kwargs: dict[str, Any] = {
        "step_scheduler_with_optimizer": False,
        "split_batches": False,
        "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=find_unused)],
    }
    if cfg.gradient_accumulation_steps > 1:
        accelerator_kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    init_logging(accelerator, level=logging.INFO)
    set_proc_accelerator(accelerator)

    # We intentionally ignore cfg.steps here (production configs set it to
    # ~1M). Override with PROFILE_STEPS=<N> if you want a longer sample.
    measure_steps = int(os.environ.get("PROFILE_STEPS", DEFAULT_MEASURE_STEPS))
    if accelerator.is_main_process:
        logging.info(
            "profile_step: warmup=%d, measured=%d, num_processes=%d, batch_size=%d, "
            "num_workers=%d, prefetch_factor=%s",
            WARMUP_STEPS,
            measure_steps,
            accelerator.num_processes,
            cfg.batch_size,
            cfg.num_workers,
            cfg.prefetch_factor,
        )

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    if cfg.val_freq > 0:
        train_dataset, _ = make_dataset_mixture(cfg)
    else:
        train_dataset = make_dataset_mixture(cfg)

    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    policy.to(torch.bfloat16)

    skip_optim = os.environ.get("PROFILE_NO_OPTIM", "0") == "1"
    if skip_optim:
        if accelerator.is_main_process:
            logging.info(
                "PROFILE_NO_OPTIM=1 set — skipping optimizer creation. "
                "Measures forward+backward compute only, no Adam state allocated."
            )
        optimizer, lr_scheduler = None, None
    else:
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

        # Optional: replace the AdamW optimizer returned by the factory with a
        # fused-kernel variant, for A/B benchmarking. Zero change to production
        # code; scoped to this script via env var. When FUSED_ADAMW=true we
        # rebuild with the same hyperparameters but ``fused=True``. If
        # FUSED_ADAMW=false we explicitly pass ``fused=False`` so the A/B is
        # symmetric. Unset = leave PyTorch defaults untouched.
        fused_env = os.environ.get("FUSED_ADAMW")
        if fused_env is not None and isinstance(optimizer, torch.optim.AdamW):
            want_fused = fused_env.lower() == "true"
            old_pg = optimizer.param_groups
            # Rebuild one group at a time so per-group hyperparameters
            # (e.g. different lr for vision vs. expert) are preserved.
            # IMPORTANT: drop any existing ``fused``/``foreach`` entries from the
            # copied dicts. torch.optim.Optimizer.add_param_group uses
            # ``param_group.setdefault(name, default)``, so a leftover
            # ``fused: None`` in the dict would silently override our top-level
            # ``fused=True`` kwarg (which only becomes the default). Stripping
            # the keys lets the default win.
            drop_keys = {"params", "fused", "foreach"}
            param_groups = [
                {k: v for k, v in pg.items() if k not in drop_keys} | {"params": pg["params"]}
                for pg in old_pg
            ]
            # Pull defaults from the first group; AdamWConfig uses uniform
            # values in the current code, but future-proof it.
            defaults = old_pg[0]
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=defaults["lr"],
                betas=defaults.get("betas", (0.9, 0.999)),
                eps=defaults.get("eps", 1e-8),
                weight_decay=defaults.get("weight_decay", 0.0),
                fused=want_fused,
            )
            # Rewire the LR scheduler so its internal optimizer-step counter
            # (set up by _LRScheduler.__init__ on the OLD optimizer) tracks
            # the new optimizer. Without this, every scheduler.step() warns
            # "Detected call of lr_scheduler.step() before optimizer.step()"
            # because the old .step wrapper is no longer called. Cosmetic
            # warning in the benchmark (throughput numbers are unaffected),
            # but noisy. In production this code path never runs: the real
            # wire-up builds the fused optimizer first, then the scheduler
            # patches it naturally.
            if lr_scheduler is not None:
                lr_scheduler.optimizer = optimizer
                if not hasattr(optimizer.step, "_with_counter"):

                    def _with_counter(step_fn, opt):
                        def wrapped(*args, **kwargs):
                            opt._step_count += 1
                            return step_fn(*args, **kwargs)

                        wrapped._with_counter = True
                        return wrapped

                    optimizer._step_count = 0
                    optimizer.step = _with_counter(optimizer.step, optimizer)
            if accelerator.is_main_process:
                # Confirm the implementation actually in effect. Read back from
                # the optimizer itself so we catch cases where PyTorch silently
                # falls back (e.g. unsupported dtype, capturable conflicts,
                # missing CUDA). PyTorch stores fused/foreach on each param
                # group after __init__.
                actual_flags = {k: optimizer.param_groups[0].get(k, "<unset>") for k in ("fused", "foreach")}
                total_params = sum(
                    p.numel() for g in optimizer.param_groups for p in g["params"] if p.requires_grad
                )
                dtype_counts: dict[str, int] = defaultdict(int)
                for g in optimizer.param_groups:
                    for p in g["params"]:
                        if p.requires_grad:
                            dtype_counts[str(p.dtype)] += 1
                logging.info(
                    "FUSED_ADAMW=%s: rebuilt AdamW with fused=%s | "
                    "optimizer.param_groups[0] reports %s | "
                    "trainable tensors by dtype: %s | total trainable params: %s",
                    fused_env,
                    want_fused,
                    actual_flags,
                    dict(dtype_counts),
                    f"{total_params:,}",
                )

    train_dataloader = train_dataset.get_dataloader()
    if skip_optim:
        policy, train_dataloader = accelerator.prepare(policy, train_dataloader)
    else:
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
    train_dl_iter = cycle(train_dataloader)

    policy.train()

    phases = defaultdict(list)

    total_steps = WARMUP_STEPS + measure_steps
    loop_start = time.perf_counter()

    for step in range(total_steps):
        measuring = step >= WARMUP_STEPS

        # Phase 1: dataload_wait
        _sync()
        t0 = time.perf_counter()
        batch = next(train_dl_iter)
        _sync()
        t1 = time.perf_counter()

        # Phase 2: forward (mirror update_policy lines 74-77)
        losses = policy.forward(batch)
        loss = cfg.loss_weighting["MSE"] * losses["MSE"] + cfg.loss_weighting["CE"] * losses["CE"]
        _sync()
        t2 = time.perf_counter()

        # Phase 3: backward + step (lines 79-92), split into fine phases so
        # we can tell compute-bound backward apart from collective/optimizer
        # time. When PROFILE_NO_OPTIM=1, skip optimizer.step / clip / zero_grad
        # so we measure compute-only and don't need Adam state memory.
        accelerator.backward(loss)
        _sync()
        t3a = time.perf_counter()

        if not skip_optim:
            accelerator.unscale_gradients(optimizer=optimizer)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
        _sync()
        t3b = time.perf_counter()

        if not skip_optim:
            optimizer.step()
        _sync()
        t3c = time.perf_counter()

        if skip_optim:
            for p in policy.parameters():
                p.grad = None
        else:
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        _sync()
        t3 = time.perf_counter()

        # Phase 4: the 5 gather_for_metrics().item() (lines 94-105) — faithful replication
        _first_loss_tensor = next(lt for lt in losses.values() if isinstance(lt, torch.Tensor))
        zero = torch.tensor(0.0, device=_first_loss_tensor.device, dtype=_first_loss_tensor.dtype)
        _ = accelerator.gather_for_metrics(loss).mean().item()
        _ = accelerator.gather_for_metrics(losses["MSE"]).to(dtype=torch.float32).mean().item()
        _ = accelerator.gather_for_metrics(losses["CE"]).to(dtype=torch.float32).mean().item()
        _ = accelerator.gather_for_metrics(losses.get("L1", zero)).to(dtype=torch.float32).mean().item()
        _ = accelerator.gather_for_metrics(losses.get("Accuracy", zero)).to(dtype=torch.float32).mean().item()
        _sync()
        t4 = time.perf_counter()

        if measuring:
            phases["dataload_wait"].append(t1 - t0)
            phases["forward"].append(t2 - t1)
            phases["bwd"].append(t3a - t2)
            phases["unscale_clip"].append(t3b - t3a)
            phases["optim_step"].append(t3c - t3b)
            phases["zero_grad_sched"].append(t3 - t3c)
            phases["backward_step"].append(t3 - t2)
            phases["sync_gather"].append(t4 - t3)
            phases["total"].append(t4 - t0)

        if accelerator.is_main_process and (step + 1) % 50 == 0:
            logging.info(
                "step %d/%d done (last step total=%.1fms)",
                step + 1,
                total_steps,
                (t4 - t0) * 1000.0,
            )

    loop_wall = time.perf_counter() - loop_start

    if accelerator.is_main_process:
        print("\n=========== profile_step results (rank 0) ===========")
        print(f"warmup={WARMUP_STEPS} measured={measure_steps} ranks={accelerator.num_processes}")
        print(
            f"batch_size={cfg.batch_size} num_workers={cfg.num_workers} prefetch_factor={cfg.prefetch_factor}"
        )
        print(f"wall-clock over full loop: {loop_wall:.2f}s")
        print()
        print(f"{'phase':<16} {'stats':<60} {'share':>8}")
        print("-" * 90)
        total_mean = mean(phases["total"]) if phases["total"] else 0.0
        ordered_keys = [
            "dataload_wait",
            "forward",
            "bwd",
            "unscale_clip",
            "optim_step",
            "zero_grad_sched",
            "backward_step",
            "sync_gather",
            "total",
        ]
        for key in ordered_keys:
            vals = phases[key]
            share = (mean(vals) / total_mean * 100.0) if vals and total_mean > 0 else 0.0
            if key == "total":
                marker = "  <-- total"
            elif key == "backward_step":
                marker = f"{share:6.1f}% (= bwd+unscale_clip+optim_step+zero_grad_sched)"
            else:
                marker = f"{share:6.1f}%"
            print(f"{key:<16} {_fmt_ms(vals):<60} {marker}")
        print()
        steps_per_sec = 1.0 / total_mean if total_mean > 0 else 0.0
        samples_per_sec = steps_per_sec * cfg.batch_size * accelerator.num_processes
        print(f"throughput: {steps_per_sec:.2f} steps/s, {samples_per_sec:.1f} global samples/s")
        print("=====================================================\n")

        # Also dump raw numbers for comparison across runs
        out_path = os.environ.get("PROFILE_STEP_JSON")
        if out_path:
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "warmup": WARMUP_STEPS,
                        "measured": measure_steps,
                        "num_processes": accelerator.num_processes,
                        "batch_size": cfg.batch_size,
                        "num_workers": cfg.num_workers,
                        "prefetch_factor": cfg.prefetch_factor,
                        "phase_means_s": {k: mean(v) for k, v in phases.items() if v},
                        "phase_medians_s": {k: median(v) for k, v in phases.items() if v},
                    },
                    f,
                    indent=2,
                )
            print(f"wrote {out_path}")


if __name__ == "__main__":
    if not is_launched_with_accelerate():
        raise Exception(
            "This script should be launched with accelerate. "
            "Use `accelerate launch --config_file <accel.yaml> "
            "src/opentau/scripts/profile_step.py --config_path=<train.json>`."
        )
    profile()
