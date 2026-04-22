#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-step timing breakdown for the training loop.

Phase-1 diagnostic for the "low GPU utilization" investigation. Mirrors
``opentau/scripts/train.py`` setup (Accelerator + DeepSpeed, same config
parser, same dataset/policy/optimizer construction), then runs a short
loop that splits wall-clock per step into four phases:

  1. ``dataload_wait``  — time blocked on ``next(train_dl_iter)``
  2. ``forward``        — ``policy.forward(batch)`` + loss combine
  3. ``backward_step``  — ``accelerator.backward`` + grad clip + ``optimizer.step``
  4. ``sync_gather``    — the 5 ``gather_for_metrics(...).item()`` calls

All ranks call ``torch.cuda.synchronize()`` at phase boundaries so the
reported times include collective + H2D sync costs. Only rank 0 prints.

Usage (same launch incantation as train.py):

    accelerate launch \
        --config_file configs/libero/reproduce_pi05_libero_accelerate_config.yaml \
        src/opentau/scripts/profile_step.py \
        --config_path=configs/libero/reproduce_pi05_libero.json \
        --steps=250

The ``steps`` field in TrainPipelineConfig is reused as the measurement
length; 20 warmup + 200 measured is a good default, so ``--steps=220`` is
a reasonable setting. Anything larger just runs longer.
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

    accelerator_kwargs: dict[str, Any] = {
        "step_scheduler_with_optimizer": False,
        "split_batches": False,
        "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=True)],
    }
    if cfg.gradient_accumulation_steps > 1:
        accelerator_kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    init_logging(accelerator, level=logging.INFO)
    set_proc_accelerator(accelerator)

    measure_steps = max(cfg.steps - WARMUP_STEPS, DEFAULT_MEASURE_STEPS)
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
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    train_dataloader = train_dataset.get_dataloader()
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
        loss = (
            cfg.loss_weighting["MSE"] * losses["MSE"] + cfg.loss_weighting["CE"] * losses["CE"]
        )
        _sync()
        t2 = time.perf_counter()

        # Phase 3: backward + step (lines 79-92)
        accelerator.backward(loss)
        accelerator.unscale_gradients(optimizer=optimizer)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
        optimizer.step()
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
        print(f"batch_size={cfg.batch_size} num_workers={cfg.num_workers} "
              f"prefetch_factor={cfg.prefetch_factor}")
        print(f"wall-clock over full loop: {loop_wall:.2f}s")
        print()
        print(f"{'phase':<16} {'stats':<60} {'share':>8}")
        print("-" * 90)
        total_mean = mean(phases["total"]) if phases["total"] else 0.0
        for key in ["dataload_wait", "forward", "backward_step", "sync_gather", "total"]:
            vals = phases[key]
            share = (mean(vals) / total_mean * 100.0) if vals and total_mean > 0 else 0.0
            marker = "  <-- total" if key == "total" else f"{share:6.1f}%"
            print(f"{key:<16} {_fmt_ms(vals):<60} {marker}")
        print()
        steps_per_sec = 1.0 / total_mean if total_mean > 0 else 0.0
        samples_per_sec = steps_per_sec * cfg.batch_size * accelerator.num_processes
        print(f"throughput: {steps_per_sec:.2f} steps/s, "
              f"{samples_per_sec:.1f} global samples/s")
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
