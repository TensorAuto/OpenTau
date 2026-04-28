#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dataloader-only throughput ceiling for the training pipeline.

Companion to ``profile_step.py`` that answers "is the dataloader keeping
up with the GPUs?" Builds the exact same
``WeightedDatasetMixture.get_dataloader()`` the training loop uses (same
``num_workers``, ``prefetch_factor``, ``pin_memory``, ``HierarchicalSampler``)
and iterates batches in a tight loop with **no model, no optimizer, no
collective**. Any slowdown observed here is pure input-pipeline cost.

Compare the ``batches/s`` reported here against the ``steps/s`` from
``profile_step.py``. If dataloader throughput is at or below training
step rate, the input pipeline is the bottleneck.

Run with the same launcher you use for training so you reproduce the
real multi-rank × N-worker pressure on the host CPU:

    accelerate launch \
        --config_file configs/libero/reproduce_pi05_libero_accelerate_config.yaml \
        src/opentau/scripts/profile_dataloader.py \
        --config_path=configs/libero/reproduce_pi05_libero.json

Each rank prints its own throughput; rank 0 also prints the cluster
min/mean/max. ``H2D`` timing moves the batch to the local GPU just like
``accelerator.prepare(dataloader)`` would — without any collective.

Environment variables (all optional):

  - ``PROFILE_BATCHES=N``  (default 300)  number of measured batches.
    ``cfg.steps`` is intentionally ignored (production configs set it
    to ~1M).
"""

import logging
import os
import time
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import cycle
from opentau.utils.utils import init_logging, is_launched_with_accelerate

WARMUP_BATCHES = 20
DEFAULT_MEASURE_BATCHES = 300


def _to_device(obj: Any, device: torch.device) -> Any:
    """Move tensors in a nested batch to ``device`` (non_blocking, like pin_memory=True)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if device.type == "cuda" else obj
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
        out = [_to_device(v, device) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    return obj


def _batch_size_of(obj: Any) -> int | None:
    """Best-effort batch size for a nested batch dict/list/tensor."""
    if isinstance(obj, torch.Tensor):
        return int(obj.shape[0]) if obj.ndim > 0 else None
    if isinstance(obj, Mapping):
        for v in obj.values():
            bs = _batch_size_of(v)
            if bs is not None:
                return bs
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        for v in obj:
            bs = _batch_size_of(v)
            if bs is not None:
                return bs
    return None


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@parser.wrap()
def profile(cfg: TrainPipelineConfig):
    cfg.validate()

    # Optional accelerate context for launched-under-accelerate use. We do NOT
    # call accelerator.prepare on the dataloader — we want the raw DataLoader
    # the training run uses, one instance per rank, so worker counts match.
    try:
        import accelerate

        accelerator = accelerate.Accelerator()
    except Exception:
        accelerator = None

    init_logging(accelerator, level=logging.INFO)

    rank = accelerator.process_index if accelerator is not None else 0
    world_size = accelerator.num_processes if accelerator is not None else 1
    is_main = rank == 0
    device = (
        accelerator.device
        if accelerator is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    # We intentionally ignore cfg.steps here (production configs set it to
    # ~1M). Override with PROFILE_BATCHES=<N> if you want a longer sample.
    measure_batches = int(os.environ.get("PROFILE_BATCHES", DEFAULT_MEASURE_BATCHES))
    if is_main:
        logging.info(
            "profile_dataloader: warmup=%d, measured=%d, world_size=%d, batch_size=%d, "
            "num_workers=%d, prefetch_factor=%s, pin_memory=%s",
            WARMUP_BATCHES,
            measure_batches,
            world_size,
            cfg.batch_size,
            cfg.num_workers,
            cfg.prefetch_factor,
            torch.cuda.is_available(),
        )

    if cfg.val_freq > 0:
        train_dataset, _ = make_dataset_mixture(cfg)
    else:
        train_dataset = make_dataset_mixture(cfg)

    train_dataloader = train_dataset.get_dataloader()
    it = cycle(train_dataloader)

    fetch_times: list[float] = []
    h2d_times: list[float] = []
    observed_batch_size: int | None = None

    total_batches = WARMUP_BATCHES + measure_batches
    loop_start = time.perf_counter()

    for i in range(total_batches):
        measuring = i >= WARMUP_BATCHES

        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        batch = _to_device(batch, device)
        _sync(device)
        t2 = time.perf_counter()

        if observed_batch_size is None:
            observed_batch_size = _batch_size_of(batch)

        if measuring:
            fetch_times.append(t1 - t0)
            h2d_times.append(t2 - t1)

        if is_main and (i + 1) % 50 == 0:
            logging.info(
                "batch %d/%d (last fetch=%.1fms, h2d=%.1fms)",
                i + 1,
                total_batches,
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
            )

    loop_wall = time.perf_counter() - loop_start

    def _summary(xs):
        if not xs:
            return "n/a"
        xs_sorted = sorted(xs)
        return (
            f"mean={mean(xs) * 1000:7.2f}ms "
            f"median={median(xs) * 1000:7.2f}ms "
            f"p95={xs_sorted[int(0.95 * (len(xs_sorted) - 1))] * 1000:7.2f}ms"
        )

    local_fetch_mean = mean(fetch_times) if fetch_times else 0.0
    local_bps = 1.0 / local_fetch_mean if local_fetch_mean > 0 else 0.0

    # Per-rank print so we can see stragglers.
    print(
        f"[rank {rank}/{world_size}] "
        f"fetch={_summary(fetch_times)} | h2d={_summary(h2d_times)} | "
        f"batches/s={local_bps:.2f} | "
        f"samples/s={local_bps * (observed_batch_size or cfg.batch_size):.1f}"
    )

    # Aggregate across ranks.
    if accelerator is not None and world_size > 1:
        local_t = torch.tensor([local_fetch_mean], device=device, dtype=torch.float64)
        gathered = accelerator.gather(local_t).cpu().tolist()
    else:
        gathered = [local_fetch_mean]

    if is_main:
        bps_per_rank = [1.0 / t if t > 0 else 0.0 for t in gathered]
        print("\n=========== profile_dataloader summary (rank 0) ===========")
        print(
            f"world_size={world_size} batch_size={cfg.batch_size} "
            f"num_workers={cfg.num_workers} prefetch_factor={cfg.prefetch_factor}"
        )
        print(f"wall-clock over full loop: {loop_wall:.2f}s")
        print(
            f"per-rank batches/s (min / mean / max): "
            f"{min(bps_per_rank):.2f} / {sum(bps_per_rank) / len(bps_per_rank):.2f} / "
            f"{max(bps_per_rank):.2f}"
        )
        total_samples_per_sec = sum(bps_per_rank) * (observed_batch_size or cfg.batch_size)
        print(f"cluster-wide samples/s (ceiling, no model): {total_samples_per_sec:.1f}")
        print("===========================================================\n")


if __name__ == "__main__":
    # Accelerate-launch is recommended so worker counts match production, but we
    # also allow single-process for a per-rank ceiling.
    if not is_launched_with_accelerate():
        print(
            "NOTE: running outside accelerate — this measures single-rank "
            "dataloader throughput only. Launch with `accelerate launch ...` "
            "to reproduce the real worker-contention pattern."
        )
    profile()
