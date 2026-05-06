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
  7. ``sync_gather``      — the 5 ``gather_for_metrics(...).item()`` calls

A ``backward_step`` aggregate (= phases 3+4+5+6) is also collected and
emitted into the JSON dump for backward-compatibility with older
reports, but is omitted from the human-readable phase table to avoid
double-counting against its components.

All ranks call ``torch.cuda.synchronize()`` at phase boundaries so the
reported times include collective + H2D sync costs.

Throughput is reported as a *global* number using the slowest-rank step
time (gather across ranks, take ``max``); collectives gate on the
slowest rank, so that's the true ceiling. Peak GPU memory is also
tracked per-rank (``torch.cuda.max_memory_{allocated,reserved}``)
across the measured loop only — peak stats are reset at the
warmup→measured transition. The worst-rank values are reported.

NOTE: under DeepSpeed, optim work fuses into backward hooks, so
``optim_step``/``unscale_clip`` will appear near-zero (tens of µs)
while FSDP/DDP show real time there. Only ``total`` step time is
directly comparable across backends.

The script does NOT use ``with accelerator.accumulate(policy):``, so
``cfg.gradient_accumulation_steps`` must be 1 (asserted at start);
sweep ``dataloader_batch_size`` for per-rank batch.

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
from opentau.optim.master_weights import MasterWeightOptimizer
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

    # The loop below has no `with accelerator.accumulate(policy):` context, so
    # optimizer.step() runs every iteration regardless of cfg.gradient_accumulation_steps.
    # That makes ga>1 silently incorrect: optim fires every micro-step (not every ga
    # micro-steps) and the throughput math at the bottom would over-report by ga.
    # Sweep `dataloader_batch_size` for per-rank batch instead.
    assert cfg.gradient_accumulation_steps == 1, (
        "profile_step.py only supports gradient_accumulation_steps=1; "
        f"got {cfg.gradient_accumulation_steps}. Pass --gradient_accumulation_steps=1 "
        "on the CLI to override the accelerate yaml default."
    )

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

    # Optional: override the policy's attention_implementation before the
    # policy is constructed. Lets us A/B eager vs sdpa without touching the
    # training config JSON. Recognised values match the pi05 validator:
    # "eager", "sdpa", "fa2" (the last falls back to eager with a warning).
    attn_impl_env = os.environ.get("ATTENTION_IMPL")
    if attn_impl_env is not None and hasattr(cfg.policy, "attention_implementation"):
        if accelerator.is_main_process:
            logging.info(
                "ATTENTION_IMPL=%s: overriding cfg.policy.attention_implementation (was %r)",
                attn_impl_env,
                cfg.policy.attention_implementation,
            )
        cfg.policy.attention_implementation = attn_impl_env

    # Optional: toggle gradient checkpointing per run. Production default is
    # False; setting GRAD_CHECKPOINT=true lets us A/B the throughput/memory
    # tradeoff without touching the training config JSON. The strict
    # distributed-backend guard in train.py is *not* duplicated here because
    # profile_step already runs under accelerator and pi05's custom forward
    # only supports the same set of backends for ckpt; if users try it
    # under an unsupported backend they'll hit the same autograd issues at
    # first backward.
    grad_ckpt_env = os.environ.get("GRAD_CHECKPOINT")
    if grad_ckpt_env is not None and hasattr(cfg.policy, "gradient_checkpointing"):
        want_ckpt = grad_ckpt_env.lower() == "true"
        if accelerator.is_main_process:
            logging.info(
                "GRAD_CHECKPOINT=%s: overriding cfg.policy.gradient_checkpointing (was %r)",
                grad_ckpt_env,
                cfg.policy.gradient_checkpointing,
            )
        cfg.policy.gradient_checkpointing = want_ckpt

    # Mirror train.py: under DeepSpeed ZeRO-3 we must disable the per-construction
    # parameter partitioning (it shards before init, breaking shape-dependent
    # initializers like SigLIP's lecun_normal_). No-op for any non-ZeRO-3 backend.
    from opentau.scripts.train import _zero3_disabled_init_context

    # Mirror train.py: under FSDP, gate the model's internal bf16 cast so the
    # policy stays fp32 for FSDP's MixedPrecision to manage the bf16 compute /
    # fp32 master split. Without this, AdamW would silently allocate Adam state
    # in bf16 (matches the bf16-cast params) and the throughput numbers would
    # not represent the production fp32-Adam regime.
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        vlm_config = getattr(cfg.policy, "vlm_config", None)
        if vlm_config is not None and hasattr(vlm_config, "disable_internal_bf16_cast"):
            vlm_config.disable_internal_bf16_cast = True

    with _zero3_disabled_init_context(accelerator):
        policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    # Same per-backend cast logic as train.py: skip the outer bf16 cast under
    # FSDP (the policy must stay fp32 going into accelerate.prepare; FSDP
    # MixedPrecision provides bf16 compute on the fly).
    if accelerator.distributed_type != accelerate.DistributedType.FSDP:
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

        # Mirror train.py's master-weights wrapping. Without this, profile_step
        # silently allocates Adam state (exp_avg, exp_avg_sq) in bf16 since the
        # optimizer is built over the bf16-cast policy parameters; the resulting
        # memory footprint is ~half of what production training pays under the
        # PR #182 fix, so the benchmark would systematically under-report
        # memory and over-report the largest batch that fits. See issue #181.
        # DeepSpeed ZeRO provides equivalent fp32-master semantics via
        # BF16_Optimizer.
        # FSDP runs with fp32 master + fp32 Adam state in this PR: the
        # ``disable_internal_bf16_cast`` flag set above (lines 215-218) gates
        # the model's internal ``to_bfloat16_like_physical_intelligence``
        # cast, so by the time ``accelerator.prepare`` wraps the policy under
        # FSDP, parameters are still fp32. The optimizer is therefore built
        # over fp32 outer params, and FSDP's ``MixedPrecision(param_dtype=bf16,
        # reduce_dtype=bf16, buffer_dtype=bf16)`` (what accelerate translates
        # ``mixed_precision: bf16`` into when no ``fsdp_reduce_dtype`` /
        # ``fsdp_buffer_dtype`` overrides are set) does only the compute-time
        # downcast — the fp32 outer master shards are preserved across forward
        # and backward, and AdamW steps on them. Adam state is fp32 (matches
        # the production regime). Wrapping with ``MasterWeightOptimizer`` *on
        # top* of FSDP would still misalign with FSDP's flat-param parameter
        # handles — every wrap clone breaks the 1:1 handle ↔ stored-param
        # identity that FSDP relies on for its all-gather hooks, observed
        # empirically as a NCCL desync during the first backward. Since FSDP
        # already gives us fp32 master semantics via the fp32-built outer
        # params, the wrap is also redundant. Skip it.
        if accelerator.distributed_type not in (
            accelerate.DistributedType.DEEPSPEED,
            accelerate.DistributedType.FSDP,
        ):
            optimizer = MasterWeightOptimizer.from_existing(optimizer)
            # Same rebind train.py performs immediately after from_existing:
            # ``make_optimizer_and_scheduler`` left ``lr_scheduler.optimizer``
            # pointing at the original (now-orphaned) AdamW. Without this,
            # ``lr_scheduler.step()`` would mutate the orphan's
            # ``param_groups[i]['lr']`` and the wrapper's inner AdamW would
            # never see the schedule. See PR #182 and 8be2cd1.
            if lr_scheduler is not None:
                lr_scheduler.optimizer = optimizer
            if accelerator.is_main_process:
                logging.info(
                    "Wrapped optimizer with MasterWeightOptimizer (fp32 master "
                    "weights + fp32 Adam state). Backend: %s.",
                    accelerator.distributed_type,
                )

    train_dataloader = train_dataset.get_dataloader()
    if skip_optim:
        policy, train_dataloader = accelerator.prepare(policy, train_dataloader)
    else:
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
        # accelerator.prepare may have migrated the policy's bf16 params from
        # CPU to GPU. The MasterWeightOptimizer's fp32 masters were cloned
        # at wrap-time (still on CPU) and would otherwise stay there, making
        # this benchmark report misleadingly low GPU memory. Re-build masters
        # from the now-migrated live params so they live on the same device.
        # No-op when masters are already on the right device.
        inner_opt_for_migrate = getattr(optimizer, "optimizer", optimizer)
        if isinstance(inner_opt_for_migrate, MasterWeightOptimizer):
            inner_opt_for_migrate.rebuild_masters_from_live(policy.parameters())
    train_dl_iter = cycle(train_dataloader)

    policy.train()

    phases = defaultdict(list)

    total_steps = WARMUP_STEPS + measure_steps
    loop_start = time.perf_counter()

    for step in range(total_steps):
        measuring = step >= WARMUP_STEPS

        # Reset CUDA peak-memory stats at the warmup→measured transition so the
        # final ``max_memory_allocated/reserved`` readings reflect only the
        # measured loop (not warmup spikes from cuDNN autotune / first-step
        # FSDP all-gather staging buffers).
        if step == WARMUP_STEPS and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
                # Mirror train.py's dispatch: when MasterWeightOptimizer is in
                # use, route the clip through it so the bf16->fp32 grad upcast
                # is amortized into the clip phase (and the subsequent step
                # skips the upcast). Otherwise use accelerator's clip path.
                inner_opt = getattr(optimizer, "optimizer", optimizer)
                if isinstance(inner_opt, MasterWeightOptimizer):
                    inner_opt.clip_grad_norm_(cfg.optimizer.grad_clip_norm)
                else:
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

    # All-rank aggregation. Must run on every rank (collective ops), so it lives
    # outside the rank-0 print block. We compute a global step time as the *max*
    # across ranks (slowest rank gates collectives, so global throughput is
    # bounded by it), and gather peak HBM so the worst-rank ceiling is visible.
    local_total_mean = mean(phases["total"]) if phases["total"] else 0.0
    if torch.cuda.is_available():
        device = accelerator.device
        local_peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        local_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        gather_tensor = torch.tensor(
            [local_total_mean, local_peak_alloc_gb, local_peak_reserved_gb],
            dtype=torch.float64,
            device=device,
        )
        gathered = accelerator.gather(gather_tensor.unsqueeze(0)).cpu()
        per_rank_total_mean = gathered[:, 0].tolist()
        per_rank_peak_alloc_gb = gathered[:, 1].tolist()
        per_rank_peak_reserved_gb = gathered[:, 2].tolist()
    else:
        per_rank_total_mean = [local_total_mean]
        per_rank_peak_alloc_gb = [0.0]
        per_rank_peak_reserved_gb = [0.0]
    global_total_mean = max(per_rank_total_mean) if per_rank_total_mean else 0.0
    worst_peak_alloc_gb = max(per_rank_peak_alloc_gb) if per_rank_peak_alloc_gb else 0.0
    worst_peak_reserved_gb = max(per_rank_peak_reserved_gb) if per_rank_peak_reserved_gb else 0.0

    if accelerator.is_main_process:
        is_deepspeed = accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED
        print("\n=========== profile_step results ===========")
        print(f"warmup={WARMUP_STEPS} measured={measure_steps} ranks={accelerator.num_processes}")
        print(
            f"batch_size={cfg.batch_size} num_workers={cfg.num_workers} prefetch_factor={cfg.prefetch_factor}"
        )
        print(f"distributed_type={accelerator.distributed_type}")
        print(f"wall-clock over full loop (rank 0): {loop_wall:.2f}s")
        if is_deepspeed:
            print(
                "NOTE: under DeepSpeed, optim cost is accounted inside the bwd phase "
                "(engine attaches optim to backward hooks). optim_step/unscale_clip "
                "will look near-zero; compare TOTAL step time across backends, not "
                "per-phase breakdowns."
            )
        print()
        print(f"{'phase':<16} {'stats (rank 0)':<60} {'share':>8}")
        print("-" * 90)
        rank0_total_mean = local_total_mean  # share % is rank-0-relative for legibility
        # backward_step is intentionally omitted from the print table — it's an
        # aggregate of bwd+unscale_clip+optim_step+zero_grad_sched and would
        # double-count if printed alongside its components. Still kept in the
        # JSON output below for backward compatibility with earlier reports.
        ordered_keys = [
            "dataload_wait",
            "forward",
            "bwd",
            "unscale_clip",
            "optim_step",
            "zero_grad_sched",
            "sync_gather",
            "total",
        ]
        for key in ordered_keys:
            vals = phases[key]
            share = (mean(vals) / rank0_total_mean * 100.0) if vals and rank0_total_mean > 0 else 0.0
            marker = "  <-- total" if key == "total" else f"{share:6.1f}%"
            print(f"{key:<16} {_fmt_ms(vals):<60} {marker}")
        print()
        # Global throughput: use slowest-rank step time (collectives gate on it).
        steps_per_sec = 1.0 / global_total_mean if global_total_mean > 0 else 0.0
        samples_per_sec = steps_per_sec * cfg.batch_size * accelerator.num_processes
        print("throughput (global, gated by slowest rank):")
        print(f"  global step time:  {global_total_mean * 1000.0:7.2f} ms")
        print(f"  steps/s:           {steps_per_sec:.3f}")
        print(
            f"  global samples/s:  {samples_per_sec:.1f}  "
            f"(= {cfg.batch_size} per-rank batch * {accelerator.num_processes} ranks / step time)"
        )
        print(
            f"  per-rank step time means (ms): "
            f"[{', '.join(f'{x * 1000.0:.2f}' for x in per_rank_total_mean)}]"
        )
        print()
        print("peak GPU memory (worst rank):")
        print(f"  allocated:   {worst_peak_alloc_gb:6.2f} GB")
        print(f"  reserved:    {worst_peak_reserved_gb:6.2f} GB")
        print(f"  per-rank allocated (GB): [{', '.join(f'{x:.2f}' for x in per_rank_peak_alloc_gb)}]")
        print(f"  per-rank reserved (GB):  [{', '.join(f'{x:.2f}' for x in per_rank_peak_reserved_gb)}]")
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
                        "distributed_type": str(accelerator.distributed_type),
                        "batch_size": cfg.batch_size,
                        "num_workers": cfg.num_workers,
                        "prefetch_factor": cfg.prefetch_factor,
                        "phase_means_s": {k: mean(v) for k, v in phases.items() if v},
                        "phase_medians_s": {k: median(v) for k, v in phases.items() if v},
                        "global_step_time_s": global_total_mean,
                        "global_samples_per_sec": (
                            (1.0 / global_total_mean) * cfg.batch_size * accelerator.num_processes
                            if global_total_mean > 0
                            else 0.0
                        ),
                        "per_rank_step_time_s": per_rank_total_mean,
                        "peak_alloc_gb_worst": worst_peak_alloc_gb,
                        "peak_reserved_gb_worst": worst_peak_reserved_gb,
                        "per_rank_peak_alloc_gb": per_rank_peak_alloc_gb,
                        "per_rank_peak_reserved_gb": per_rank_peak_reserved_gb,
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
