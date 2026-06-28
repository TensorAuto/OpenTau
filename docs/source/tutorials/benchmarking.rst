Benchmarking and Profiling
==========================

OpenTau ships three diagnostic scripts under ``src/opentau/scripts/`` that
together pin down where a training run is spending its time. All three read
the same ``TrainPipelineConfig`` as ``opentau-train``, so you can point them
at any training config JSON and reproduce the exact model / dataset / batch
size the real training uses.

+----------------------------+------------------------------------------------------------+
| Script                     | When to reach for it                                       |
+============================+============================================================+
| ``profile_step.py``        | Measure per-step wall-clock, broken down into forward /    |
|                            | backward / optimizer / sync phases. Use this first when    |
|                            | asking "why is training slow?"                             |
+----------------------------+------------------------------------------------------------+
| ``profile_dataloader.py``  | Measure the dataloader-only throughput ceiling (no model,  |
|                            | no collective). Use this to rule out input-pipeline        |
|                            | starvation before looking at the GPU.                      |
+----------------------------+------------------------------------------------------------+
| ``find_unused_params.py``  | List parameters that receive no gradient during a real     |
|                            | forward+backward. Use before setting                       |
|                            | ``FIND_UNUSED_PARAMS=false`` on a new policy.              |
+----------------------------+------------------------------------------------------------+

A worked example end-to-end — diagnosing a real "low GPU utilization" issue,
ruling out dataloader starvation, and confirming the bottleneck is
DeepSpeed's per-parameter hook overhead — is documented in
`issue #177 <https://github.com/TensorAuto/OpenTau/issues/177>`_.

profile_step.py — per-step timing breakdown
-------------------------------------------

Mirrors ``opentau-train``'s setup (Accelerator, dataset mixture, policy,
optimizer, LR scheduler) and runs a short loop. Splits per-step
wall-clock into eight phases and reports mean / median / p95 for each.

Basic usage — same launch incantation as ``opentau-train``:

.. code-block:: bash

    accelerate launch \
        --config_file configs/examples/accelerate_ddp_config.yaml \
        src/opentau/scripts/profile_step.py \
        --config_path=configs/libero/reproduce_pi05_libero.json \
        --batch_size=12

Example output (8×A100, DDP, pi05 / LIBERO at ``bs=12``):

.. code-block:: text

    =========== profile_step results (rank 0) ===========
    warmup=20 measured=200 ranks=8
    batch_size=12 num_workers=16 prefetch_factor=8
    wall-clock over full loop: 265.89s

    phase            stats                                                           share
    ------------------------------------------------------------------------------------------
    dataload_wait    mean=   1.24ms  median=   1.18ms  p95=   1.60ms                 0.1%
    forward          mean= 378.86ms  median= 378.65ms  p95= 381.75ms                32.8%
    bwd              mean= 706.32ms  median= 704.74ms  p95= 710.24ms                61.1%
    unscale_clip     mean=  19.55ms  median=  19.43ms  p95=  21.12ms                 1.7%
    optim_step       mean=  39.22ms  median=  39.17ms  p95=  39.81ms                 3.4%
    zero_grad_sched  mean=   1.79ms  median=   1.74ms  p95=   2.74ms                 0.2%
    backward_step    mean= 766.89ms  median= 765.58ms  p95= 770.31ms                66.3%
    sync_gather      mean=   9.64ms  median=   6.47ms  p95=   9.78ms                 0.8%
    total            mean=1156.63ms  median=1151.73ms  p95=1158.28ms                <-- total

    throughput: 0.86 steps/s, 83.0 global samples/s
    =====================================================

Reading the output:

- **High share in** ``dataload_wait`` (> ~5%) means the dataloader is not
  keeping up with the GPUs. Run ``profile_dataloader.py`` to confirm and
  raise ``num_workers`` / set ``persistent_workers=True`` /
  ``enable_cpu_affinity=True`` as appropriate.
- **High share in** ``bwd`` **with a large gap between 1-GPU and N-GPU**
  typically indicates distributed-backend overhead. Try a single-GPU run
  (``PROFILE_NO_OPTIM=1``) for comparison — if single-GPU ``bwd`` is
  close to ~2× forward and N-GPU is much larger, the delta is host-side
  work, not compute.
- **High share in** ``optim_step`` is normal for large models; if > 10%
  and you're on CUDA, make sure ``AdamConfig.fused`` is ``True`` (the
  default since `PR #176 <https://github.com/TensorAuto/OpenTau/pull/176>`_).
- **High share in** ``sync_gather`` points at the
  ``accelerator.gather_for_metrics(...).item()`` calls in
  ``update_policy``. They run every step (not just every ``log_freq``)
  and can be gated behind ``log_freq`` if they become a bottleneck.

Environment variables (all optional):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Effect
   * - ``PROFILE_STEPS``
     - 200
     - Number of measured steps (after 20 warmup). ``cfg.steps`` from
       the training config is intentionally ignored because production
       configs typically set it to 1M.
   * - ``PROFILE_NO_OPTIM``
     - 0
     - When ``1``, skip optimizer creation and ``optimizer.step`` /
       ``zero_grad`` entirely. Useful for isolating raw forward +
       backward compute on a single GPU (no Adam state allocated, so
       a large bf16 model fits without ZeRO partitioning).
   * - ``FIND_UNUSED_PARAMS``
     - true
     - Toggles DDP's ``find_unused_parameters`` kwarg. Silently
       ignored under DeepSpeed. Set to ``false`` after auditing the
       policy with ``find_unused_params.py``.
   * - ``FUSED_ADAMW``
     - (unset)
     - When ``true`` / ``false``, force-toggle
       ``torch.optim.AdamW(fused=...)`` for an A/B comparison without
       editing the optimizer config JSON. Unset leaves the factory
       default alone.
   * - ``PROFILE_STEP_JSON``
     - (unset)
     - When set to a file path, rank 0 writes a JSON summary of phase
       means and medians after the loop. Convenient for scripted A/B
       sweeps.

profile_dataloader.py — dataloader throughput ceiling
-----------------------------------------------------

Builds the exact same ``WeightedDatasetMixture.get_dataloader()`` the
training loop uses (``num_workers``, ``prefetch_factor``, ``pin_memory``,
``HierarchicalSampler``) and iterates batches with **no model, no
optimizer, no collective**. Any slowdown here is pure input-pipeline
cost.

Run under the same launcher as training so the host CPU sees the real
multi-rank × N-worker pressure:

.. code-block:: bash

    accelerate launch \
        --config_file configs/examples/accelerate_ddp_config.yaml \
        src/opentau/scripts/profile_dataloader.py \
        --config_path=configs/libero/reproduce_pi05_libero.json

Example output (8 ranks):

.. code-block:: text

    [rank 0/8] fetch=mean=  80.76ms median=   0.20ms p95= 605.39ms | h2d=mean= 0.40ms ...
    [rank 1/8] fetch=mean=  83.40ms median=   0.16ms p95= 754.82ms | h2d=mean= 0.49ms ...
    ... one line per rank ...

    =========== profile_dataloader summary (rank 0) ===========
    world_size=8 batch_size=12 num_workers=16 prefetch_factor=8
    wall-clock over full loop: 27.23s
    per-rank batches/s (min / mean / max): 11.99 / 12.45 / 12.93
    cluster-wide samples/s (ceiling, no model): 597.6
    ===========================================================

Reading the output:

- **Compare** ``cluster-wide samples/s`` **against the** ``samples/s``
  **from** ``profile_step.py``. If dataloader throughput is at or below
  the training step rate, the input pipeline is your bottleneck.
  If dataloader throughput is comfortably ahead, the bottleneck is
  GPU-side (forward / backward / optim).
- **Bimodal fetch distribution** (median ≈ 0 ms, p95 ≈ hundreds of ms)
  means you're alternately hitting the prefetch buffer and blocking
  on worker decode. That's normal — only the mean matters for
  long-run throughput.

Environment variables:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Effect
   * - ``PROFILE_BATCHES``
     - 300
     - Number of measured batches after 20 warmup. Raise if p95 is
       high and you want a more stable mean.

find_unused_params.py — list parameters DDP would reject
--------------------------------------------------------

Runs one forward + backward on a real batch on a single GPU (no DDP, no
DeepSpeed) and prints every parameter where ``param.requires_grad``
is ``True`` but ``param.grad`` is ``None`` after backward. Those are
exactly the parameters DDP would refuse to sync with
``find_unused_parameters=False``.

Run as a plain Python invocation — no ``accelerate launch`` needed:

.. code-block:: bash

    python src/opentau/scripts/find_unused_params.py \
        --config_path=configs/libero/reproduce_pi05_libero.json

Example output (pi05, after the PR that dropped ``gemma_expert.lm_head``):

.. code-block:: text

    #==============================================================================
    # pi05 parameter audit — single forward + backward, single GPU
    # include_zero_grad=False
    #==============================================================================

    ========== UNUSED (requires_grad=True, grad is None) — DDP will refuse without
                      find_unused_parameters=True (0 tensors, 0 params) ==========

    ========== FROZEN (requires_grad=False) — context (8 tensors, 256 params) ==========
      [normalize_discrete_actions.buffer_actions.max]  (1 tensors, 32 params)
        - normalize_discrete_actions.buffer_actions.max  shape=(32,)
      ...

    # USED (requires_grad=True, grad is non-trivial): 814 tensors
    # Tip: if UNUSED list is empty, you can flip
    #  DistributedDataParallelKwargs(find_unused_parameters=False) safely.

Recommended workflow:

1. Run ``find_unused_params.py`` on your policy.
2. If ``UNUSED`` is empty, set ``FIND_UNUSED_PARAMS=false`` when launching
   ``opentau-train`` (or drop the ``DistributedDataParallelKwargs``
   kwarg in ``train.py`` for your fork) to reclaim the per-step
   graph-walk cost.
3. If ``UNUSED`` is non-empty, each reported tensor is either an orphan
   in the model graph (fix by freezing it or deleting the module) or
   a parameter that's only conditionally reached (fix by adding an
   unconditional graph edge, e.g. ``+ 0 * unused_param.sum()`` in the
   loss).

Environment variables:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Effect
   * - ``INCLUDE_ZERO_GRAD``
     - false
     - When ``true``, also list parameters whose grad tensor exists
       but is all-zero. These are typically paths touched on some
       batches but producing no learning signal on this one. Usually
       safe to ignore; useful for auditing conditionally-active heads.

Example: a typical benchmarking session
---------------------------------------

Given low ``samples/s`` in a training run, a sensible sequence to rule
out candidates in order of likelihood:

.. code-block:: bash

    # 1. Is the dataloader keeping up? (~2-10 minutes)
    accelerate launch \
        --config_file configs/examples/accelerate_ddp_config.yaml \
        src/opentau/scripts/profile_dataloader.py \
        --config_path=<your_config.json>

    # 2. Where does per-step time go? (~4 minutes at 200 steps)
    accelerate launch \
        --config_file configs/examples/accelerate_ddp_config.yaml \
        src/opentau/scripts/profile_step.py \
        --config_path=<your_config.json> \
        --batch_size=<your_bs>

    # 3. Is DDP's find_unused_parameters=True costing you? Only if
    #    backward_step looks unusually high. (~1 minute, single GPU)
    python src/opentau/scripts/find_unused_params.py \
        --config_path=<your_config.json>

    # 4. A/B an optimizer or distributed-backend change without
    #    touching any config file:
    FUSED_ADAMW=false accelerate launch ... profile_step.py ...
    FUSED_ADAMW=true  accelerate launch ... profile_step.py ...

DeepSpeed ZeRO-2 vs ZeRO-3 for pi05 full fine-tuning
----------------------------------------------------

ZeRO-3 shards the model *parameters* across ranks on top of the gradient and
optimizer-state sharding that ZeRO-2 already does. That extra sharding pays off
only when a single replica of the model does not fit in one GPU's memory — it
adds a per-layer parameter all-gather in the forward (and a matching
reduce-scatter in the backward) that ZeRO-2 does not need. pi05 is ≈3.3B
parameters and fits comfortably replicated on an 80 GB GPU, so ZeRO-3 has
nothing to gain and pays the all-gather cost.

Measured on **8×A100-80GB, full fine-tuning (no frozen weights;
``freeze_vision_encoder=false``, ``train_expert_only=false``), bf16, sdpa
attention, ``use_torch_compile=false``, ``gradient_accumulation_steps=1``, 8
ranks**, with the ``configs/examples/accelerate_deepspeed*`` configs and the
pi05 reference policy (2 cameras at 224×224, ``chunk_size=10``,
``predict_response=true``) on ``TensorAuto/libero``:

.. list-table::
   :header-rows: 1
   :widths: 14 14 12 12 14 14

   * - Backend
     - Per-rank batch
     - Global batch
     - sec/step
     - samples/s
     - Peak GPU mem
   * - ZeRO-2
     - 8
     - 64
     - 5.64
     - 11.4
     - 53.3 GiB
   * - ZeRO-3
     - 8
     - 64
     - 7.71
     - 8.3
     - 62.8 GiB
   * - ZeRO-2
     - 16
     - 128
     - 5.29
     - 24.2
     - 78.7 GiB
   * - ZeRO-3
     - 16
     - 128
     - 9.86
     - 13.0
     - 79.2 GiB

Both backends OOM at the same per-rank batch size on this hardware (16 fits,
18 OOMs): ZeRO-3 frees ~5 GB of replicated parameters per rank, but its
parameter all-gather/prefetch buffers plus the extra allocator fragmentation
consume a comparable amount, so the maximum batch is unchanged. At a matched
batch size ZeRO-2 is ~1.4× faster at batch 8 and ~1.9× faster at batch 16
(the per-step parameter all-gather is the difference; both keep fp32 master
weights and step the optimizer identically).

**Recommendation:** use plain DDP (fastest) or ZeRO-2 for pi05 and similarly
sized policies. Reach for ZeRO-3 only when a single replica no longer fits per
GPU (much larger backbones / many-billion-parameter experts). ZeRO-3 *is* fully
supported and validated for pi05 — training, checkpoint save, resume, offline
checkpoint consolidation (``convert_checkpoint.sh``), and in-training validation
all work — it is simply not the throughput-optimal choice at this model size.

To reproduce, run the same config under each accelerate file (both at 8 ranks /
``gradient_accumulation_steps=1``) and read the per-step time from the logs;
``samples/s = per_rank_batch × num_ranks ÷ sec_per_step``:

.. code-block:: bash

    COMMON="--policy.freeze_vision_encoder=false --policy.train_expert_only=false \
        --policy.use_torch_compile=false --policy.attention_implementation=sdpa \
        --batch_size=16 --dataloader_batch_size=16 --gradient_accumulation_steps=1"

    # ZeRO-2
    accelerate launch --config_file configs/examples/accelerate_deepspeed_config.yaml --num_processes 8 \
        src/opentau/scripts/train.py --config_path=configs/examples/pi05_training_config.json $COMMON

    # ZeRO-3
    accelerate launch --config_file configs/examples/accelerate_deepspeed_zero3_config.yaml \
        src/opentau/scripts/train.py --config_path=configs/examples/pi05_training_config.json $COMMON

.. note::
   If a ZeRO-3 run OOMs from fragmentation (the error mentions
   "reserved but unallocated" memory), set
   ``PYTORCH_ALLOC_CONF=expandable_segments:True`` in the environment to recover
   the fragmented blocks.
