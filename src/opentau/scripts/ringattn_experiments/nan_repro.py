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

"""Standalone reproducer for the step-2 NaN seen in real pi07 + ZeRO-2 runs.

Run with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.nan_repro

What this exercises
-------------------
* Real-shape (per-layer) Q/K/V tensors that match what pi07's interleaved
  decoder layer passes to ``ring_attention_forward`` under a small but
  realistic config (hidden_size matching the production Gemma 3 backbone,
  bf16 dtype, GQA head split).
* A pi07-style block-causal attention mask produced by
  :func:`make_att_2d_masks` over a sequence that includes real PAD rows
  (so the kernel sees fully-masked Q rows, the most likely NaN trigger).
* Forward + backward of ``_RingAttention.apply`` against random upstream
  ``grad_O``.

The script prints whether the forward output or any of dq/dk/dv contains
non-finite values, with min/max / NaN counts per output. ``RING_DEBUG_PROBES=1``
turns on the in-kernel finite checks for free.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from opentau.policies.pi07.low_level.modeling_pi07_low_level import make_att_2d_masks
from opentau.policies.pi07.ring_attention import _RingAttention


def _summary(name: str, t: torch.Tensor) -> str:
    finite = torch.isfinite(t)
    n_finite = int(finite.sum())
    n_total = int(finite.numel())
    if n_finite == n_total:
        return (
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4g} max={t.max().item():.4g} (all finite)"
        )
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"NON-FINITE {n_total - n_finite}/{n_total} elements "
        f"(NaN count={int((~finite & ~torch.isinf(t)).sum())})"
    )


def main() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Production-ish shapes (matches the Gemma 3 backbone of pi07 low-level).
    B = 4
    Hq = 8
    Hkv = 4
    Dh = 256
    # Sequence length close to what real pi07 hits (≈6500 tokens for a
    # 6-step, 4-cam, 448px prefix). Keep it divisible by world_size.
    S_total = 4096
    assert S_total % world_size == 0, "S_total must divide world_size"
    Sq_local = S_total // world_size

    # Block-causal mask with several blocks + a PAD tail, mirroring pi07's
    # actual prefix structure (image tokens | language tokens | state tokens |
    # action indicator | discrete actions | trailing pad).
    n_pad = 17  # awkward non-multiple-of-q_tile pad count to stress the loop
    pad_masks = torch.ones(B, S_total, dtype=torch.bool, device=device)
    pad_masks[:, -n_pad:] = False
    att_masks = torch.zeros(B, S_total, dtype=torch.int32, device=device)
    # Mimic pi07: image block (start at 0), language block, state block,
    # action indicator, discrete actions.
    block_starts = [0, 1024, 2048, 2560, 3072, 3584]
    for s in block_starts:
        att_masks[:, s] = 1
    attn_mask_full = make_att_2d_masks(pad_masks, att_masks)

    # Random Q/K/V on rank 0, broadcast so every rank has the same global tensors.
    # Scale up magnitudes a bit to mimic deep-layer activations after 30+ layers
    # of an untrained model — the conditions under which the production NaN
    # appeared. Real grad_O coming back through 34 layers' o_proj backward can
    # also have large magnitudes, hence 10x scale on grad_O.
    torch.manual_seed(0)
    scale_qkv = 5.0
    scale_grad = 10.0
    if rank == 0:
        q_full = scale_qkv * torch.randn(B, S_total, Hq, Dh, device=device, dtype=torch.bfloat16)
        k_full = scale_qkv * torch.randn(B, S_total, Hkv, Dh, device=device, dtype=torch.bfloat16)
        v_full = scale_qkv * torch.randn(B, S_total, Hkv, Dh, device=device, dtype=torch.bfloat16)
        grad_O_full = scale_grad * torch.randn(B, S_total, Hq, Dh, device=device, dtype=torch.bfloat16)
    else:
        q_full = torch.empty(B, S_total, Hq, Dh, device=device, dtype=torch.bfloat16)
        k_full = torch.empty(B, S_total, Hkv, Dh, device=device, dtype=torch.bfloat16)
        v_full = torch.empty(B, S_total, Hkv, Dh, device=device, dtype=torch.bfloat16)
        grad_O_full = torch.empty(B, S_total, Hq, Dh, device=device, dtype=torch.bfloat16)
    # ZERO OUT the padded Q rows in grad_O — under real training, downstream
    # loss masking sets grad_O for pad rows to 0. Replicating that here is
    # important: with grad_O random on pad rows, the kernel sees a different
    # numerical regime than production.
    grad_O_full[:, -n_pad:] = 0.0
    dist.broadcast(q_full, src=0)
    dist.broadcast(k_full, src=0)
    dist.broadcast(v_full, src=0)
    dist.broadcast(grad_O_full, src=0)

    # Shard along seq.
    q_local = torch.chunk(q_full, world_size, dim=1)[rank].contiguous().requires_grad_(True)
    k_local = torch.chunk(k_full, world_size, dim=1)[rank].contiguous().requires_grad_(True)
    v_local = torch.chunk(v_full, world_size, dim=1)[rank].contiguous().requires_grad_(True)
    grad_O_local = torch.chunk(grad_O_full, world_size, dim=1)[rank].contiguous()

    scaling = Dh**-0.5

    # Forward.
    out_local = _RingAttention.apply(
        q_local,
        k_local,
        v_local,
        attn_mask_full,
        scaling,
        Hq,
        Hkv,
        dist.group.WORLD,
        (Sq_local,) * world_size,
        (Sq_local,) * world_size,
    )

    if rank == 0:
        print(_summary("forward.out_local", out_local))

    # Backward via .backward(gradient=grad_O_local) — runs _RingAttention.backward.
    out_local.backward(gradient=grad_O_local)
    dq, dk, dv = q_local.grad, k_local.grad, v_local.grad
    dist.barrier()
    for r in range(world_size):
        if r == rank:
            print(f"[rank {rank}]")
            print("  " + _summary("dq", dq))
            print("  " + _summary("dk", dk))
            print("  " + _summary("dv", dv))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
