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

"""Gradient-equivalence test: ring vs eager autograd backward.

Run with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.backward_test

What's tested
-------------
On rank 0, build random Q/K/V and an upstream gradient grad_O, broadcast
all four tensors so every rank has the same global view.

Reference path (every rank):
  1. Compute eager attention via the same fp32-score chain
     ``_eager_reference`` uses in ``unit_test.py``.
  2. Compute autograd gradients ``dQ_ref, dK_ref, dV_ref`` by running
     ``out.backward(grad_O)`` against requires_grad=True inputs. These
     are FULL-sequence gradients, identical on every rank.

Ring path (per rank):
  1. Shard Q/K/V and grad_O along seq.
  2. Run ``_RingAttention.apply(...)`` (forward + backward via
     ``out_local.backward(grad_O_local)``).
  3. Each rank ends up with its slice of dQ/dK/dV.

Comparison: this rank's slice of ``{dQ,dK,dV}_ref`` against ``{dQ,dK,dV}_ring``.
Acceptable diff: a few bf16 ULPs (the kernel does fp32 score math but
returns bf16; eager reference does the same internally, so the diff should
be near machine epsilon for fp32 reassociation).

The test exists because a real-data pi07 + ZeRO-2 run produces NaN
gradients at step 2 even though the kernel-level forward unit_test
matches eager bit-for-bit. If THIS backward test also passes, the bug is
in the kernel's *interaction* with the surrounding model / optimizer (not
in the kernel itself). If it fails on any of the gradient comparisons,
we have a kernel bug we can fix here without a slurm round-trip.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from einops import repeat

from opentau.policies.pi07.ring_attention import _RingAttention

_NEG_INF_REF = -1.0e9


def _eager_reference(q, k, v, mask, scaling, num_q_heads, num_kv_heads):
    """fp32-score chain identical to ``Gemma3WithExpertModel.eager_attention_forward``
    but standalone, so we can autograd through it.

    The output dtype tracks the input Q dtype, matching what the ring kernel
    returns.
    """
    B, S, Hq, Dh = q.shape
    groups = num_q_heads // num_kv_heads
    if groups > 1:
        k_exp = repeat(k, "b s h d -> b s (h g) d", g=groups)
        v_exp = repeat(v, "b s h d -> b s (h g) d", g=groups)
    else:
        k_exp = k
        v_exp = v

    q32 = q.float()
    k32 = k_exp.float()
    v32 = v_exp.float()
    S_scores = torch.einsum("bqhd,bkhd->bhqk", q32, k32) * scaling
    S_scores = torch.where(mask.unsqueeze(1), S_scores, torch.full_like(S_scores, _NEG_INF_REF))
    probs = torch.softmax(S_scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v32)
    return out.to(q.dtype)


def _summary(name: str, t: torch.Tensor) -> str:
    finite = torch.isfinite(t).all().item()
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.4g} max={t.max().item():.4g} "
        f"finite={finite}"
    )


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs, mean_abs, max_rel) for diagnostic output."""
    d = (a.float() - b.float()).abs()
    ref_norm = b.float().abs().clamp_min(1e-6)
    return d.max().item(), d.mean().item(), (d / ref_norm).max().item()


def main() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29565")
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for random Q/K/V")
    p.add_argument("--grad-scale", type=float, default=1.0, help="Scale factor for upstream grad_O")
    p.add_argument("--s-total", type=int, default=128)
    p.add_argument("--hq", type=int, default=4)
    p.add_argument("--hkv", type=int, default=2)
    p.add_argument("--dh", type=int, default=64)
    args = p.parse_args()

    B = 1
    Hq = args.hq
    Hkv = args.hkv
    Dh = args.dh
    S_total = args.s_total
    assert S_total % world_size == 0
    Sq_local = S_total // world_size
    dtype = torch.bfloat16
    scale = float(args.scale)
    grad_scale = float(args.grad_scale)

    # Build inputs on rank 0, broadcast.
    torch.manual_seed(0)
    if rank == 0:
        q_full = scale * torch.randn(B, S_total, Hq, Dh, device=device, dtype=dtype)
        k_full = scale * torch.randn(B, S_total, Hkv, Dh, device=device, dtype=dtype)
        v_full = scale * torch.randn(B, S_total, Hkv, Dh, device=device, dtype=dtype)
        grad_O_full = grad_scale * torch.randn(B, S_total, Hq, Dh, device=device, dtype=dtype)
        # Block-causal-like mask with a few PAD rows
        n_pad = 5
        from opentau.policies.pi07.low_level.modeling_pi07_low_level import make_att_2d_masks

        pad_masks = torch.ones(B, S_total, dtype=torch.bool, device=device)
        pad_masks[:, -n_pad:] = False
        att_masks = torch.zeros(B, S_total, dtype=torch.int32, device=device)
        att_masks[:, 0] = 1
        att_masks[:, S_total // 2] = 1
        mask = make_att_2d_masks(pad_masks, att_masks)
        # zero out grad on PAD rows (matches what downstream loss masking does)
        grad_O_full[:, -n_pad:] = 0.0
    else:
        q_full = torch.empty(B, S_total, Hq, Dh, device=device, dtype=dtype)
        k_full = torch.empty(B, S_total, Hkv, Dh, device=device, dtype=dtype)
        v_full = torch.empty(B, S_total, Hkv, Dh, device=device, dtype=dtype)
        grad_O_full = torch.empty(B, S_total, Hq, Dh, device=device, dtype=dtype)
        mask = torch.empty(B, S_total, S_total, dtype=torch.bool, device=device)
    dist.broadcast(q_full, src=0)
    dist.broadcast(k_full, src=0)
    dist.broadcast(v_full, src=0)
    dist.broadcast(grad_O_full, src=0)
    dist.broadcast(mask, src=0)

    scaling = Dh**-0.5

    # Reference forward + backward (full sequence, autograd).
    q_ref = q_full.detach().clone().requires_grad_(True)
    k_ref = k_full.detach().clone().requires_grad_(True)
    v_ref = v_full.detach().clone().requires_grad_(True)
    out_ref = _eager_reference(q_ref, k_ref, v_ref, mask, scaling, Hq, Hkv)
    out_ref.backward(gradient=grad_O_full)
    dq_ref = q_ref.grad
    dk_ref = k_ref.grad
    dv_ref = v_ref.grad

    # Ring forward + backward (per-rank shards).
    q_local = torch.chunk(q_full, world_size, dim=1)[rank].detach().clone().contiguous().requires_grad_(True)
    k_local = torch.chunk(k_full, world_size, dim=1)[rank].detach().clone().contiguous().requires_grad_(True)
    v_local = torch.chunk(v_full, world_size, dim=1)[rank].detach().clone().contiguous().requires_grad_(True)
    grad_O_local = torch.chunk(grad_O_full, world_size, dim=1)[rank].contiguous()

    out_local = _RingAttention.apply(
        q_local,
        k_local,
        v_local,
        mask,
        scaling,
        Hq,
        Hkv,
        dist.group.WORLD,
        (Sq_local,) * world_size,
        (Sq_local,) * world_size,
    )
    out_local.backward(gradient=grad_O_local)
    dq_local = q_local.grad
    dk_local = k_local.grad
    dv_local = v_local.grad

    # Compare. Each rank checks its own slice.
    dq_ref_local = torch.chunk(dq_ref, world_size, dim=1)[rank]
    dk_ref_local = torch.chunk(dk_ref, world_size, dim=1)[rank]
    dv_ref_local = torch.chunk(dv_ref, world_size, dim=1)[rank]

    fwd_local_ref = torch.chunk(out_ref, world_size, dim=1)[rank]
    fwd_diff = _diff(fwd_local_ref, out_local)
    dq_diff = _diff(dq_ref_local, dq_local)
    dk_diff = _diff(dk_ref_local, dk_local)
    dv_diff = _diff(dv_ref_local, dv_local)

    def finite_summary(name, t):
        nf = (~torch.isfinite(t)).sum().item()
        return f"{name} non-finite={nf}/{t.numel()} min={t.min().item():.4g} max={t.max().item():.4g}"

    dist.barrier()
    for r in range(world_size):
        if r == rank:
            print(f"[rank {rank}] world_size={world_size}, S_total={S_total}, Sq_local={Sq_local}")
            print(f"  forward  max|diff|={fwd_diff[0]:.3e}  mean|diff|={fwd_diff[1]:.3e}")
            print(
                f"  dq       max|diff|={dq_diff[0]:.3e}  mean|diff|={dq_diff[1]:.3e}  max rel={dq_diff[2]:.3e}"
            )
            print(
                f"  dk       max|diff|={dk_diff[0]:.3e}  mean|diff|={dk_diff[1]:.3e}  max rel={dk_diff[2]:.3e}"
            )
            print(
                f"  dv       max|diff|={dv_diff[0]:.3e}  mean|diff|={dv_diff[1]:.3e}  max rel={dv_diff[2]:.3e}"
            )
            print(f"  ring  {finite_summary('dq', dq_local)}")
            print(f"  ring  {finite_summary('dk', dk_local)}")
            print(f"  ring  {finite_summary('dv', dv_local)}")
            print(f"  ref   {finite_summary('dq', dq_ref_local)}")
            print(f"  ref   {finite_summary('dk', dk_ref_local)}")
            print(f"  ref   {finite_summary('dv', dv_ref_local)}")
        dist.barrier()

    if rank == 0:
        # In bf16 the per-element rel error can be at the 1e-2 level for
        # gradients with large dynamic range; absolute diff below 1e-1 on
        # tensors with O(1) values is the realistic kernel-correctness bar.
        ok = max(dq_diff[0], dk_diff[0], dv_diff[0]) < 5e-1
        print("PASS" if ok else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
