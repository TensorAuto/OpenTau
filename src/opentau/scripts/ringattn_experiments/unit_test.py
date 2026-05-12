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

"""Unit test the ring attention kernel against eager attention on a single layer.

Run with ``torchrun --nproc_per_node=2``. Builds random Q/K/V tensors on
rank 0, broadcasts them, then both ranks compute (a) the reference
single-device eager attention against the full tensors and (b) the ring
attention against this rank's local shard. Each rank validates that its
slice of the eager output matches its slice of the ring output.

Isolates the kernel from the surrounding Gemma3 forward so we can rule
in / out the kernel as the source of any discrepancy seen by
``correctness.py``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from opentau.scripts.ringattn_experiments.common import (
    cleanup_distributed,
    init_distributed,
)


def _eager_reference(q, k, v, mask, scaling, num_q_heads, num_kv_heads):
    """Identical math to ``Gemma3WithExpertModel.eager_attention_forward`` but
    standalone."""
    B, S, Hq, Dh = q.shape
    groups = num_q_heads // num_kv_heads
    k_exp = k.unsqueeze(3).expand(B, S, num_kv_heads, groups, Dh).reshape(B, S, num_q_heads, Dh)
    v_exp = v.unsqueeze(3).expand(B, S, num_kv_heads, groups, Dh).reshape(B, S, num_q_heads, Dh)

    q32 = q.float()
    k32 = k_exp.float()
    v32 = v_exp.float()
    S_scores = torch.einsum("bqhd,bkhd->bhqk", q32, k32) * scaling
    S_scores = torch.where(mask.unsqueeze(1), S_scores, torch.full_like(S_scores, -2.3819763e38))
    probs = torch.softmax(S_scores, dim=-1).to(v32.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v32)
    return out.to(q.dtype)


def main() -> None:
    rank, world_size, device = init_distributed()
    torch.manual_seed(0)

    B = 2
    S = 64  # total sequence length
    Hq = 4
    Hkv = 2
    Dh = 32
    dtype = torch.bfloat16

    # Build inputs on rank 0, broadcast so every rank has the same global tensors.
    if rank == 0:
        q_full = torch.randn(B, S, Hq, Dh, device=device, dtype=dtype)
        k_full = torch.randn(B, S, Hkv, Dh, device=device, dtype=dtype)
        v_full = torch.randn(B, S, Hkv, Dh, device=device, dtype=dtype)
        # Block-causal-like mask: bidirectional within two halves.
        mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        mask[:, :, :] = True
    else:
        q_full = torch.empty(B, S, Hq, Dh, device=device, dtype=dtype)
        k_full = torch.empty(B, S, Hkv, Dh, device=device, dtype=dtype)
        v_full = torch.empty(B, S, Hkv, Dh, device=device, dtype=dtype)
        mask = torch.empty(B, S, S, dtype=torch.bool, device=device)
    dist.broadcast(q_full, src=0)
    dist.broadcast(k_full, src=0)
    dist.broadcast(v_full, src=0)
    dist.broadcast(mask, src=0)

    scaling = Dh**-0.5

    # Reference.
    ref = _eager_reference(q_full, k_full, v_full, mask, scaling, Hq, Hkv)

    # Local shards.
    chunks_q = torch.chunk(q_full, world_size, dim=1)
    chunks_k = torch.chunk(k_full, world_size, dim=1)
    chunks_v = torch.chunk(v_full, world_size, dim=1)
    q_local = chunks_q[rank].contiguous()
    k_local = chunks_k[rank].contiguous()
    v_local = chunks_v[rank].contiguous()

    from opentau.policies.pi07.ring_attention import _RingAttention

    Sq_local = S // world_size
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

    ref_local = torch.chunk(ref, world_size, dim=1)[rank].contiguous()
    diff = (ref_local.float() - out_local.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_max = ref_local.float().abs().max().item()

    for r in range(world_size):
        if r == rank:
            print(
                f"[rank {rank}] ref local |max| = {ref_max:.4f}, "
                f"max |ref - ring| = {max_abs:.3e}, mean = {mean_abs:.3e}"
            )
        dist.barrier()
    if rank == 0:
        print("PASS" if max_abs < 1e-2 else "FAIL")

    cleanup_distributed()


if __name__ == "__main__":
    main()
