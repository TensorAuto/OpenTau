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

"""Find the longest sequence each attention implementation can handle.

Run with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.longest_context

Binary-search the largest seq length that runs a forward+backward without
hitting CUDA OOM, separately for SDPA and ring. Both branches use the
same tiny pi07 backbone; the only difference is the attention path.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from opentau.scripts.ringattn_experiments.common import (
    attention_implementation,
    build_model,
    build_tiny_gemma3_config,
    cleanup_distributed,
    init_distributed,
    synthesise_prefix_inputs,
)


def _try_run(model, impl: str, seq_len: int, hidden_size: int, device) -> bool:
    """Returns True iff a forward+backward at ``seq_len`` fits in memory."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        embs, attn_mask, position_ids = synthesise_prefix_inputs(
            1, seq_len, hidden_size, device=device, seed=42
        )
        embs.requires_grad_(True)
        with attention_implementation(model, impl):
            (out, _), _ = model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                n_cross_att_tokens=seq_len,
                use_cache=False,
                fill_kv_cache=True,
            )
        assert out is not None
        loss = out.float().sum()
        loss.backward()
        torch.cuda.synchronize(device)
        del embs, attn_mask, position_ids, out, loss
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False


def _all_agree(local: bool, device) -> bool:
    """All-reduce a bool: True iff every rank reports True."""
    t = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _search_max(model, impl: str, hidden_size: int, device, world_size: int, rank: int):
    """Binary-search the largest seq_len that fits. Returns (max_ok, first_oom)."""
    # Probe a coarse upper bound by doubling from a chunky starting point.
    # We're not interested in tiny contexts here — the question is the
    # large-context ceiling.
    lo = 16384
    hi = 16384
    while True:
        ok = _try_run(model, impl, hi, hidden_size, device)
        ok_all = _all_agree(ok, device)
        if rank == 0:
            print(f"  {impl}: seq_len={hi:>6} -> {'OK' if ok_all else 'OOM'}")
        if not ok_all:
            break
        lo = hi
        hi *= 2
        # Hard ceiling to keep the search tractable.
        if hi > 262144:
            return lo, None
    # Binary search [lo, hi).
    # ``lo`` is known-OK, ``hi`` is known-OOM.
    while hi - lo > max(world_size, 256):
        mid = ((lo + hi) // 2 // world_size) * world_size  # keep multiple of ws
        if mid <= lo:
            break
        ok = _try_run(model, impl, mid, hidden_size, device)
        ok_all = _all_agree(ok, device)
        if rank == 0:
            print(f"  {impl}: seq_len={mid:>6} -> {'OK' if ok_all else 'OOM'}")
        if ok_all:
            lo = mid
        else:
            hi = mid
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    args = ap.parse_args()

    rank, world_size, device = init_distributed()
    torch.manual_seed(0)

    cfg = build_tiny_gemma3_config(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers)
    model = build_model(cfg, device=device).train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    if rank == 0:
        print(f"world_size = {world_size}, hidden = {args.hidden_size}, layers = {args.num_layers}")
        print("Searching SDPA max context:")
    sdpa_max, sdpa_oom = _search_max(model, "sdpa", args.hidden_size, device, world_size, rank)

    if rank == 0:
        print("Searching ring max context:")
    ring_max, ring_oom = _search_max(model, "ring", args.hidden_size, device, world_size, rank)

    if rank == 0:
        print()
        print(f"SDPA: largest OK = {sdpa_max}  (first OOM = {sdpa_oom})")
        print(f"RING: largest OK = {ring_max}  (first OOM = {ring_oom})")
        if ring_max > sdpa_max:
            print(f"RING extends max context by {ring_max / sdpa_max:.2f}x")

    cleanup_distributed()


if __name__ == "__main__":
    main()
