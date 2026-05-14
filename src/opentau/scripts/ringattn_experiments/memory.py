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

"""Peak-memory comparison: ring vs SDPA at the same context length.

Run with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.memory \
        --seq-len 8192

Both branches run a forward + backward on a tiny pi07 backbone (no real
pretrained weights) at the requested sequence length and report per-rank
peak CUDA allocation. Ring is expected to use less memory than SDPA at
large sequence length because the per-rank attention block scales with
``S / world_size`` rather than ``S``.

We measure forward + backward (loss = sum of output) rather than just
forward — backward dominates peak memory in a real training step.
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
    peak_memory_gb,
    reset_peak_memory,
    synthesise_prefix_inputs,
)


def _run(model, impl: str, embs, attn_mask, position_ids, seq_len, device):
    reset_peak_memory(device)
    embs_grad = embs.detach().clone().requires_grad_(True)
    with attention_implementation(model, impl):
        (out, _), _ = model.forward(
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs_grad, None],
            n_cross_att_tokens=seq_len,
            use_cache=False,
            fill_kv_cache=True,
        )
    assert out is not None
    # Check forward output finiteness.
    n_nf_out = (~torch.isfinite(out)).sum().item()
    if n_nf_out > 0:
        rank = dist.get_rank()
        print(f"[rank {rank}] {impl} forward out has {n_nf_out} non-finite elements")
    loss = out.float().sum()
    loss.backward()
    # Check gradient finiteness on every model parameter.
    nf_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n_nf = (~torch.isfinite(p.grad)).sum().item()
        if n_nf > 0:
            nf_params.append((name, n_nf, p.grad.numel()))
    if nf_params:
        rank = dist.get_rank()
        print(f"[rank {rank}] {impl} GRADIENT NON-FINITE on {len(nf_params)} params:")
        for name, n_nf, total in nf_params[:5]:
            print(f"    {name}: {n_nf}/{total} non-finite")
    return peak_memory_gb(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    args = ap.parse_args()

    rank, world_size, device = init_distributed()
    torch.manual_seed(0)

    cfg = build_tiny_gemma3_config(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers)
    model = build_model(cfg, device=device).train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    embs, attn_mask, position_ids = synthesise_prefix_inputs(
        1, args.seq_len, args.hidden_size, device=device, seed=42
    )

    # SDPA baseline.
    sdpa_peak = _run(model, "sdpa", embs, attn_mask, position_ids, args.seq_len, device)
    dist.barrier()
    # Drop everything between runs so the second run's peak isn't biased by
    # the first run's leftover memory.
    torch.cuda.empty_cache()

    ring_peak = _run(model, "ring", embs, attn_mask, position_ids, args.seq_len, device)

    # All-gather per-rank peaks for a single report.
    peaks_sdpa = [0.0] * world_size
    peaks_ring = [0.0] * world_size
    g_sdpa = torch.tensor([sdpa_peak], device=device)
    g_ring = torch.tensor([ring_peak], device=device)
    out_sdpa = [torch.zeros_like(g_sdpa) for _ in range(world_size)]
    out_ring = [torch.zeros_like(g_ring) for _ in range(world_size)]
    dist.all_gather(out_sdpa, g_sdpa)
    dist.all_gather(out_ring, g_ring)
    peaks_sdpa = [t.item() for t in out_sdpa]
    peaks_ring = [t.item() for t in out_ring]

    if rank == 0:
        print(f"world_size = {world_size}")
        print(f"seq_len    = {args.seq_len}")
        print(f"hidden     = {args.hidden_size}")
        print(f"layers     = {args.num_layers}")
        print()
        print(f"{'rank':>6} | {'sdpa peak (GiB)':>18} | {'ring peak (GiB)':>18} | {'reduction':>10}")
        print("-" * 64)
        for r in range(world_size):
            red = (peaks_sdpa[r] - peaks_ring[r]) / max(peaks_sdpa[r], 1e-9) * 100
            print(f"{r:>6} | {peaks_sdpa[r]:>18.3f} | {peaks_ring[r]:>18.3f} | {red:>9.1f}%")
        max_sdpa = max(peaks_sdpa)
        max_ring = max(peaks_ring)
        print()
        print(f"max sdpa peak = {max_sdpa:.3f} GiB")
        print(f"max ring peak = {max_ring:.3f} GiB")
        delta = max_sdpa - max_ring
        print(f"savings       = {delta:.3f} GiB ({delta / max_sdpa * 100:.1f}%)")

    cleanup_distributed()


if __name__ == "__main__":
    main()
