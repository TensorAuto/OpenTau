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

"""Memory scaling table: ring vs SDPA across a range of sequence lengths.

Run with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.memory_scaling

Sweeps a fixed list of sequence lengths and prints a single table so the
reader can see where the crossover between SDPA's flash backend and ring
sits on this hardware.
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


def _run(model, impl: str, seq_len: int, hidden_size: int, device) -> float | None:
    try:
        torch.cuda.empty_cache()
        reset_peak_memory(device)
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
        peak = peak_memory_gb(device)
        del embs, attn_mask, position_ids, out, loss
        torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768, 65536],
    )
    args = ap.parse_args()

    rank, world_size, device = init_distributed()
    torch.manual_seed(0)

    cfg = build_tiny_gemma3_config(hidden_size=args.hidden_size, num_hidden_layers=args.num_layers)
    model = build_model(cfg, device=device).train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    rows = []
    for s in args.seq_lens:
        sdpa = _run(model, "sdpa", s, args.hidden_size, device)
        ring = _run(model, "ring", s, args.hidden_size, device)
        rows.append((s, sdpa, ring))
        if rank == 0:
            sdpa_s = f"{sdpa:.3f}" if sdpa is not None else "OOM"
            ring_s = f"{ring:.3f}" if ring is not None else "OOM"
            print(f"  seq_len={s:>6}  sdpa={sdpa_s:>8}  ring={ring_s:>8}")

    if rank == 0:
        print()
        print(f"world_size = {world_size}, hidden = {args.hidden_size}, layers = {args.num_layers}")
        print(f"{'seq_len':>8} | {'sdpa (GiB)':>11} | {'ring (GiB)':>11} | {'reduction':>10}")
        print("-" * 50)
        for s, sdpa, ring in rows:
            if sdpa is None or ring is None:
                sdpa_s = "OOM" if sdpa is None else f"{sdpa:.3f}"
                ring_s = "OOM" if ring is None else f"{ring:.3f}"
                print(f"{s:>8} | {sdpa_s:>11} | {ring_s:>11} | {'-':>10}")
            else:
                red = (sdpa - ring) / sdpa * 100
                print(f"{s:>8} | {sdpa:>11.3f} | {ring:>11.3f} | {red:>9.1f}%")

    cleanup_distributed()


if __name__ == "__main__":
    main()
