#!/usr/bin/env python
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
"""Benchmark the custom block-causal flash-attention kernel vs eager / sdpa.

Reports per-op latency and peak attention memory for the eager backend (the
``pi07_paligemma`` default, ``attention_implementation="eager"``), PyTorch's
``sdpa`` (cuBLAS/cuDNN-backed), and the custom ``flash_cuda`` kernel, for both
forward and forward+backward, at representative pi07_paligemma shapes
(head_dim=256, MQA). Also demonstrates the regime where eager runs out of memory
but the fused kernel still completes.

Honest summary of what to expect: ``flash_cuda`` reconstructs the block-causal
mask in-kernel and never materializes the (B, H, S, S) scores/probs or the
dense mask, so its peak attention memory is far below eager's. On raw latency it
currently trails the cuBLAS/cuDNN-backed eager and sdpa paths at head_dim=256
(the gap narrows at longer sequence) — closing it requires FlashAttention-2-class
engineering (cp.async double-buffering + register-resident accumulation), tracked
as follow-up.

Usage:
    python -m opentau.scripts.benchmark_flash_attn
    python -m opentau.scripts.benchmark_flash_attn --oom-demo
"""

from __future__ import annotations

import argparse

import torch

from opentau.policies.flash_attn_cuda import (
    flash_attn_blockmask,
    is_available,
    load_error,
    make_att_block_ids,
)

BIG_NEG = -2.3819763e38


def _mask_and_blocks(b, s, device, pattern="prefixlm"):
    att = torch.zeros(b, s, dtype=torch.int32, device=device)
    if pattern == "prefixlm":
        att[:, s // 2] = 1
        att[:, s // 2 + 1 :] = 1
    elif pattern == "causal":
        att[:] = 1
    pad = torch.ones(b, s, dtype=torch.bool, device=device)
    q_blk, k_blk, q_valid, k_valid = make_att_block_ids(pad, att)
    dense = (k_blk[:, None, :] <= q_blk[:, :, None]) & pad[:, :, None] & pad[:, None, :]
    return (q_blk, k_blk, q_valid, k_valid), dense


def _eager(q, k, v, mask, scale):
    g = q.shape[2] // k.shape[2]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    aw = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    aw = torch.where(mask[:, None], aw, BIG_NEG)
    p = torch.softmax(aw, dim=-1).to(vf.dtype)
    return torch.matmul(p, vf).permute(0, 2, 1, 3).contiguous()


def _sdpa(q, k, v, mask, scale):
    g = q.shape[2] // k.shape[2]
    qf = q.permute(0, 2, 1, 3)
    kf = k.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    vf = v.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask[:, None], scale=scale)
    return o.permute(0, 2, 1, 3).contiguous()


def _bench(fn, iters=20):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, (torch.cuda.max_memory_allocated() - base) / 1e6


def run(shapes, dtype=torch.bfloat16, backward=False):
    device = "cuda"
    tag = "fwd+bwd" if backward else "fwd"
    hdr = f"{'shape':28s} {'mode':8s} {'eager ms':>9s} {'sdpa ms':>9s} {'flash ms':>9s} "
    hdr += f"{'eager MB':>9s} {'sdpa MB':>9s} {'flash MB':>9s} {'mem x':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for b, s, h, hkv, d in shapes:
        (q_blk, k_blk, q_valid, k_valid), dense = _mask_and_blocks(b, s, device)
        scale = d**-0.5
        q = torch.randn(b, s, h, d, device=device, dtype=dtype)
        k = torch.randn(b, s, hkv, d, device=device, dtype=dtype)
        v = torch.randn(b, s, hkv, d, device=device, dtype=dtype)

        def make(fn, *args):
            if not backward:
                return lambda: fn(*args)
            q0, k0, v0, rest = args[0], args[1], args[2], args[3:]

            def step():
                qq = q0.detach().requires_grad_(True)
                kk = k0.detach().requires_grad_(True)
                vv = v0.detach().requires_grad_(True)
                fn(qq, kk, vv, *rest).sum().backward()

            return step

        et, em = _bench(make(_eager, q, k, v, dense, scale))
        st, sm = _bench(make(_sdpa, q, k, v, dense, scale))
        ft, fm = _bench(make(flash_attn_blockmask, q, k, v, q_blk, k_blk, q_valid, k_valid, scale))
        name = f"B{b} S{s} H{h} Hkv{hkv} D{d}"
        print(
            f"{name:28s} {tag:8s} {et:9.3f} {st:9.3f} {ft:9.3f} {em:9.1f} {sm:9.1f} {fm:9.1f} {em / fm:6.2f}x"
        )


def oom_demo():
    """Show a shape where the eager (materialized) path OOMs but flash runs."""
    device = "cuda"
    b, h, hkv, d = 4, 8, 1, 256
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free/total: {free / 1e9:.1f}/{total / 1e9:.1f} GB")
    for s in [8192, 16384, 24576]:
        (q_blk, k_blk, q_valid, k_valid), dense = _mask_and_blocks(b, s, device, "causal")
        scale = d**-0.5
        q = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        # eager would need ~B*H*S^2*4 bytes for fp32 scores alone:
        eager_scores_gb = b * h * s * s * 4 / 1e9
        try:
            torch.cuda.reset_peak_memory_stats()
            _eager(q, k, v, dense, scale)
            torch.cuda.synchronize()
            eager_status = f"ok ({torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)"
        except torch.cuda.OutOfMemoryError:
            eager_status = "OOM"
            torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
            flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
            torch.cuda.synchronize()
            flash_status = f"ok ({torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak)"
        except torch.cuda.OutOfMemoryError:
            flash_status = "OOM"
            torch.cuda.empty_cache()
        print(
            f"B{b} S{s} H{h} D{d}: eager fp32-scores would need ~{eager_scores_gb:.1f} GB | "
            f"eager: {eager_status} | flash: {flash_status}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oom-demo", action="store_true", help="run the eager-OOM-vs-flash demo")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")
    if not is_available():
        raise SystemExit(f"flash_cuda kernel unavailable: {load_error()}")

    print(f"Device: {torch.cuda.get_device_name(0)}\n")
    shapes = [(2, 512, 8, 1, 256), (2, 768, 8, 1, 256), (1, 1024, 8, 1, 256), (1, 2048, 8, 1, 256)]
    print("=== Forward ===")
    run(shapes, backward=False)
    print("\n=== Forward + backward ===")
    run(shapes, backward=True)
    if args.oom_demo:
        print("\n=== Eager-OOM vs flash (enabling win) ===")
        oom_demo()


if __name__ == "__main__":
    main()
