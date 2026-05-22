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

"""Microbenchmark the three attention backends on a pi07_paligemma-like shape.

Measures wall-clock for fwd-only and fwd+bwd, plus torch.cuda peak memory.
Single GPU; for the distributed picture, run a smoke training step under
``opentau-train`` with the corresponding ``attention_implementation`` and
compare ``nvidia-smi`` / wandb step times.

Example:

    python -m opentau.scripts.bench_paligemma_attention \\
        --backends eager sdpa flex \\
        --batch-size 2 --s-q 1024 --s-kv 1280 --prefix-kv 256 --iters 30

The default shape mirrors pi07_paligemma low_level training: head_dim=256,
8 query heads, 1 KV head (GQA 8x), bf16. A KV-prefix of 256 simulates the
PaliGemma vision/language prefix that the action expert cross-attends to.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from opentau.policies.pi05.paligemma_with_expert import PaliGemmaWithExpertModel


def _make_fake_self(num_attention_heads: int, num_key_value_heads: int, head_dim: int):
    text_config = SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    paligemma_config = SimpleNamespace(text_config=text_config)
    config = SimpleNamespace(paligemma_config=paligemma_config)
    return SimpleNamespace(config=config)


def _block_causal_mask(b: int, s_q: int, s_kv: int, prefix_kv: int, *, device):
    mask = torch.zeros(b, s_q, s_kv, dtype=torch.bool, device=device)
    mask[:, :, :prefix_kv] = True
    causal_kv = s_kv - prefix_kv
    for q_idx in range(s_q):
        tail = min(q_idx + 1, causal_kv)
        if tail > 0:
            mask[:, q_idx, prefix_kv : prefix_kv + tail] = True
    return mask


_FNS = {
    "eager": PaliGemmaWithExpertModel.eager_attention_forward,
    "sdpa": PaliGemmaWithExpertModel.sdpa_attention_forward,
    "flex": PaliGemmaWithExpertModel.flex_attention_forward,
}


def _time(fn, q, k, v, mask, fake, b, head_dim, *, n_warmup: int, n_iter: int, do_backward: bool):
    # Warmup — includes torch.compile JIT for flex.
    for _ in range(n_warmup):
        q_ = q.clone().detach().requires_grad_(do_backward)
        k_ = k.clone().detach().requires_grad_(do_backward)
        v_ = v.clone().detach().requires_grad_(do_backward)
        out = fn(fake, mask, b, head_dim, q_, k_, v_)
        if do_backward:
            out.float().sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        q_ = q.clone().detach().requires_grad_(do_backward)
        k_ = k.clone().detach().requires_grad_(do_backward)
        v_ = v.clone().detach().requires_grad_(do_backward)
        out = fn(fake, mask, b, head_dim, q_, k_, v_)
        if do_backward:
            out.float().sum().backward()
    end.record()
    torch.cuda.synchronize()
    ms_per_iter = start.elapsed_time(end) / n_iter
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return ms_per_iter, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["eager", "sdpa", "flex"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--s-q", type=int, default=1024, help="Q sequence length")
    parser.add_argument("--s-kv", type=int, default=1280, help="KV sequence length")
    parser.add_argument("--prefix-kv", type=int, default=256, help="KV prefix that all Q can attend to")
    parser.add_argument("--num-att", type=int, default=8)
    parser.add_argument("--num-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--fwd-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark requires a GPU.")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    fake = _make_fake_self(args.num_att, args.num_kv, args.head_dim)
    q = torch.randn(args.batch_size, args.s_q, args.num_att, args.head_dim, dtype=dtype, device=device)
    k = torch.randn(args.batch_size, args.s_kv, args.num_kv, args.head_dim, dtype=dtype, device=device)
    v = torch.randn(args.batch_size, args.s_kv, args.num_kv, args.head_dim, dtype=dtype, device=device)
    mask = _block_causal_mask(args.batch_size, args.s_q, args.s_kv, args.prefix_kv, device=device)

    direction = "fwd" if args.fwd_only else "fwd+bwd"
    print(
        f"\nbatch={args.batch_size} S_q={args.s_q} S_kv={args.s_kv} prefix_kv={args.prefix_kv} "
        f"H={args.num_att}/{args.num_kv} D={args.head_dim} {args.dtype} {direction}"
    )
    print(f"{'backend':<8} {'ms/iter':>10} {'peak_MB':>10} {'speedup':>10}")
    print("-" * 42)

    baseline_ms = None
    for backend in args.backends:
        if backend not in _FNS:
            print(f"unknown backend: {backend}")
            continue
        ms, peak = _time(
            _FNS[backend],
            q,
            k,
            v,
            mask,
            fake,
            args.batch_size,
            args.head_dim,
            n_warmup=args.warmup,
            n_iter=args.iters,
            do_backward=not args.fwd_only,
        )
        if baseline_ms is None:
            baseline_ms = ms
            speedup = "1.00x"
        else:
            speedup = f"{baseline_ms / ms:.2f}x"
        print(f"{backend:<8} {ms:>10.3f} {peak:>10.1f} {speedup:>10}")


if __name__ == "__main__":
    main()
