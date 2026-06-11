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
dense mask, so its peak attention memory is below eager's. Two important caveats:

  * The per-op tables (forward / forward+backward) compare a SINGLE attention
    call with NO gradient checkpointing on either side. Those large multipliers
    are vs non-checkpointed eager, which is not the realistic training config.
  * Realistic pi07_paligemma training uses ``gradient_checkpointing=True``,
    which already frees eager's per-layer (B,H,S,S) scores (recomputed in
    backward). The fair, decision-relevant comparison is the stacked-layer
    section with checkpointing on both sides ("eager+ckpt vs flash+ckpt"), where
    the advantage is ~2x, not 20-40x. flash composes with checkpointing too.

On raw latency it currently trails the cuBLAS/cuDNN-backed eager and sdpa paths
at head_dim=256 (the gap narrows at longer sequence) — closing it requires
FlashAttention-2-class engineering (cp.async double-buffering + register-resident
accumulation), tracked as follow-up.

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


def _blocks_only(b, s, device, pattern="prefixlm"):
    """Compact block-ids ONLY (no dense mask) — what the flash path actually uses."""
    att = torch.zeros(b, s, dtype=torch.int32, device=device)
    if pattern == "prefixlm":
        att[:, s // 2] = 1
        att[:, s // 2 + 1 :] = 1
    elif pattern == "causal":
        att[:] = 1
    pad = torch.ones(b, s, dtype=torch.bool, device=device)
    return make_att_block_ids(pad, att)


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


def stacked_memory_comparison(b=2, s=1024, h=8, hkv=1, d=256, layers=18):
    """Fair full-model-style comparison: a stack of attention layers run fwd+bwd,
    with and without ``torch.utils.checkpoint``, for eager / sdpa / flash.

    This is the decision-relevant comparison (the per-op tables above compare a
    single attention call with no checkpointing). Realistic pi07_paligemma
    training uses ``gradient_checkpointing=True``, which already frees eager's
    per-layer (B,H,S,S) scores by recomputing them in backward; so the honest
    apples-to-apples for training is the "+ckpt" rows. Reports both peak memory
    and fwd+bwd wall-clock latency for each backend.
    """
    import torch.utils.checkpoint as cp

    device = "cuda"
    (q_blk, k_blk, q_valid, k_valid), dense = _mask_and_blocks(b, s, device)
    scale = d**-0.5

    def eager_layer(x):
        g = h // hkv
        q = x.permute(0, 2, 1, 3).float()
        k = x[:, :, :hkv].permute(0, 2, 1, 3).float().repeat_interleave(g, 1)
        v = x[:, :, :hkv].permute(0, 2, 1, 3).float().repeat_interleave(g, 1)
        aw = torch.where(dense[:, None], torch.matmul(q, k.transpose(-1, -2)) * scale, BIG_NEG)
        return torch.matmul(torch.softmax(aw, -1), v).permute(0, 2, 1, 3).to(x.dtype)

    def sdpa_layer(x):
        g = h // hkv
        q = x.permute(0, 2, 1, 3)
        k = x[:, :, :hkv].permute(0, 2, 1, 3).repeat_interleave(g, 1)
        v = x[:, :, :hkv].permute(0, 2, 1, 3).repeat_interleave(g, 1)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense[:, None], scale=scale)
        return o.permute(0, 2, 1, 3)

    def flash_layer(x):
        return flash_attn_blockmask(
            x, x[:, :, :hkv].contiguous(), x[:, :, :hkv].contiguous(), q_blk, k_blk, q_valid, k_valid, scale
        )

    def fwd_bwd(layer, ckpt):
        x = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(layers):
            x = (cp.checkpoint(layer, x, use_reentrant=False) if ckpt else layer(x)) + 1.0
        x.sum().backward()

    def measure(layer, ckpt, iters=10):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(2):  # warmup
                fwd_bwd(layer, ckpt)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1e9
            s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            for _ in range(iters):
                fwd_bwd(layer, ckpt)
            e0.record()
            torch.cuda.synchronize()
            return f"{s0.elapsed_time(e0) / iters:7.2f} ms / {mem:5.2f} GB"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "        OOM        "

    print(
        f"{layers} stacked attention layers, B{b} S{s} H{h} Hkv{hkv} D{d} bf16, fwd+bwd (latency / peak mem):"
    )
    print(f"  {'backend':6s} {'no-ckpt':>22s}   {'+ckpt (realistic training)':>26s}")
    for name, layer in [("eager", eager_layer), ("sdpa", sdpa_layer), ("flash", flash_layer)]:
        print(f"  {name:6s} {measure(layer, False):>22s}   {measure(layer, True):>26s}")


def sdpa_backend_probe(b=1, s=4096, h=8, hkv=1, d=256):
    """Show which torch SDPA backends accept a *block-causal* attn_mask.

    PyTorch's fused FlashAttention backend only supports ``is_causal`` or no
    mask; an arbitrary block-causal mask is rejected, leaving the
    memory-efficient or math backend. Crucially, both of those still take the
    dense ``(B, 1, S, S)`` mask as input — O(S²) memory — whereas flash_cuda
    reconstructs the mask from compact block-ids (O(S)).
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    device = "cuda"
    _, dense = _mask_and_blocks(b, s, device)
    scale = d**-0.5
    g = h // hkv
    q = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    k = (
        torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        .permute(0, 2, 1, 3)
        .repeat_interleave(g, 1)
    )
    v = (
        torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        .permute(0, 2, 1, 3)
        .repeat_interleave(g, 1)
    )
    mask = dense[:, None]
    # Also build a NON-SQUARE cross-attention mask (expert pass: query len N,
    # key len n_cross + N) — the shape pi07_paligemma's action expert uses.
    n_cross = s // 2
    qx = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    kx = torch.randn(b, h, n_cross + s, d, device=device, dtype=torch.bfloat16)
    vx = torch.randn(b, h, n_cross + s, d, device=device, dtype=torch.bfloat16)
    cross_mask = torch.ones(b, 1, s, n_cross + s, dtype=torch.bool, device=device)

    def probe(qq, kk, vv, mm, label):
        print(f"{label}:")
        for name, backend in [
            ("flash (FlashAttention)", SDPBackend.FLASH_ATTENTION),
            ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math (== eager)", SDPBackend.MATH),
        ]:
            try:
                with sdpa_kernel([backend]):
                    torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mm, scale=scale)
                torch.cuda.synchronize()
                print(f"  {name:24s}: accepts the mask")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:24s}: REJECTS ({type(e).__name__})")

    probe(q, k, v, mask, f"SDPA backends for a SQUARE block-causal bool mask (S={s}, D={d})")
    probe(
        qx,
        kx,
        vx,
        cross_mask,
        f"SDPA backends for a NON-SQUARE cross-attn bool mask (Sq={s}, Sk={n_cross + s}, D={d})",
    )
    print("  note: every accepting backend still needs the dense (B,1,Sq,Sk) mask as input (O(Sq*Sk)).")


def long_context_sweep(b=1, h=8, hkv=1, d=256, seqs=(4096, 8192, 16384, 32768, 65536, 98304)):
    """Forward peak-memory + latency vs sequence length for eager / sdpa / flash.

    Demonstrates the regime flash_cuda uniquely serves: eager materializes the
    O(S²) scores and sdpa needs the O(S²) dense mask, so both OOM as S grows;
    flash uses compact block-ids (O(S)) and keeps running. Block-causal
    (prefix-LM) mask, bf16.
    """
    device = "cuda"
    scale = d**-0.5
    print(f"Forward, B{b} H{h} Hkv{hkv} D{d} bf16, prefix-LM block-causal mask.")
    print(f"{'S':>8s} {'dense mask':>10s} | {'eager':>16s} | {'sdpa':>16s} | {'flash_cuda':>16s}")
    print(
        f"{'':>8s} {'(B,S,S)':>10s} | {'ms / peak GB':>16s} | {'ms / peak GB':>16s} | {'ms / peak GB':>16s}"
    )
    print("-" * 80)

    def timed(fn):
        # returns "ms / GB" or "OOM"
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                fn()
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 5
            gb = torch.cuda.max_memory_allocated() / 1e9
            return f"{ms:7.1f} / {gb:5.1f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "       OOM      "

    for s in seqs:
        q = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, s, hkv, d, device=device, dtype=torch.bfloat16)
        q_blk, k_blk, q_valid, k_valid = _blocks_only(b, s, device)  # compact (O(S)), no dense mask
        mask_gb = b * s * s / 1e9  # dense bool mask bytes

        def eager_fn(q=q, k=k, v=v, s=s):
            _, dense = _mask_and_blocks(b, s, device)  # build dense mask here (counts against eager)
            _eager(q, k, v, dense, scale)

        def sdpa_fn(q=q, k=k, v=v, s=s):
            _, dense = _mask_and_blocks(b, s, device)
            _sdpa(q, k, v, dense, scale)

        def flash_fn(q=q, k=k, v=v, q_blk=q_blk, k_blk=k_blk, q_valid=q_valid, k_valid=k_valid):
            flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)

        e_res, s_res, f_res = timed(eager_fn), timed(sdpa_fn), timed(flash_fn)
        del q, k, v
        print(f"{s:>8d} {mask_gb:8.2f}GB | {e_res:>16s} | {s_res:>16s} | {f_res:>16s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oom-demo", action="store_true", help="run the eager-OOM-vs-flash demo")
    parser.add_argument(
        "--long-context", action="store_true", help="run the long-context sweep + SDPA backend probe"
    )
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
    print("\n=== Stacked-layer memory: eager vs flash, with/without gradient checkpointing ===")
    stacked_memory_comparison()
    if args.oom_demo:
        print("\n=== Eager-OOM vs flash (enabling win) ===")
        oom_demo()
    if args.long_context:
        print("\n=== SDPA backend probe (block-causal mask) ===")
        sdpa_backend_probe()
        print("\n=== Long-context forward sweep: eager vs sdpa vs flash ===")
        long_context_sweep()


if __name__ == "__main__":
    main()
