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
"""Tests for the custom block-causal CUDA flash-attention backend.

Three layers of validation:
  1. CPU: the compact block-ids reproduce ``make_att_2d_masks`` bit-for-bit (the
     correctness contract of the in-kernel masking). Runs without a GPU.
  2. GPU op-level: the kernel's forward (fp32) matches a dense reference to fp32
     tolerance and its backward matches autograd; bf16 matches ``sdpa``.
  3. GPU full-model: a tiny PaliGemmaWithExpertModel produces the same output
     (within bf16 noise) under ``attention_implementation="flash_cuda"`` and
     ``"sdpa"``, including the cross-attention / KV-cache expert pass.
  4. GPU perf: the kernel uses less peak memory than the eager backend (the
     headline win — no SxS materialization). Latency is measured and reported.
"""

import pytest
import torch

from opentau.policies import flash_attn_cuda
from opentau.policies.flash_attn_cuda import flash_attn_blockmask, make_att_block_ids
from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import make_att_2d_masks
from tests.utils import require_cuda

BIG_NEG = -2.3819763e38


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _att_masks(pattern: str, b: int, n: int, device) -> torch.Tensor:
    att = torch.zeros(b, n, dtype=torch.int32, device=device)
    if pattern == "causal":
        att[:] = 1
    elif pattern == "prefixlm":
        att[:, n // 2] = 1
        att[:, n // 2 + 1 :] = 1
    elif pattern == "multiblock":
        att[:, :: max(1, n // 5)] = 1
    elif pattern == "bidir":
        pass  # all zeros
    else:
        raise ValueError(pattern)
    return att


def _dense_mask(q_blk, k_blk, q_valid, k_valid) -> torch.Tensor:
    return (k_blk[:, None, :] <= q_blk[:, :, None]) & q_valid[:, :, None] & k_valid[:, None, :]


def _reference_attention(q, k, v, q_blk, k_blk, q_valid, k_valid, scale):
    """Dense masked attention in fp32; returns (out (B,Sq,H,D), valid_row (B,Sq))."""
    b, sq, h, d = q.shape
    hkv = k.shape[2]
    g = h // hkv
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    mask = _dense_mask(q_blk, k_blk, q_valid, k_valid)[:, None]
    scores = scores.masked_fill(~mask, float("-inf"))
    valid_row = mask.any(dim=-1, keepdim=True)
    probs = torch.where(valid_row, torch.softmax(scores, dim=-1), torch.zeros_like(scores))
    out = torch.matmul(probs, vf).permute(0, 2, 1, 3).contiguous()
    return out, valid_row.squeeze(1).squeeze(-1)


def _make_blocks(b, n, device, pattern, cross=0, pad_tail=0):
    pad = torch.ones(b, n, dtype=torch.bool, device=device)
    if pad_tail:
        pad[0, -pad_tail:] = False
    att = _att_masks(pattern, b, n, device)
    if cross:
        cross_pad = torch.ones(b, cross, dtype=torch.bool, device=device)
        cross_pad[0, -1:] = False
        return make_att_block_ids(pad, att, cross, cross_pad)
    return make_att_block_ids(pad, att)


# --------------------------------------------------------------------------- #
# 1. CPU: block-ids reproduce the dense mask exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pattern", ["causal", "prefixlm", "multiblock", "bidir"])
def test_block_ids_reconstruct_dense_mask(pattern):
    b, n = 3, 24
    pad = torch.ones(b, n, dtype=torch.bool)
    pad[0, -4:] = False  # trailing padding on one sample
    att = _att_masks(pattern, b, n, torch.device("cpu"))
    q_blk, k_blk, q_valid, k_valid = make_att_block_ids(pad, att)
    recon = _dense_mask(q_blk, k_blk, q_valid, k_valid)
    ref = make_att_2d_masks(pad, att)
    assert torch.equal(recon, ref)


def test_block_ids_reconstruct_cross_attention_mask():
    b, n, cross = 2, 16, 10
    pad = torch.ones(b, n, dtype=torch.bool)
    pad[0, -3:] = False
    att = _att_masks("causal", b, n, torch.device("cpu"))
    cross_pad = torch.ones(b, cross, dtype=torch.bool)
    cross_pad[1, -2:] = False
    q_blk, k_blk, q_valid, k_valid = make_att_block_ids(pad, att, cross, cross_pad)
    recon = _dense_mask(q_blk, k_blk, q_valid, k_valid)
    ref = make_att_2d_masks(pad, att, n_cross_att_tokens=cross, cross_att_pad_masks=cross_pad)
    assert recon.shape == ref.shape == (b, n, cross + n)
    assert torch.equal(recon, ref)


# --------------------------------------------------------------------------- #
# 2. GPU op-level: forward / backward equivalence
# --------------------------------------------------------------------------- #
@require_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("pattern", ["causal", "prefixlm", "multiblock", "bidir"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("hkv", [1, 2])
def test_forward_fp32_matches_dense_reference(pattern, head_dim, hkv):
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(0)
    b, sq, h = 2, 48, 4
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, sq, "cuda", pattern, pad_tail=3)
    scale = head_dim**-0.5
    q = torch.randn(b, sq, h, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=torch.float32)

    out = flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    ref, _ = _reference_attention(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    # Compare only valid query rows (fully-masked padded rows are discarded downstream).
    vr = q_valid[:, :, None, None]
    torch.testing.assert_close(out * vr, ref * vr, atol=1e-4, rtol=1e-4)


@require_cuda
@pytest.mark.gpu
def test_forward_fp32_cross_attention_non_square():
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(1)
    b, sq, h, hkv, d, cross = 2, 33, 8, 1, 256, 20
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, sq, "cuda", "causal", cross=cross)
    sk = cross + sq
    scale = d**-0.5
    q = torch.randn(b, sq, h, d, device="cuda")
    k = torch.randn(b, sk, hkv, d, device="cuda")
    v = torch.randn(b, sk, hkv, d, device="cuda")
    out = flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    ref, _ = _reference_attention(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    vr = q_valid[:, :, None, None]
    torch.testing.assert_close(out * vr, ref * vr, atol=1e-4, rtol=1e-4)


@require_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("pattern", ["causal", "prefixlm", "multiblock"])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_backward_matches_autograd_reference(pattern, head_dim):
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(0)
    b, sq, h, hkv = 2, 40, 4, 1
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, sq, "cuda", pattern, pad_tail=2)
    scale = head_dim**-0.5
    q = torch.randn(b, sq, h, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(b, sq, h, head_dim, device="cuda")

    out = flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    out.backward(grad_out)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    ref, _ = _reference_attention(qr, kr, vr, q_blk, k_blk, q_valid, k_valid, scale)
    ref.backward(grad_out)

    qmask = q_valid[:, :, None, None]
    kmask = k_valid[:, :, None, None]
    torch.testing.assert_close(dq * qmask, qr.grad * qmask, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(dk * kmask, kr.grad * kmask, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(dv * kmask, vr.grad * kmask, atol=1e-3, rtol=1e-3)


def _sdpa_ref(q, k, v, q_blk, k_blk, q_valid, k_valid, scale):
    h, hkv = q.shape[2], k.shape[2]
    mask = _dense_mask(q_blk, k_blk, q_valid, k_valid)[:, None]
    qf = q.permute(0, 2, 1, 3)
    kf = k.permute(0, 2, 1, 3).repeat_interleave(h // hkv, dim=1)
    vf = v.permute(0, 2, 1, 3).repeat_interleave(h // hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask, scale=scale)
    return ref.permute(0, 2, 1, 3).float()


# Exercise the Tensor Core (WMMA) path (fp16/bf16) across head dims, GQA/MQA,
# every mask pattern, and padding — the fp32 path uses a different (reference)
# kernel, so WMMA needs its own coverage.
@require_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["causal", "prefixlm", "multiblock", "bidir"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("hkv", [1, 2])
def test_wmma_matches_sdpa(dtype, pattern, head_dim, hkv):
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(0)
    b, sq, h = 2, 70, 4
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, sq, "cuda", pattern, pad_tail=5)
    scale = head_dim**-0.5
    q = torch.randn(b, sq, h, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(b, sq, hkv, head_dim, device="cuda", dtype=dtype)

    out = flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale).float()
    ref = _sdpa_ref(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    vr = q_valid[:, :, None, None].float()
    torch.testing.assert_close(out * vr, ref * vr, atol=2e-2, rtol=2e-2)


@require_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_wmma_cross_attention_non_square_matches_sdpa(dtype):
    """Tensor Core path on the non-square expert/cross-attention layout."""
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(2)
    b, sq, h, hkv, d, cross = 2, 50, 8, 1, 256, 37
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, sq, "cuda", "causal", cross=cross)
    sk = cross + sq
    scale = d**-0.5
    q = torch.randn(b, sq, h, d, device="cuda", dtype=dtype)
    k = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype)
    v = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype)
    out = flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale).float()
    ref = _sdpa_ref(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
    vr = q_valid[:, :, None, None].float()
    torch.testing.assert_close(out * vr, ref * vr, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# 3. GPU integration: the dispatch methods (flash_attention_forward vs
#    sdpa_attention_forward) agree, exercised through the real model methods at
#    the real PaliGemma head config (H=8, MQA Hkv=1, head_dim=256). This covers
#    the integration surface — block-id vs dense-mask consistency, GQA handling,
#    output reshape to (B, S, H*head_dim) — without building the ~3B-param model.
# --------------------------------------------------------------------------- #
@require_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("pattern", ["prefixlm", "causal"])
def test_dispatch_methods_flash_matches_sdpa(pattern):
    from types import SimpleNamespace

    from opentau.policies.pi05.paligemma_with_expert import (
        PaliGemmaWithExpertConfig,
        PaliGemmaWithExpertModel,
    )

    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(0)
    # Default config carries the real PaliGemma head dims (H=8, Hkv=1, head_dim=256).
    cfg = PaliGemmaWithExpertConfig()
    text_cfg = cfg.paligemma_config.text_config
    h = text_cfg.num_attention_heads
    hkv = text_cfg.num_key_value_heads
    d = text_cfg.head_dim
    b, s = 2, 96

    fake = SimpleNamespace(config=cfg)
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, s, "cuda", pattern, pad_tail=4)
    # For square self-attention the pad mask equals q_valid; rebuild the dense
    # mask from the same (pad, att) so the two backends see the same masking.
    dense = make_att_2d_masks(q_valid, _att_masks(pattern, b, s, "cuda"))

    sdpa_out = PaliGemmaWithExpertModel.sdpa_attention_forward(fake, dense, b, d, q, k, v).float()
    flash_out = PaliGemmaWithExpertModel.flash_attention_forward(
        fake, (q_blk, k_blk, q_valid, k_valid), b, d, q, k, v
    ).float()

    assert sdpa_out.shape == flash_out.shape == (b, s, h * d)
    # Compare only valid query rows (padded rows are masked/garbage and discarded).
    vr = q_valid.reshape(b, s, 1).float()
    torch.testing.assert_close(flash_out * vr, sdpa_out * vr, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# 4. GPU perf: memory reduction vs eager (headline win); latency reported
# --------------------------------------------------------------------------- #
@require_cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_memory_reduction_vs_eager(capsys):
    if not flash_attn_cuda.is_available():
        pytest.skip(f"flash_cuda kernel unavailable: {flash_attn_cuda.load_error()}")
    torch.manual_seed(0)
    b, s, h, hkv, d = 2, 1024, 8, 1, 256
    q_blk, k_blk, q_valid, k_valid = _make_blocks(b, s, "cuda", "prefixlm")
    mask = _dense_mask(q_blk, k_blk, q_valid, k_valid)
    scale = d**-0.5
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16)

    def eager():
        g = h // hkv
        qf = q.float().permute(0, 2, 1, 3)
        kf = k.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
        vf = v.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
        aw = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        aw = torch.where(mask[:, None], aw, BIG_NEG)
        p = torch.softmax(aw, dim=-1).to(vf.dtype)
        return torch.matmul(p, vf)

    def flash():
        return flash_attn_blockmask(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1e6

    def latency(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        for _ in range(iters):
            fn()
        e0.record()
        torch.cuda.synchronize()
        return s0.elapsed_time(e0) / iters

    eager_mem, flash_mem = peak(eager), peak(flash)
    eager_ms, flash_ms = latency(eager), latency(flash)
    with capsys.disabled():
        print(
            f"\n[flash_cuda perf] B{b} S{s} H{h} Hkv{hkv} D{d} bf16:"
            f"\n  peak attn mem: eager {eager_mem:.1f} MB | flash {flash_mem:.1f} MB"
            f" ({eager_mem / flash_mem:.2f}x less)"
            f"\n  latency:       eager {eager_ms:.3f} ms | flash {flash_ms:.3f} ms"
            f" ({eager_ms / flash_ms:.2f}x)"
        )
    # Headline guarantee: the fused kernel never materializes the SxS scores/mask,
    # so its peak attention memory is well below eager's.
    assert flash_mem < eager_mem
