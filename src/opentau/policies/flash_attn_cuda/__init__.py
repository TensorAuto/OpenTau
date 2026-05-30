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

"""Custom CUDA FlashAttention with in-kernel block-causal masking.

This package provides a dependency-free (no ``flash-attn``, no ``flex_attention``)
fused attention forward + backward whose block-causal mask is reconstructed
*inside the CUDA kernel* from a compact per-token block-id representation. The
dense ``(B, S, S)`` attention mask is therefore never materialized, which is both
the memory win over the eager backend and a direct way to "handle the block
causal attention masks in the code".

Public API:
  - :func:`make_att_block_ids` -- build the compact ``(q_blk, k_blk, q_valid,
    k_valid)`` representation from the same ``pad_masks`` / ``att_masks`` that
    :func:`make_att_2d_masks` consumes. Provably reproduces the dense mask:
    ``attend(i, j) == q_valid[i] & k_valid[j] & (k_blk[j] <= q_blk[i])``.
  - :func:`flash_attn_blockmask` -- autograd-aware attention given the compact
    representation.
  - :func:`is_available` -- whether the CUDA kernel compiled (else fall back).

Masking convention matches ``make_att_2d_masks``: ``True`` = attend. Cross-
attention key columns get ``k_blk = INT32_MIN`` so they are always attended
(subject to padding).
"""

from __future__ import annotations

import torch

from ._loader import get_extension, is_available, load_error

__all__ = [
    "INT_BLK_MIN",
    "flash_attn_blockmask",
    "is_available",
    "load_error",
    "make_att_block_ids",
]

# Sentinel block-id for cross-attention columns: always <= any real (cumsum)
# block-id, so those columns are unconditionally attended (gated only by
# padding). Must match the int32 INT_MIN the kernel compares against.
INT_BLK_MIN = -(2**31)


def make_att_block_ids(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    n_cross_att_tokens: int | None = None,
    cross_att_pad_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact block-id representation equivalent to ``make_att_2d_masks``.

    Mirrors ``make_att_2d_masks`` exactly but returns four 1-D-per-token tensors
    instead of a dense 2-D mask. The kernel reconstructs the mask via
    ``attend(i, j) == q_valid[i] & k_valid[j] & (k_blk[j] <= q_blk[i])``.

    Args:
        pad_masks: bool ``(B, N)``, ``True`` where the token is real (not padding).
        att_masks: int/bool ``(B, N)``, ``1`` opens a new attention block, ``0``
            continues the current block (the same convention as
            ``make_att_2d_masks``).
        n_cross_att_tokens: if set, prepend that many cross-attention key columns
            (always attended), matching the cross-attention branch of
            ``make_att_2d_masks``.
        cross_att_pad_masks: bool ``(B, n_cross_att_tokens)`` padding for the
            cross columns. Required iff ``n_cross_att_tokens`` is set.

    Returns:
        ``(q_blk, k_blk, q_valid, k_valid)`` where ``q_blk`` is int32 ``(B, N)``,
        ``k_blk`` is int32 ``(B, N)`` or ``(B, n_cross_att_tokens + N)``,
        ``q_valid`` is bool ``(B, N)``, ``k_valid`` matches ``k_blk``'s length.
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}")

    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1).to(torch.int32)
    q_blk = cumsum
    q_valid = pad_masks.to(torch.bool)

    if n_cross_att_tokens is None:
        return q_blk, cumsum, q_valid, q_valid

    assert cross_att_pad_masks is not None, (
        "cross_att_pad_masks must be provided if n_cross_att_tokens is provided"
    )
    assert cross_att_pad_masks.shape == (att_masks.size(0), n_cross_att_tokens), (
        "cross_att_pad_masks must have shape (batch_size, n_cross_att_tokens)"
    )

    b = att_masks.size(0)
    cross_blk = torch.full((b, n_cross_att_tokens), INT_BLK_MIN, dtype=torch.int32, device=att_masks.device)
    k_blk = torch.cat([cross_blk, cumsum], dim=1)
    k_valid = torch.cat([cross_att_pad_masks.to(torch.bool), q_valid], dim=1)
    return q_blk, k_blk, q_valid, k_valid


def _prep(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype).contiguous()


class _FlashAttnBlockMask(torch.autograd.Function):
    """Autograd wrapper around the custom CUDA fwd/bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, q_blk, k_blk, q_valid, k_valid, scale):  # noqa: ANN001
        ext = get_extension()
        if ext is None:
            raise RuntimeError(f"flash_cuda kernel unavailable: {load_error()}")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_blk = _prep(q_blk, torch.int32)
        k_blk = _prep(k_blk, torch.int32)
        q_valid = _prep(q_valid, torch.bool)
        k_valid = _prep(k_valid, torch.bool)
        out, lse = ext.flash_fwd(q, k, v, q_blk, k_blk, q_valid, k_valid, float(scale))
        ctx.save_for_backward(q, k, v, out, lse, q_blk, k_blk, q_valid, k_valid)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: ANN001
        ext = get_extension()
        q, k, v, out, lse, q_blk, k_blk, q_valid, k_valid = ctx.saved_tensors
        dq, dk, dv = ext.flash_bwd(
            grad_out.contiguous(), q, k, v, out, lse, q_blk, k_blk, q_valid, k_valid, ctx.scale
        )
        return dq, dk, dv, None, None, None, None, None


def flash_attn_blockmask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_blk: torch.Tensor,
    k_blk: torch.Tensor,
    q_valid: torch.Tensor,
    k_valid: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Block-causal flash attention.

    Args:
        q: ``(B, Sq, H, D)`` queries.
        k: ``(B, Sk, Hkv, D)`` keys (GQA/MQA: ``H % Hkv == 0``).
        v: ``(B, Sk, Hkv, D)`` values.
        q_blk: int32 ``(B, Sq)`` query block-ids.
        k_blk: int32 ``(B, Sk)`` key block-ids (cross columns use ``INT_BLK_MIN``).
        q_valid: bool ``(B, Sq)`` query padding mask (``True`` = real token).
        k_valid: bool ``(B, Sk)`` key padding mask.
        scale: softmax scale (typically ``head_dim ** -0.5``).

    Returns:
        ``(B, Sq, H, D)`` attention output, same dtype as ``q``.
    """
    return _FlashAttnBlockMask.apply(q, k, v, q_blk, k_blk, q_valid, k_valid, scale)
