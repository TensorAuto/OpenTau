# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ring attention for pi07.

Implements Liu, Zaharia & Abbeel, "Ring Attention with Blockwise
Transformers for Near-Infinite Context" (arXiv:2310.01889) directly on
torch.distributed P2P primitives. Per-block compute uses online softmax
so the (Q_total, K_total) score matrix never materialises on any rank;
the backward recomputes per-block S from saved (q,k,v,LSE) — the
blockwise rematerialization in §3.2.2 of the paper, which replaces the
per-layer torch.utils.checkpoint pi07 used before.

Process group layout for 2-D parallelism with DeepSpeed ZeRO-2:
ring_size * dp_size == world_size. Ring sub-groups are contiguous
([0, R), [R, 2R), ...). The DP axis is implicit: ZeRO runs over WORLD
with the loss pre-scaled by ring_size so that the MEAN reduction over
WORLD collapses to a SUM over ring members and a MEAN over DP — this
avoids issuing collectives on a second NCCL communicator that competes
with ZeRO's on the same CUDA streams.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
from einops import rearrange, repeat

# Finite stand-in for -inf for masked attention scores. Picked well below
# log(fp32_eps) so exp(masked - unmasked_max) underflows cleanly to 0,
# but far enough from fp32's representable limit that compositions with
# (S - LSE) etc. don't cancel into NaN under bf16 mixed-precision.
_NEG_INF = -1.0e9


# --- Process group state ---------------------------------------------------

_RING_GROUP: dist.ProcessGroup | None = None


def set_ring_group(group: dist.ProcessGroup | None) -> None:
    """Pin the ring process group used by ring_attention_forward."""
    global _RING_GROUP
    _RING_GROUP = group


def get_ring_group() -> dist.ProcessGroup | None:
    """Return the active ring group, or None when ring is inactive.

    Inactive cases collapse to a single-device SDPA fallback:
    distributed not available, not initialised, or world size 1.
    """
    if _RING_GROUP is not None:
        return _RING_GROUP
    if not dist.is_available() or not dist.is_initialized():
        return None
    if dist.get_world_size() <= 1:
        return None
    return dist.group.WORLD


def ring_world_size(group: dist.ProcessGroup | None = None) -> int:
    g = group if group is not None else get_ring_group()
    return 1 if g is None else dist.get_world_size(g)


def ring_rank(group: dist.ProcessGroup | None = None) -> int:
    g = group if group is not None else get_ring_group()
    return 0 if g is None else dist.get_rank(g)


def build_ring_groups(ring_size: int) -> dist.ProcessGroup:
    """Build contiguous ring sub-groups of size ``ring_size`` and pin ours.

    With world=8 and ring_size=4 we get rings {0,1,2,3} and {4,5,6,7}.
    DP is handled implicitly by ZeRO over WORLD (see module docstring), so
    we do not create a DP sub-group communicator. When ring_size equals
    world size we reuse the WORLD group rather than duplicating it.

    All ranks must call this in the same order — ``dist.new_group``
    requires every rank participate even in sub-groups they don't join.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("build_ring_groups requires torch.distributed initialised.")
    world = dist.get_world_size()
    rank = dist.get_rank()
    if ring_size < 1 or world % ring_size != 0:
        raise ValueError(f"world_size {world} must be divisible by ring_size {ring_size}.")

    num_rings = world // ring_size
    if num_rings == 1:
        my_group: dist.ProcessGroup = dist.group.WORLD  # type: ignore[assignment]
    else:
        my_group = None  # type: ignore[assignment]
        for rid in range(num_rings):
            ranks = list(range(rid * ring_size, (rid + 1) * ring_size))
            g = dist.new_group(ranks=ranks)
            if rank in ranks:
                my_group = g
        assert my_group is not None
    set_ring_group(my_group)
    return my_group


def dp_rank(ring_size: int | None = None) -> int:
    """This rank's DP index = world_rank // ring_size."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    R = ring_size if ring_size is not None else ring_world_size()
    return 0 if R <= 0 else dist.get_rank() // R


def dp_world_size(ring_size: int | None = None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    R = ring_size if ring_size is not None else ring_world_size()
    return 1 if R <= 0 else dist.get_world_size() // R


# --- Differentiable shard / gather along a seq dim -------------------------


class _AllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup):
        ctx.seq_dim = seq_dim
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        x_c = x.contiguous()
        moved = x_c.transpose(0, seq_dim).contiguous()
        out = torch.empty(
            (moved.shape[0] * ctx.world_size, *moved.shape[1:]),
            dtype=moved.dtype,
            device=moved.device,
        )
        dist.all_gather_into_tensor(out, moved, group=group)
        return out.transpose(0, seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        chunks = torch.chunk(grad_out, ctx.world_size, dim=ctx.seq_dim)
        return chunks[ctx.rank].contiguous(), None, None


def split_seq(x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Return this rank's contiguous slice along ``seq_dim``.

    Length along ``seq_dim`` must be divisible by ring world size; the
    caller pads upstream and arranges for the corresponding mask /
    position_ids slices to mask the pad inert.
    """
    g = group if group is not None else get_ring_group()
    ws = 1 if g is None else dist.get_world_size(g)
    if ws == 1:
        return x
    L = x.shape[seq_dim]
    if L % ws != 0:
        raise ValueError(f"seq length {L} along dim {seq_dim} not divisible by ring ws {ws}.")
    return torch.chunk(x, ws, dim=seq_dim)[dist.get_rank(g)].contiguous()


def gather_seq(x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Differentiable all-gather along ``seq_dim``. Backward = take-this-rank's-slice."""
    g = group if group is not None else get_ring_group()
    ws = 1 if g is None else dist.get_world_size(g)
    if ws == 1:
        return x
    return _AllGatherSeq.apply(x, seq_dim, g)


def broadcast_batch_within_ring(batch) -> None:
    """In-place broadcast every CUDA tensor in ``batch`` from each ring's
    lowest-ranked member to its followers.

    Belt-and-braces guarantee that all ranks within a ring see identical
    sample data — they must, because ring attention computes a single
    logical forward pass distributed across the ring. Seeding by DP
    index narrows the gap but doesn't cover worker-side stochasticity
    (random augmentation, dataloader pickling order) reliably.

    No-op when ring is inactive.
    """
    g = get_ring_group()
    if g is None or dist.get_world_size(g) <= 1:
        return
    my_local = dist.get_rank(g)
    my_global = dist.get_rank()
    src_global = my_global - my_local  # local rank 0 within the ring

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                dist.broadcast(obj, src=src_global, group=g)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(batch)


# --- Ring rotation (P2P with global rank translation) ----------------------


def _ring_rotate(
    *tensors: torch.Tensor,
    group: dist.ProcessGroup,
    forward_dir: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Rotate ``tensors`` one step around the ring.

    forward_dir=True sends to rank+1, receives from rank-1 (used in fwd).
    forward_dir=False sends to rank-1, receives from rank+1 (used in bwd
    to walk dK/dV opposite the K/V they belong to).

    ``P2POp`` requires the peer argument to be a GLOBAL rank even when
    ``group`` is a sub-group; we translate through ``get_global_rank``.
    """
    ws = dist.get_world_size(group)
    if ws == 1:
        return tuple(tensors)
    local = dist.get_rank(group)
    if forward_dir:
        send_to_local = (local + 1) % ws
        recv_from_local = (local - 1) % ws
    else:
        send_to_local = (local - 1) % ws
        recv_from_local = (local + 1) % ws
    send_to_global = dist.get_global_rank(group, send_to_local)
    recv_from_global = dist.get_global_rank(group, recv_from_local)

    ops: list[dist.P2POp] = []
    recv_buffers: list[torch.Tensor] = []
    for t in tensors:
        tc = t.contiguous()
        rb = torch.empty_like(tc)
        ops.append(dist.P2POp(dist.isend, tc, send_to_global, group=group))
        ops.append(dist.P2POp(dist.irecv, rb, recv_from_global, group=group))
        recv_buffers.append(rb)
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    return tuple(recv_buffers)


# --- GQA expand / reduce ---------------------------------------------------


def _expand_kv(t: torch.Tensor, num_kv: int, num_q: int) -> torch.Tensor:
    """(B, S, Hkv, D) -> (B, S, Hq, D) by repeating each KV head."""
    if num_kv == num_q:
        return t
    g = num_q // num_kv
    return repeat(t, "b s h d -> b s (h g) d", g=g)


def _reduce_kv_grad(g: torch.Tensor, num_kv: int, num_q: int) -> torch.Tensor:
    """Inverse of :func:`_expand_kv` for grads — sum-reduce groups back."""
    if num_kv == num_q:
        return g
    groups = num_q // num_kv
    return rearrange(g, "b s (h gr) d -> b s h gr d", gr=groups).sum(dim=3)


# --- Core ring autograd Function ------------------------------------------

_RING_Q_TILE = 256


def set_ring_q_tile(tile: int) -> None:
    global _RING_Q_TILE
    _RING_Q_TILE = int(tile)


def _compute_block(
    q: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    mask_block: torch.Tensor,
    scaling: float,
    num_q: int,
    num_kv: int,
):
    """Return (S [B,Hq,Sq,Sk], v32, k32) for one (Q-tile, K-block).

    fp32 score + softmax accumulator matches eager_attention_forward to
    within bf16 reassociation noise. Per-tile, not global — peak score
    memory is O(B*Hq*q_tile*Sk_block), not O(B*Hq*Sq*Sk_total).
    """
    k_e = _expand_kv(k_block, num_kv, num_q)
    v_e = _expand_kv(v_block, num_kv, num_q)
    q32 = q.to(torch.float32)
    k32 = k_e.to(torch.float32)
    v32 = v_e.to(torch.float32)
    S = torch.einsum("bqhd,bkhd->bhqk", q32, k32) * scaling
    S = torch.where(mask_block.unsqueeze(1), S, torch.full_like(S, _NEG_INF))
    return S, v32, k32


class _RingAttention(torch.autograd.Function):
    """Ring attention with online softmax (fwd) and blockwise remat (bwd).

    Saved tensors are per-rank shards plus the final output and LSE; full
    score matrices and per-block softmax outputs are recomputed in
    backward from saved (q,k,v,LSE) — the blockwise rematerialization in
    Liu et al. §3.2.2.
    """

    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,  # (B, Sq_local, Hq, D)
        k_local: torch.Tensor,  # (B, Sk_local, Hkv, D)
        v_local: torch.Tensor,  # (B, Sk_local, Hkv, D)
        attn_mask: torch.Tensor,  # (B, Sq_total, Sk_total) bool, replicated
        scaling: float,
        num_q: int,
        num_kv: int,
        group: dist.ProcessGroup,
        q_lens: tuple[int, ...],
        k_lens: tuple[int, ...],
    ) -> torch.Tensor:
        ws = dist.get_world_size(group)
        rank = dist.get_rank(group)

        B, Sq, Hq, D = q_local.shape
        device = q_local.device
        out_dtype = q_local.dtype

        # Per-rank Q and K offsets into the replicated mask.
        q_off = [0]
        for L in q_lens:
            q_off.append(q_off[-1] + L)
        k_off = [0]
        for L in k_lens:
            k_off.append(k_off[-1] + L)
        my_q_lo = q_off[rank]

        m = torch.full((B, Hq, Sq), _NEG_INF, device=device, dtype=torch.float32)
        l = torch.zeros((B, Hq, Sq), device=device, dtype=torch.float32)
        O = torch.zeros((B, Sq, Hq, D), device=device, dtype=torch.float32)

        k_cur = k_local.contiguous()
        v_cur = v_local.contiguous()
        owner = rank
        q_tile = min(_RING_Q_TILE, Sq)

        for step in range(ws):
            k_lo, k_hi = k_off[owner], k_off[owner + 1]

            for qs in range(0, Sq, q_tile):
                qe = min(qs + q_tile, Sq)
                q_tile_local = q_local[:, qs:qe].contiguous()
                mt = attn_mask[:, my_q_lo + qs : my_q_lo + qe, k_lo:k_hi]

                S, v32, _ = _compute_block(q_tile_local, k_cur, v_cur, mt, scaling, num_q, num_kv)
                m_s = m[:, :, qs:qe]
                l_s = l[:, :, qs:qe]
                O_s = O[:, qs:qe]

                block_max = S.max(dim=-1).values  # (B, Hq, q_tile)
                m_new = torch.maximum(m_s, block_max)
                # Fully-masked-row safety: when block_max == _NEG_INF and m_s
                # == _NEG_INF, m_new == _NEG_INF; alpha = exp(0) = 1 and
                # P = exp(0) = 1 so l accumulates Sk_block. l > 0 holds for
                # every row by the time forward returns, so O/l is finite.
                alpha = torch.exp(m_s - m_new)
                P = torch.exp(S - m_new.unsqueeze(-1))
                l_new = alpha * l_s + P.sum(dim=-1)
                O_new = alpha.permute(0, 2, 1).unsqueeze(-1) * O_s + torch.einsum("bhqk,bkhd->bqhd", P, v32)
                m[:, :, qs:qe] = m_new
                l[:, :, qs:qe] = l_new
                O[:, qs:qe] = O_new

            if step < ws - 1:
                k_cur, v_cur = _ring_rotate(k_cur, v_cur, group=group, forward_dir=True)
                owner = (owner - 1) % ws

        O = O / l.permute(0, 2, 1).unsqueeze(-1)
        LSE = m + torch.log(l)

        # Mask any row that *never* saw an unmasked key. l == Sk_total
        # exactly for those (P=1 everywhere). Test on m instead: m stays
        # _NEG_INF iff no real score was ever observed. Zero out so they
        # contribute nothing to downstream loss / gradients.
        no_real_score = m <= _NEG_INF / 2  # (B, Hq, Sq)
        if no_real_score.any():
            O = torch.where(no_real_score.permute(0, 2, 1).unsqueeze(-1), torch.zeros_like(O), O)
            LSE = torch.where(no_real_score, torch.zeros_like(LSE), LSE)

        ctx.save_for_backward(q_local, k_local, v_local, O, LSE, attn_mask, no_real_score)
        ctx.scaling = scaling
        ctx.num_q = num_q
        ctx.num_kv = num_kv
        ctx.group = group
        ctx.q_off = tuple(q_off)
        ctx.k_off = tuple(k_off)
        return O.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_O: torch.Tensor):
        q_local, k_local, v_local, O, LSE, attn_mask, no_real_score = ctx.saved_tensors
        scaling = ctx.scaling
        num_q = ctx.num_q
        num_kv = ctx.num_kv
        group = ctx.group
        q_off = ctx.q_off
        k_off = ctx.k_off

        ws = dist.get_world_size(group)
        rank = dist.get_rank(group)
        my_q_lo = q_off[rank]

        # Zero grad on no-real-score rows so they don't propagate junk back.
        grad_O32 = grad_O.to(torch.float32)
        if no_real_score.any():
            grad_O32 = torch.where(
                no_real_score.permute(0, 2, 1).unsqueeze(-1),
                torch.zeros_like(grad_O32),
                grad_O32,
            )

        O32 = O.to(torch.float32)
        D = (O32 * grad_O32).sum(dim=-1)  # (B, Sq, Hq)
        D_bhq = D.permute(0, 2, 1)  # (B, Hq, Sq)

        dq_local = torch.zeros_like(q_local, dtype=torch.float32)
        dk_cur = torch.zeros_like(k_local, dtype=torch.float32)
        dv_cur = torch.zeros_like(v_local, dtype=torch.float32)

        k_cur = k_local.contiguous()
        v_cur = v_local.contiguous()
        owner = rank
        Sq = q_local.shape[1]
        q_tile = min(_RING_Q_TILE, Sq)

        for step in range(ws):
            k_lo, k_hi = k_off[owner], k_off[owner + 1]

            for qs in range(0, Sq, q_tile):
                qe = min(qs + q_tile, Sq)
                q_tile_local = q_local[:, qs:qe].contiguous()
                mt = attn_mask[:, my_q_lo + qs : my_q_lo + qe, k_lo:k_hi]

                S, v32, k32 = _compute_block(q_tile_local, k_cur, v_cur, mt, scaling, num_q, num_kv)
                LSE_tile = LSE[:, :, qs:qe]
                # For no_real_score rows LSE is 0; S is _NEG_INF; P =
                # exp(_NEG_INF) ≈ 0, so dV/dK/dQ contributions for those
                # rows underflow to 0 — exactly what we want.
                P = torch.exp(S - LSE_tile.unsqueeze(-1))

                grad_O_tile = grad_O32[:, qs:qe]
                D_tile = D_bhq[:, :, qs:qe]

                dv_tile_e = torch.einsum("bhqk,bqhd->bkhd", P, grad_O_tile)
                dv_tile = _reduce_kv_grad(dv_tile_e, num_kv, num_q)

                dP = torch.einsum("bqhd,bkhd->bhqk", grad_O_tile, v32)
                dS = P * (dP - D_tile.unsqueeze(-1))

                dq_local[:, qs:qe] = dq_local[:, qs:qe] + torch.einsum("bhqk,bkhd->bqhd", dS, k32) * scaling
                dk_tile_e = torch.einsum("bhqk,bqhd->bkhd", dS, q_tile_local.to(torch.float32)) * scaling
                dk_tile = _reduce_kv_grad(dk_tile_e, num_kv, num_q)

                dk_cur = dk_cur + dk_tile
                dv_cur = dv_cur + dv_tile

            if step < ws - 1:
                k_cur, v_cur, dk_cur, dv_cur = _ring_rotate(
                    k_cur, v_cur, dk_cur, dv_cur, group=group, forward_dir=True
                )
                owner = (owner - 1) % ws

        # One more rotation in fwd direction brings dk/dv back to their owner.
        if ws > 1:
            dk_cur, dv_cur = _ring_rotate(dk_cur, dv_cur, group=group, forward_dir=True)

        return (
            dq_local.to(q_local.dtype),
            dk_cur.to(k_local.dtype),
            dv_cur.to(v_local.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# --- Public entry point ----------------------------------------------------


def _sdpa_fallback(
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q: int,
    num_kv: int,
    scaling: float,
) -> torch.Tensor:
    """Numerically-equivalent SDPA path for ws==1 / no distributed."""
    k_e = _expand_kv(k, num_kv, num_q)
    v_e = _expand_kv(v, num_kv, num_q)
    q_t = q.transpose(1, 2)
    k_t = k_e.transpose(1, 2)
    v_t = v_e.transpose(1, 2)
    attn = attention_mask[:, None, :, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn,
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
    )
    return out.permute(0, 2, 1, 3).reshape(batch_size, -1, num_q * head_dim)


def ring_attention_forward(
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    scaling: float | None = None,
    q_lengths_per_rank: Sequence[int] | None = None,
    k_lengths_per_rank: Sequence[int] | None = None,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Drop-in replacement for eager_/sdpa_attention_forward.

    Inputs are the local per-rank shards of Q/K/V; ``attention_mask`` is the
    *replicated* (B, Q_total, K_total) bool mask. Returns the local slice of
    attention output as (B, Sq_local, num_query_heads * head_dim).
    """
    g = group if group is not None else get_ring_group()
    ws = ring_world_size(g)
    if scaling is None:
        scaling = head_dim**-0.5

    if ws == 1 or g is None:
        return _sdpa_fallback(
            attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
            num_query_heads,
            num_kv_heads,
            scaling,
        )

    if q_lengths_per_rank is None:
        q_lengths_per_rank = (query_states.shape[1],) * ws
    if k_lengths_per_rank is None:
        k_lengths_per_rank = (key_states.shape[1],) * ws

    out_local = _RingAttention.apply(
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
        num_query_heads,
        num_kv_heads,
        g,
        tuple(q_lengths_per_rank),
        tuple(k_lengths_per_rank),
    )
    return out_local.reshape(batch_size, -1, num_query_heads * head_dim)
