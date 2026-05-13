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

"""Ring attention for pi07.

Implements the algorithm from Liu, Zaharia & Abbeel,
"Ring Attention with Blockwise Transformers for Near-Infinite Context"
(https://arxiv.org/abs/2310.01889).

The implementation is intentionally written in plain PyTorch on top of
``torch.distributed`` P2P primitives (``isend`` / ``irecv``), so it works on
any NCCL-capable cluster (A100 + NVLink + InfiniBand, the production target)
without depending on a kernel library that doesn't support pi07's
block-causal mask. The per-block compute uses online softmax, so the
``(Q_total, K_total)`` attention-scores matrix is never materialised on any
rank — peak attention memory scales with the per-rank block size rather
than the global sequence length. This blockwise rematerialisation in the
backward pass is the same memory-saving mechanism described in the paper's
Section 3.2.2 and replaces the per-layer ``torch.utils.checkpoint`` that
pi07 used previously.

The module supports pi07's general 2-D boolean attention mask (the
cumsum-based block-causal pattern produced by
:func:`opentau.policies.pi07.low_level.modeling_pi07_low_level.make_att_2d_masks`),
arbitrary GQA ratios (``num_attention_heads`` >= ``num_key_value_heads``),
and falls back transparently to the existing SDPA path when the ring
process group has world size 1 or distributed is not initialised — both
useful for unit tests and single-GPU debugging.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
from einops import rearrange, repeat

# A finite stand-in for -inf so masked attention scores still survive a
# ``torch.maximum`` / ``torch.exp`` chain without producing NaNs when an
# entire row is masked out (e.g. padding rows). Same constant as the
# eager attention forward in ``gemma3_with_expert.py``.
_NEG_INF = -2.3819763e38


# ---------------------------------------------------------------------------
# Process group plumbing
# ---------------------------------------------------------------------------

_RING_GROUP: dist.ProcessGroup | None = None


def set_ring_group(group: dist.ProcessGroup | None) -> None:
    """Pin a specific process group to use for ring attention.

    When None (the default), :func:`get_ring_group` returns the global
    default group. Override this when combining ring attention with another
    form of parallelism on a different axis (e.g. ring along a sub-group
    while DP-replicates across orthogonal groups).
    """
    global _RING_GROUP
    _RING_GROUP = group


def get_ring_group() -> dist.ProcessGroup | None:
    """Return the active ring process group, or None when ring is inactive.

    Inactive cases (caller should fall back to single-device attention):
      * ``torch.distributed`` is not available, or
      * distributed has not been initialised, or
      * the active group has world size 1.
    """
    if _RING_GROUP is not None:
        return _RING_GROUP
    if not dist.is_available() or not dist.is_initialized():
        return None
    if dist.get_world_size() <= 1:
        return None
    return dist.group.WORLD


def ring_world_size(group: dist.ProcessGroup | None = None) -> int:
    group = group if group is not None else get_ring_group()
    return 1 if group is None else dist.get_world_size(group)


def ring_rank(group: dist.ProcessGroup | None = None) -> int:
    group = group if group is not None else get_ring_group()
    return 0 if group is None else dist.get_rank(group)


# Module-global DP sub-group, populated alongside the ring sub-group by
# :func:`build_ring_and_dp_groups`. Kept here (not in the trainer) so any
# code path can look it up without importing trainer internals.
_DP_GROUP: dist.ProcessGroup | None = None


def get_dp_group() -> dist.ProcessGroup | None:
    """Return the DP-replication process group, or None when ring is inactive.

    The DP group contains one rank from each ring sub-group at the same
    intra-ring offset (i.e. ranks ``r``, ``r + ring_group_size``,
    ``r + 2 * ring_group_size`` ... for offset ``r``). When ring spans the
    full world (``ring_group_size == world_size`` or unset), there is no
    DP axis — this returns None.

    Returns None when distributed is not initialised so single-GPU callers
    can use the same code path.
    """
    return _DP_GROUP


def build_ring_and_dp_groups(ring_group_size: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup | None]:
    """Construct the ring sub-group; expose DP topology arithmetically.

    Ranks ``[0, R), [R, 2R), ...`` form ring sub-groups of size ``R =
    ring_group_size``. The DP axis (one rank from each ring sub-group at
    the same intra-ring offset) is conceptual: pi07's only ZeRO + ring
    integration uses ZeRO over the WORLD group with a ``loss *= R``
    pre-backward to correct the MEAN-reduction, so no actual DP-axis
    collective is issued. Skipping the DP ``new_group`` calls keeps the
    NCCL communicator count minimal — that matters because every extra
    sub-group communicator competes with ZeRO's world-group collectives
    on the same CUDA streams, and NCCL is documented as unsafe under
    concurrent multi-communicator use on shared streams. Empirically, on
    A100 + DeepSpeed ZeRO-2, creating the orthogonal DP groups was enough
    to hang the first training step.

    When ``ring_group_size == world_size`` we reuse the WORLD group as
    the ring group instead of creating a duplicate communicator — same
    rationale.

    All ranks must call this function in the same order (it issues
    ``dist.new_group`` collectives that every rank participates in, even
    for sub-groups they don't belong to). Calls :func:`set_ring_group` so
    subsequent ``get_ring_group()`` returns the correct sub-group.

    Args:
        ring_group_size: Number of ranks per ring sub-group. Must divide
            world size. ``ring_group_size == world_size`` is the
            single-ring case (no DP replication).

    Returns:
        ``(ring_group, dp_group)``. ``dp_group`` is always None today —
        the DP axis is handled implicitly by ZeRO over WORLD. The tuple
        shape is kept for forward-compatibility with future call sites
        that may want an explicit DP communicator.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("build_ring_and_dp_groups requires torch.distributed to be initialised.")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % ring_group_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ring_group_size ({ring_group_size})."
        )
    num_rings = world_size // ring_group_size

    if num_rings == 1:
        # Single ring spanning WORLD — no need for a duplicate communicator.
        my_ring_group: dist.ProcessGroup = dist.group.WORLD  # type: ignore[assignment]
    else:
        # Multiple ring sub-groups. Every rank must call ``new_group`` for
        # every sub-group (NCCL contract for sub-group creation).
        my_ring_group = None  # type: ignore[assignment]
        for ring_id in range(num_rings):
            ranks = list(range(ring_id * ring_group_size, (ring_id + 1) * ring_group_size))
            g = dist.new_group(ranks=ranks)
            if rank in ranks:
                my_ring_group = g
        assert my_ring_group is not None, "rank not assigned to any ring sub-group"

    set_ring_group(my_ring_group)
    global _DP_GROUP
    _DP_GROUP = None
    return my_ring_group, None


def get_dp_rank(ring_group_size: int | None = None) -> int:
    """Return this rank's DP index, computed arithmetically.

    The DP index identifies which ring sub-group a rank belongs to. With
    contiguous ring sub-groups (rank ``[0, R)``, ``[R, 2R)``, ...) the
    DP index is just ``world_rank // R``. When ``ring_group_size`` is
    None we infer it from :func:`ring_world_size`, which returns the
    pinned ring sub-group's size after :func:`build_ring_and_dp_groups`
    has run.

    Used by the trainer to seed the dataloader sampler so ranks in the
    same ring sub-group draw an identical sample stream; this is the only
    consumer of the DP-axis identity in the current implementation, so
    expressing it as an integer keeps us out of NCCL sub-group territory.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    world_rank = dist.get_rank()
    if ring_group_size is None:
        ring_group_size = ring_world_size()
    if ring_group_size <= 0:
        return 0
    return world_rank // ring_group_size


def get_dp_world_size(ring_group_size: int | None = None) -> int:
    """Number of DP-replicas across the world (``world_size / ring_group_size``)."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    if ring_group_size is None:
        ring_group_size = ring_world_size()
    if ring_group_size <= 0:
        return 1
    return dist.get_world_size() // ring_group_size


def _broadcast_batch_in_ring(batch, src_offset: int = 0) -> None:
    """Broadcast every tensor in ``batch`` from the ring leader to its followers.

    The trainer calls this on every step under 2D parallelism so the
    sequence-parallel ranks within a ring sub-group provably agree on the
    batch they are jointly attending over. The seeded sampler already
    arranges that — this broadcast is a cheap belt-and-braces guard
    against any worker-side stochasticity (random augmentations, etc.)
    that could otherwise let the per-rank batches diverge.

    Args:
        batch: A nested container (dict / list / tuple) of ``torch.Tensor``
            and arbitrary scalar metadata. Tensors are broadcast in place;
            non-tensor leaves are left untouched (every rank in the
            sub-group already has them deterministically by virtue of the
            shared sampler stream).
        src_offset: Intra-ring offset of the broadcast source. Defaults to
            ``0`` (the rank with the lowest global rank in the sub-group).

    No-op when the ring sub-group has size 1 (or no ring is active).
    """
    group = get_ring_group()
    if group is None or dist.get_world_size(group) <= 1:
        return
    # Translate intra-ring offset to global rank: ring sub-groups are
    # contiguous, so the source is ``my_global_rank - my_local_rank +
    # src_offset``.
    my_local = dist.get_rank(group)
    my_global = dist.get_rank()
    src_global = my_global - my_local + src_offset

    def _walk(obj):
        if isinstance(obj, torch.Tensor):
            # ``broadcast`` requires the tensor to be on a CUDA device for
            # NCCL; CPU tensors would need GLOO. We assume training has
            # already moved inputs to GPU by this point (pin_memory + the
            # implicit device move accelerate does); guard otherwise.
            if obj.is_cuda:
                dist.broadcast(obj, src=src_global, group=group)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(batch)


# ---------------------------------------------------------------------------
# Sharding helpers (used by the surrounding model to feed per-rank shards
# into the attention forward).
# ---------------------------------------------------------------------------


def split_seq(x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Returns this rank's contiguous slice of ``x`` along ``seq_dim``.

    Length along ``seq_dim`` must be divisible by ``world_size``. Callers are
    responsible for padding to a multiple of world size before sharding —
    pi07's masks already zero-out padded positions so the contribution is
    inert as long as the mask is sliced consistently.
    """
    ws = ring_world_size(group)
    if ws == 1:
        return x
    total = x.shape[seq_dim]
    if total % ws != 0:
        raise ValueError(
            f"sequence length {total} along dim {seq_dim} is not divisible by ring world_size {ws}; "
            f"pad to a multiple of world_size before sharding."
        )
    chunks = torch.chunk(x, ws, dim=seq_dim)
    return chunks[ring_rank(group)].contiguous()


def gather_seq(x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Differentiable all-gather along ``seq_dim`` across the ring group.

    Backward is a take-this-rank's-slice operation on the gradient, which is
    what reduce-scatter would give us; we implement it directly to keep the
    forward fast.
    """
    ws = ring_world_size(group)
    if ws == 1:
        return x
    return _AllGatherSeq.apply(x, seq_dim, group)


class _AllGatherSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, seq_dim: int, group: dist.ProcessGroup):
        ctx.seq_dim = seq_dim
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        x_contig = x.contiguous()
        # all_gather_into_tensor concatenates along dim 0; move seq_dim there
        # for the collective, then move it back.
        moved = x_contig.transpose(0, seq_dim).contiguous()
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


# ---------------------------------------------------------------------------
# Ring rotation primitive
# ---------------------------------------------------------------------------


def _ring_rotate(
    *tensors: torch.Tensor,
    group: dist.ProcessGroup,
    forward_dir: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Rotate ``tensors`` one step around the ring.

    ``forward_dir=True`` sends to rank+1 and receives from rank-1; used in
    the forward pass to walk K/V around the ring. ``forward_dir=False``
    sends to rank-1 and receives from rank+1; used in the backward to walk
    dK/dV and (recomputed) K/V the opposite way so each rank ends up with
    gradients for the K/V it originally owned.
    """
    ws = dist.get_world_size(group)
    if ws == 1:
        return tuple(tensors)
    rank = dist.get_rank(group)
    if forward_dir:
        send_to = (rank + 1) % ws
        recv_from = (rank - 1) % ws
    else:
        send_to = (rank - 1) % ws
        recv_from = (rank + 1) % ws

    ops: list[dist.P2POp] = []
    recv_buffers: list[torch.Tensor] = []
    for t in tensors:
        t_c = t.contiguous()
        recv = torch.empty_like(t_c)
        ops.append(dist.P2POp(dist.isend, t_c, send_to, group=group))
        ops.append(dist.P2POp(dist.irecv, recv, recv_from, group=group))
        recv_buffers.append(recv)
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return tuple(recv_buffers)


# ---------------------------------------------------------------------------
# GQA helpers
# ---------------------------------------------------------------------------


def _expand_kv(t: torch.Tensor, num_kv_heads: int, num_query_heads: int) -> torch.Tensor:
    """Expand ``(B, S, Hkv, Dh)`` to ``(B, S, Hq, Dh)`` by repeating each KV head.

    GQA: each KV head is shared by ``num_query_heads // num_kv_heads`` query
    heads, so the expansion is just a repeat along the head axis. We use
    ``repeat`` rather than ``expand`` because subsequent kernels (matmul into
    the score tensor) prefer a materialised tensor for predictable strides.
    """
    if num_kv_heads == num_query_heads:
        return t
    groups = num_query_heads // num_kv_heads
    return repeat(t, "b s h d -> b s (h g) d", g=groups)


def _reduce_kv_grad(g: torch.Tensor, num_kv_heads: int, num_query_heads: int) -> torch.Tensor:
    """Inverse of :func:`_expand_kv` for gradients.

    Sum-reduce the per-query-head gradient back into one gradient per KV
    head. Shape ``(B, S, Hq, Dh)`` -> ``(B, S, Hkv, Dh)``.
    """
    if num_kv_heads == num_query_heads:
        return g
    groups = num_query_heads // num_kv_heads
    return rearrange(g, "b s (h g) d -> b s h g d", g=groups).sum(dim=3)


# ---------------------------------------------------------------------------
# Ring attention autograd Function
# ---------------------------------------------------------------------------


# Per-block Q tile size. Memory of the per-block ``(B, H, q_tile, Sk_block)``
# score / probability tensors scales with this; smaller is more memory-
# frugal but costs a small constant on the matmul side. 256 is a good
# default in practice — flash-attention's ``BLOCK_M`` is in the same
# ballpark for similar head counts. Override via ``set_ring_q_tile`` for
# very wide / very narrow models.
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
    num_query_heads: int,
    num_kv_heads: int,
):
    """Per-block attention math used by both forward and backward.

    Returns the masked, scaled score matrix ``S`` (in fp32) plus the fp32
    expansions of K, V that share its layout. Kept here so forward and
    backward can share the same numeric recipe. The caller is responsible
    for the online-softmax update around it.
    """
    k_exp = _expand_kv(k_block, num_kv_heads, num_query_heads)
    v_exp = _expand_kv(v_block, num_kv_heads, num_query_heads)
    # Cast to fp32 for the score matmul + softmax accumulator. The cost is
    # localised to one block at a time; this matches eager_attention_forward
    # and lets us match its numerical output to within bf16 reassociation
    # noise.
    q32 = q.to(torch.float32)
    k32 = k_exp.to(torch.float32)
    v32 = v_exp.to(torch.float32)
    # S: (B, Hq, Sq, Sk_block)
    S = torch.einsum("bqhd,bkhd->bhqk", q32, k32) * scaling
    # mask_block: (B, Sq, Sk_block) — broadcast across heads.
    S = torch.where(mask_block.unsqueeze(1), S, torch.full_like(S, _NEG_INF))
    return S, v32, k32


class _RingAttention(torch.autograd.Function):
    """Custom autograd op that walks K/V around the ring with online softmax.

    Saved tensors are intentionally limited to the per-rank shards plus the
    final output and log-sum-exp: full ``(Q_total, K_total)`` attention
    matrices are never stored, and per-block score matrices ``S`` and their
    softmax exponentials are recomputed during backward. This is the
    blockwise rematerialisation described in the paper's Section 3.2.2.

    Inputs:
        q_local:   ``(B, Sq_local, Hq, Dh)`` — this rank's slice of Q.
        k_local:   ``(B, Sk_local, Hkv, Dh)`` — this rank's slice of K.
        v_local:   ``(B, Sk_local, Hkv, Dh)`` — this rank's slice of V.
        attn_mask: ``(B, Sq_total, Sk_total)`` bool — *full* mask, replicated
                   on every rank. Sliced per ring step.
        scaling: pre-softmax scale factor (``query_pre_attn_scalar ** -0.5``
                 for Gemma 3).
        num_query_heads, num_kv_heads: GQA head counts.
        group: ring process group. Must have world size >= 2; the surrounding
               wrapper falls back to SDPA when there's nothing to ring.
        q_lengths_per_rank / k_lengths_per_rank: per-rank seq lengths along
            Q and K axes. Used to slice the replicated mask. Lengths can
            differ across ranks (sequence length need not be exactly
            divisible by world size, although the surrounding sharder
            currently always pads to make it so).
    """

    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        attn_mask: torch.Tensor,
        scaling: float,
        num_query_heads: int,
        num_kv_heads: int,
        group: dist.ProcessGroup,
        q_lengths_per_rank: tuple[int, ...],
        k_lengths_per_rank: tuple[int, ...],
    ) -> torch.Tensor:
        ws = dist.get_world_size(group)
        rank = dist.get_rank(group)

        B, Sq_local, Hq, Dh = q_local.shape
        device = q_local.device
        out_dtype = q_local.dtype

        # Mask slicing offsets along Q (constant) and K (changes per step).
        q_offsets = [0]
        for L in q_lengths_per_rank:
            q_offsets.append(q_offsets[-1] + L)
        k_offsets = [0]
        for L in k_lengths_per_rank:
            k_offsets.append(k_offsets[-1] + L)
        my_q_lo = q_offsets[rank]

        # Running stats in fp32 — same convention as flash-attn's LSE.
        m = torch.full((B, Hq, Sq_local), _NEG_INF, device=device, dtype=torch.float32)
        l = torch.zeros((B, Hq, Sq_local), device=device, dtype=torch.float32)
        O = torch.zeros((B, Sq_local, Hq, Dh), device=device, dtype=torch.float32)

        k_cur = k_local.contiguous()
        v_cur = v_local.contiguous()
        owner = rank  # which rank's K/V do we currently hold
        q_tile = min(_RING_Q_TILE, Sq_local)

        for step in range(ws):
            k_lo, k_hi = k_offsets[owner], k_offsets[owner + 1]

            # Tile across Q so the per-iteration ``(B, H, q_tile, Sk_block)``
            # score / probability tensors stay bounded by ``q_tile``. Without
            # tiling, peak memory scales with ``Sq_local * Sk_block`` per
            # ring step, which dwarfs SDPA's flash-attention backend on long
            # contexts and defeats the whole point of ring attention.
            for q_start in range(0, Sq_local, q_tile):
                q_end = min(q_start + q_tile, Sq_local)
                q_tile_local = q_local[:, q_start:q_end].contiguous()
                mask_tile = attn_mask[:, my_q_lo + q_start : my_q_lo + q_end, k_lo:k_hi]

                S, v32, _k32 = _compute_block(
                    q_tile_local, k_cur, v_cur, mask_tile, scaling, num_query_heads, num_kv_heads
                )
                m_slice = m[:, :, q_start:q_end]
                l_slice = l[:, :, q_start:q_end]
                O_slice = O[:, q_start:q_end]

                block_max = S.max(dim=-1).values  # (B, Hq, q_tile)
                m_new = torch.maximum(m_slice, block_max)
                alpha = torch.exp(m_slice - m_new)
                P = torch.exp(S - m_new.unsqueeze(-1))
                l_new = alpha * l_slice + P.sum(dim=-1)
                O_new = alpha.permute(0, 2, 1).unsqueeze(-1) * O_slice + torch.einsum(
                    "bhqk,bkhd->bqhd", P, v32
                )
                m[:, :, q_start:q_end] = m_new
                l[:, :, q_start:q_end] = l_new
                O[:, q_start:q_end] = O_new

            if step < ws - 1:
                k_cur, v_cur = _ring_rotate(k_cur, v_cur, group=group, forward_dir=True)
                owner = (owner - 1) % ws

        # Finalize: divide by accumulator l. LSE saved for backward.
        O = O / l.permute(0, 2, 1).unsqueeze(-1)
        LSE = m + torch.log(l)  # (B, Hq, Sq_local) — fp32

        # Save for backward.
        ctx.save_for_backward(q_local, k_local, v_local, O, LSE, attn_mask)
        ctx.scaling = scaling
        ctx.num_query_heads = num_query_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.group = group
        ctx.q_offsets = tuple(q_offsets)
        ctx.k_offsets = tuple(k_offsets)
        return O.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_O: torch.Tensor):
        # Recover saved state.
        q_local, k_local, v_local, O, LSE, attn_mask = ctx.saved_tensors
        scaling = ctx.scaling
        num_query_heads = ctx.num_query_heads
        num_kv_heads = ctx.num_kv_heads
        group = ctx.group
        q_offsets = ctx.q_offsets
        k_offsets = ctx.k_offsets

        ws = dist.get_world_size(group)
        rank = dist.get_rank(group)
        my_q_lo = q_offsets[rank]

        # Cast incoming grad and saved O to fp32 for the accumulator math.
        grad_O32 = grad_O.to(torch.float32)
        O32 = O.to(torch.float32)
        # Row-wise correction term D = sum_k (O * dO). Per (B, Sq, Hq).
        D = (O32 * grad_O32).sum(dim=-1)  # (B, Sq_local, Hq)
        # Move heads to dim 1 for use in dS computation.
        D_bhq = D.permute(0, 2, 1)  # (B, Hq, Sq_local)

        # Outputs we accumulate.
        dq_local = torch.zeros_like(q_local, dtype=torch.float32)
        dk_cur = torch.zeros_like(k_local, dtype=torch.float32)
        dv_cur = torch.zeros_like(v_local, dtype=torch.float32)

        # We rotate (K, V, dK, dV) so that on every step we hold the K/V of
        # ``owner`` and the partially-accumulated dK/dV destined for that
        # same rank. After ws-1 rotations the bundle has travelled all the
        # way around the ring and dK/dV are once again with their owner.
        k_cur = k_local.contiguous()
        v_cur = v_local.contiguous()
        owner = rank
        Sq_local = q_local.shape[1]
        q_tile = min(_RING_Q_TILE, Sq_local)

        for step in range(ws):
            k_lo, k_hi = k_offsets[owner], k_offsets[owner + 1]

            # Q-tiled within each ring step (see forward) — peak per-tile
            # tensor is ``(B, H, q_tile, Sk_block)``, mirroring forward's
            # memory budget.
            for q_start in range(0, Sq_local, q_tile):
                q_end = min(q_start + q_tile, Sq_local)
                q_tile_local = q_local[:, q_start:q_end].contiguous()
                mask_tile = attn_mask[:, my_q_lo + q_start : my_q_lo + q_end, k_lo:k_hi]

                # Recompute S and P for this tile (blockwise
                # rematerialisation: nothing about S survives the forward →
                # backward boundary).
                S, v32, k32 = _compute_block(
                    q_tile_local, k_cur, v_cur, mask_tile, scaling, num_query_heads, num_kv_heads
                )
                LSE_tile = LSE[:, :, q_start:q_end]
                P = torch.exp(S - LSE_tile.unsqueeze(-1))

                grad_O_tile = grad_O32[:, q_start:q_end]
                D_tile = D_bhq[:, :, q_start:q_end]

                # dV_block += P^T @ grad_O.
                dv_tile_expanded = torch.einsum("bhqk,bqhd->bkhd", P, grad_O_tile)
                dv_tile = _reduce_kv_grad(dv_tile_expanded, num_kv_heads, num_query_heads)

                # dP, dS.
                dP = torch.einsum("bqhd,bkhd->bhqk", grad_O_tile, v32)
                dS = P * (dP - D_tile.unsqueeze(-1))

                dq_local[:, q_start:q_end] = (
                    dq_local[:, q_start:q_end] + torch.einsum("bhqk,bkhd->bqhd", dS, k32) * scaling
                )
                dk_tile_expanded = (
                    torch.einsum("bhqk,bqhd->bkhd", dS, q_tile_local.to(torch.float32)) * scaling
                )
                dk_tile = _reduce_kv_grad(dk_tile_expanded, num_kv_heads, num_query_heads)

                dk_cur = dk_cur + dk_tile
                dv_cur = dv_cur + dv_tile

            if step < ws - 1:
                # Rotate (K, V, dK, dV) one step forward (same direction as
                # the forward pass) so that on the next iteration we're
                # holding the next rank's K/V and that same rank's
                # accumulated dK/dV.
                k_cur, v_cur, dk_cur, dv_cur = _ring_rotate(
                    k_cur, v_cur, dk_cur, dv_cur, group=group, forward_dir=True
                )
                owner = (owner - 1) % ws

        # After ws-1 rotations: dk_cur, dv_cur belong to ``owner`` which is
        # now (rank - (ws-1)) mod ws == (rank + 1) mod ws. One more rotation
        # in the forward direction puts them back home.
        if ws > 1:
            dk_cur, dv_cur = _ring_rotate(dk_cur, dv_cur, group=group, forward_dir=True)

        return (
            dq_local.to(q_local.dtype),
            dk_cur.to(k_local.dtype),
            dv_cur.to(v_local.dtype),
            None,  # attn_mask
            None,  # scaling
            None,  # num_query_heads
            None,  # num_kv_heads
            None,  # group
            None,  # q_lengths_per_rank
            None,  # k_lengths_per_rank
        )


# ---------------------------------------------------------------------------
# Public entry point — drop-in replacement for the eager/sdpa interfaces.
# ---------------------------------------------------------------------------


def _sdpa_fallback(
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    scaling: float | None,
) -> torch.Tensor:
    """Single-device fallback for ``ring_attention_forward``.

    Numerically identical to ``Gemma3WithExpertModel.sdpa_attention_forward``
    (same GQA expansion, same scale handling, same mask broadcasting), so a
    single-rank smoke run produces bit-equivalent output to the existing
    SDPA path.
    """
    k = _expand_kv(key_states, num_kv_heads, num_query_heads)
    v = _expand_kv(value_states, num_kv_heads, num_query_heads)

    q = query_states.transpose(1, 2)  # (B, Hq, Sq, Dh)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_mask = attention_mask[:, None, :, :]

    sdpa_kwargs = {"attn_mask": attn_mask, "dropout_p": 0.0, "is_causal": False}
    if scaling is not None:
        sdpa_kwargs["scale"] = scaling

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    out = out.permute(0, 2, 1, 3)
    return out.reshape(batch_size, -1, num_query_heads * head_dim)


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
    """Drop-in attention interface backed by ring attention.

    Matches the call signature of ``eager_attention_forward`` /
    ``sdpa_attention_forward`` in ``Gemma3WithExpertModel`` so it can be
    plugged into ``InterleavedDecoderLayer`` via ``get_attention_interface``
    without touching the per-layer body.

    Args:
        attention_mask: Boolean ``(B, Q_total, K_total)`` mask, replicated
            on every rank. The replicated form keeps the layer interface
            unchanged and the mask is small relative to activations.
        batch_size: Batch size (matches the existing interface).
        head_dim: Per-head dimension.
        query_states / key_states / value_states: This rank's local shards
            with shapes ``(B, Sq_local, Hq, Dh)`` / ``(B, Sk_local, Hkv, Dh)``.
        num_query_heads / num_kv_heads: GQA head counts.
        scaling: Pre-softmax scale factor; ``head_dim ** -0.5`` when None
            (matches the existing eager / SDPA interfaces).
        q_lengths_per_rank / k_lengths_per_rank: Per-rank seq lengths.
            Defaults to even chunks of the full mask. Passing them explicitly
            handles the (rare) un-padded case.
        group: Ring process group. Defaults to :func:`get_ring_group`.

    Returns:
        ``(B, Sq_local, Hq * head_dim)`` — this rank's slice of the attention
        output, ready to feed into ``o_proj`` exactly like the existing
        attention interfaces. The surrounding model is responsible for
        gathering the per-rank outputs once at the end of the stack (see
        :func:`gather_seq`).
    """
    group = group if group is not None else get_ring_group()
    ws = ring_world_size(group)
    if scaling is None:
        scaling = head_dim**-0.5

    if ws == 1 or group is None:
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

    # Build per-rank length tables if not supplied. We trust shard_along_seq
    # to have padded total Q / total K to a multiple of world size, which is
    # the only configuration the surrounding pi07 forward sets up.
    if q_lengths_per_rank is None:
        Sq_local = query_states.shape[1]
        q_lengths_per_rank = (Sq_local,) * ws
    if k_lengths_per_rank is None:
        Sk_local = key_states.shape[1]
        k_lengths_per_rank = (Sk_local,) * ws

    out_local = _RingAttention.apply(
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
        num_query_heads,
        num_kv_heads,
        group,
        tuple(q_lengths_per_rank),
        tuple(k_lengths_per_rank),
    )
    # (B, Sq_local, Hq, Dh) -> (B, Sq_local, Hq * Dh)
    return out_local.reshape(batch_size, -1, num_query_heads * head_dim)
