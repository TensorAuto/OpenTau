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

"""Best-of-N action-chunk sampling: fan out over noise, score, keep one chunk.

A flow-matching policy maps one Gaussian draw to exactly one action chunk, deterministically.
Sampling ``n_candidates`` draws therefore yields ``n_candidates`` distinct chunks, and an
:class:`ActionChunkCritic` picks the one that reaches the robot.

**Why this is nearly free.** The VLM prefix pass runs once per *observation*, and the KV cache
it fills is never written again during denoising (``fill_kv_cache=False`` in every
``denoise_step``). Expanding that cache across candidates lets one prefix pass serve all N,
so the added cost is only the action expert's Euler loop at a wider batch — and at batch 1
that loop is occupancy-bound, not FLOP-bound, so widening it rides largely free. Measured on
an RTX 5090 (bf16, sdpa, 3 cameras, 1024 prefix tokens, chunk 50, 10 steps): N=4 costs 1%
over N=1, N=8 costs 3%, and the knee is at N=16 (39%). Naive replication of the observation
across the batch — one prefix pass per candidate — costs 1.66x at N=4 and 2.51x at N=8 for
the same result, and runs out of memory at N=32 where the fused path does not.

**Off by default, and it stays off unless an entry point arms it.** ``n_candidates`` is
``1`` on every freshly constructed policy wrapper regardless of what the config says;
:func:`configure_candidates` is the only writer. A config that travels with a checkpoint
therefore cannot self-arm best-of-N in a script that never opted in — the same posture as
``accel_prefix`` (:func:`opentau.policies.accel.configure_accel`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, runtime_checkable

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn

from opentau.policies.accel import action_dim_scale, resolve_action_dim_mask

#: Selector name that resolves to :class:`MedoidCritic` rather than a checkpoint path.
MEDOID = "medoid"


@runtime_checkable
class ActionChunkCritic(Protocol):
    """Scores candidate action chunks so the policy can keep the best one.

    Implementations are independently trained models with their own normalization; both
    inputs therefore cross this seam in **raw** (un-normalized) units. Coupling a critic to
    the *policy's* norm head would mean that swapping policy checkpoints silently shifts the
    critic's input distribution.
    """

    def score_chunks(
        self,
        batch: dict[str, Tensor],
        candidates: Tensor,
        *,
        row_mask: Tensor,
        dataset_index: Tensor,
    ) -> Tensor:
        """Return one score per candidate; higher is better.

        Args:
            batch: The raw observation batch, ``B`` rows — deliberately *not* expanded to
                ``B * N``, so a VLM critic can embed the observation once and cross-attend
                all N chunks rather than re-paying the cost this feature exists to remove.
            candidates: ``(B, N, chunk, action_dim)`` in raw units.
            row_mask: ``(B, chunk)`` or ``(1, chunk)`` bool. ``True`` marks a row that will
                actually be executed — not frozen by real-time chunking, and inside the
                ``n_action_steps`` window. Rows outside it never reach the robot and
                scoring them dilutes the signal.
            dataset_index: ``(B,)`` long, the norm-head row index already resolved by the
                policy.

        Returns:
            ``(B, N)`` float32.
        """
        ...


def expand_candidates(t: Tensor, n: int) -> Tensor:
    """Interleave-repeat along batch: source row ``i`` becomes rows ``[i*n, (i+1)*n)``.

    The ``.contiguous()`` is required, not defensive tidying. ``einops.repeat`` returns a
    **stride-0 alias sharing storage** whenever the source batch is 1 — measured on a real
    pi05 cache slice, a ``(1, 1024, 1, 256)`` source expands to ``stride[0] == 0`` with an
    unchanged 512 KiB storage — and ``B == 1`` is the entire serving path (``select_action``
    and the gRPC server both produce exactly that shape). Materializing buys three things:
    the "never aliases the caller" guarantee this module documents; a stable stride signature
    across ``B``, so a varying-batch server does not force a Dynamo recompile per shape under
    ``mode="max-autotune"``; and safety against the ``StaticCache`` optimization anticipated
    by the TODO in ``paligemma_with_expert.py``, which would write candidate ``i``'s K/V
    through stride 0 into all N rows with correct shapes and no error.

    Today the aliased and materialized forms produce bit-identical outputs, because nothing
    writes to the cache. That is a property of the current implementation, not a guarantee.
    """
    return repeat(t, "b ... -> (b n) ...", n=n).contiguous()


def expand_kv_cache(past_key_values: dict[int, dict[str, Tensor]], n: int) -> dict[int, dict[str, Tensor]]:
    """Rebuild the prefix KV cache with each entry repeated across candidates.

    Rebuilds rather than mutating: the caller's cache may be reused (and, under
    ``predict_response``, was just written by the autoregressive loop).

    Note the ``.items()``. The cache is a **dict keyed by layer index**, not the
    ``list[dict[str, Tensor]]`` its type annotations long claimed; a list comprehension over
    it silently yields a list of integer keys.
    """
    return {
        layer_idx: {name: expand_candidates(t, n) for name, t in entry.items()}
        for layer_idx, entry in past_key_values.items()
    }


def collapse_candidates(t: Tensor, n: int) -> Tensor:
    """Inverse of :func:`expand_candidates`: ``(B*N, ...) -> (B, N, ...)``.

    Paired with ``(b n)`` ordering. A transposed convention is invisible in shapes and shows
    up only as candidates attributed to the wrong observation once ``B > 1``.
    """
    return rearrange(t, "(b n) ... -> b n ...", n=n)


def select_candidate(scores: Tensor) -> Tensor:
    """Argmax over finite scores, with degenerate rows falling back to candidate 0.

    Deliberately **total** — it never raises. ``select_action`` runs on every rank and is
    followed by gather collectives, so a per-rank data-dependent raise aborts one rank while
    the others block forever at NCCL. A row whose scores are all non-finite gets candidate 0
    instead, and the caller warns.

    Two behaviours worth stating because a reader would not assume them:

    * **NaN must not win.** ``torch.argmax`` returns the NaN's index on a raw tensor
      (measured: ``argmax([1.0, nan, 3.0]) -> 1``), so non-finite entries are replaced with
      ``-inf`` before the reduction.
    * **Ties resolve to the lowest index**, which is load-bearing: candidate 0 carries the
      noise draw the ``n_candidates == 1`` path would have taken, so a degenerate critic
      degenerates to legacy behaviour rather than to something arbitrary. No RNG is involved
      anywhere — ``set_seed`` offsets the global stream by ``process_index``, so a random
      tie-break would desync ranks.

    Args:
        scores: ``(B, N)``, higher is better.

    Returns:
        ``(B,)`` long, the chosen candidate index per batch row.
    """
    finite = torch.isfinite(scores)
    masked = torch.where(finite, scores.float(), torch.full_like(scores, float("-inf"), dtype=torch.float32))
    all_bad = ~finite.any(dim=1)
    safe = torch.where(rearrange(all_bad, "b -> b 1"), torch.zeros_like(masked), masked)
    picked = torch.argmax(safe, dim=1)
    return torch.where(all_bad, torch.zeros_like(picked), picked)


#: Set once a degenerate-score warning has been emitted, so a serving loop reports the
#: condition without printing a line per request.
_WARNED_DEGENERATE = False


def warn_if_no_candidate_scored(score_rows: list[list[float]]) -> None:
    """Log once when a batch row's scores were all non-finite.

    :func:`select_candidate` falls back to candidate 0 there rather than raising, because a
    data-dependent raise inside ``select_action`` would abort one rank while its peers block
    at the next collective. That fallback must still be *observable* — otherwise a critic
    returning all-NaN looks exactly like a critic that works and happens to prefer candidate
    0 every time.

    Takes the already-synced host-side scores rather than the device tensor, so it costs no
    extra device-to-host round trip beyond the one the caller already paid.

    Args:
        score_rows: ``(B, N)`` scores as nested Python lists.
    """
    global _WARNED_DEGENERATE
    if _WARNED_DEGENERATE:
        return
    bad = [i for i, row in enumerate(score_rows) if not any(math.isfinite(v) for v in row)]
    if not bad:
        return
    _WARNED_DEGENERATE = True
    logging.warning(
        "action-chunk critic returned no finite score for batch row(s) %s; those rows fell "
        "back to candidate 0. Best-of-N is not selecting on anything for them. This is "
        "logged once per process.",
        bad,
    )


class MedoidCritic(nn.Module):
    """Reference critic: keep the candidate closest to all the others.

    A learning-free stand-in that makes best-of-N testable and benchmarkable end to end
    before a trained critic exists. The intuition is consensus — with N draws from the same
    conditional, the chunk minimising total distance to its peers sits in the densest region
    of the sampled posterior, and outlying draws (the ones a flow model occasionally emits)
    lose. It is **not** a quality model: it cannot tell a confidently-wrong mode from a
    correct one, and on a genuinely multimodal task it will prefer the more populated mode
    rather than the better one.

    It is opt-in by name (``action_chunk_critic_path: "medoid"``) and never a default. An
    implicit fallback here would be worse than a hard failure: best-of-N would appear to work
    while selecting on no real signal.

    **The one critic that borrows the policy's norm head.** Raw action dims live on wildly
    different scales — a gripper in ``[0, 1]`` against a shoulder joint in radians — so an
    unweighted distance is dominated by whichever dim happens to have the largest units.
    Having no buffers of its own, this critic divides by the policy's per-dim scale and drops
    the policy's degenerate dims. That coupling is deliberate and specific to a parameter-free
    selector; a *trained* critic must carry its own normalization (see
    :class:`ActionChunkCritic`).
    """

    def __init__(self, unnormalize: nn.Module, *, max_action_dim: int) -> None:
        super().__init__()
        # Held by reference, not registered: this critic owns no parameters, and registering
        # the policy's own Unnormalize as a child would duplicate it into the critic's
        # state_dict.
        object.__setattr__(self, "_unnormalize", unnormalize)
        self._max_action_dim = max_action_dim

    def score_chunks(
        self,
        batch: dict[str, Tensor],
        candidates: Tensor,
        *,
        row_mask: Tensor,
        dataset_index: Tensor,
    ) -> Tensor:
        """Negative mean distance to the other candidates, per :class:`ActionChunkCritic`."""
        del batch  # consensus is computed over the candidates alone
        bsize, n_cand, _, action_dim = candidates.shape
        if n_cand < 3:
            # At N == 2 every pairwise distance is the same number, so every candidate ties
            # and selection collapses to candidate 0 — best-of-N that silently is not.
            raise ValueError(
                f"{type(self).__name__} needs n_candidates >= 3 to rank anything (got {n_cand}): "
                "with two candidates the single pairwise distance is shared, every score ties, "
                "and selection always falls back to candidate 0."
            )

        scale = action_dim_scale(
            self._unnormalize, max_action_dim=self._max_action_dim, dataset_index=dataset_index
        )
        keep = resolve_action_dim_mask(
            self._unnormalize, max_action_dim=self._max_action_dim, dataset_index=dataset_index
        )
        # sample_actions has already truncated to the real action width, so the scale/mask
        # vectors (built at max_action_dim) must be cut to match.
        scale = scale[:, :action_dim]
        keep = keep[:, :action_dim]

        normed = candidates.float() / rearrange(scale, "b d -> b 1 1 d")
        # Name the batch axis explicitly rather than letting an ellipsis pattern right-align
        # it. A ``(B, chunk)`` row_mask -- which this protocol documents, and which
        # ``executed_row_mask`` produces for a per-sample ``delay`` -- becomes
        # ``(B, chunk, 1)`` under ``"... c -> ... c 1"``, whose leading B then lands on the
        # *candidate* axis of keep's ``(B, 1, 1, D)``: a shape error when B != N, and
        # silently scoring candidate j against observation j's mask when B == N.
        row_mask = row_mask.to(torch.bool)
        if row_mask.ndim != 2:
            raise ValueError(f"row_mask must be (B, chunk) or (1, chunk); got {tuple(row_mask.shape)}.")
        valid = rearrange(keep, "b d -> b 1 1 d") & rearrange(row_mask, "b c -> b 1 c 1")
        normed = torch.where(valid, normed, torch.zeros_like(normed))

        # (B, N, N) pairwise L2 over the flattened executed rows.
        flat = rearrange(normed, "b n c d -> b n (c d)")
        dist = torch.cdist(flat, flat)
        # Exclude the zero self-distance so the mean is over genuine peers.
        total = reduce(dist, "b n m -> b n", "sum") / max(n_cand - 1, 1)
        scores = -total
        if scores.shape != (bsize, n_cand):  # pragma: no cover - shape contract
            raise RuntimeError(f"expected ({bsize}, {n_cand}) scores, got {tuple(scores.shape)}")
        return scores


def attach_critic(policy: nn.Module, critic: Any) -> None:
    """Bind ``critic`` to ``policy`` without it becoming part of the policy.

    ``object.__setattr__`` is mandatory rather than stylistic. Plain assignment of an
    ``nn.Module`` routes through ``nn.Module.__setattr__``, which registers it in
    ``_modules`` — putting the critic's tensors into ``policy.state_dict()`` (which gets
    written to checkpoints) and its parameters into ``policy.parameters()`` (which
    ``get_optim_params`` returns to the optimizer). Both were measured.

    The cost of staying out of ``_modules`` is that ``policy.to(...)`` and
    ``accelerator.prepare`` do not move the critic, which is why
    :func:`configure_candidates` takes an explicit ``device`` and places it itself.
    """
    object.__setattr__(policy, "_action_chunk_critic", critic)


def load_critic(path: str | None, *, policy: nn.Module) -> Any:
    """Resolve ``action_chunk_critic_path`` to a critic instance.

    Only the built-in :data:`MEDOID` selector exists today. A checkpoint path raises rather
    than silently falling back, so a config naming a critic that this build cannot load fails
    at startup instead of selecting on nothing.
    """
    if path == MEDOID:
        return MedoidCritic(policy.unnormalize_outputs, max_action_dim=policy.config.max_action_dim)
    raise NotImplementedError(
        f"action_chunk_critic_path={path!r} names a learned critic, which this build cannot "
        f"load; only the built-in {MEDOID!r} selector is available. Pass "
        f"action_chunk_critic_path={MEDOID!r} to use consensus selection."
    )


def refuse_candidates(cfg: Any, *, reason: str) -> None:
    """Raise when a config asks for best-of-N in an entry point that cannot honour it.

    Every script that builds a policy from ``cfg.policy`` either arms candidates or calls
    this. Silently ignoring the field is the failure mode worth spending a line to prevent:
    an operator sets ``n_candidates=4``, sees no error, and believes best-of-N is running.
    """
    requested = getattr(getattr(cfg, "policy", cfg), "n_candidates", 1) or 1
    if int(requested) > 1:
        raise ValueError(f"n_candidates={requested} is not supported here: {reason}")


def configure_candidates(
    policy: nn.Module,
    cfg: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    critic: Any = None,
    override: int | None = None,
) -> int:
    """Arm best-of-N sampling on ``policy``, if anything asked for it.

    The single enable knob shared by every serving entry point, so the precedence order and
    the refusals cannot drift between them. Call it *before* an entry point's warmup
    ``sample_actions`` calls: critic loading, the dtype cast, the shape smoke-test and any
    out-of-memory then land at startup rather than on the first robot request.

    Resolution order, first non-``None`` wins:

    1. ``override`` — an entry point's own flag.
    2. ``cfg.policy.n_candidates`` — read through ``getattr`` so a config predating the field
       still resolves.

    Precedence uses ``is None`` at every step, never ``or``-truthiness: ``override=1`` is a
    caller deliberately *disabling* a config-enabled N, and an ``or`` chain would fall through
    it. Whenever the policy is wired for candidates the resolved value is written to
    ``policy.n_candidates``, including the N==1 case — returning early without writing would
    leave a stale value from a previous call. An *unwired* policy is never written to, at any
    N: it is a no-op at 1 and a refusal above it.

    Args:
        policy: The loaded policy wrapper.
        cfg: The parsed pipeline config; only ``cfg.policy`` is read. Typed loosely to keep
            this module off the config package's import graph.
        device: Where to place the critic. Explicit because :func:`attach_critic` keeps it
            out of ``policy._modules``, so no later ``.to(...)`` will move it; under
            multi-rank eval each rank needs its own local device.
        dtype: Cast applied to the critic, routed through the SigLIP-preserving helper.
        critic: A pre-built critic, bypassing ``action_chunk_critic_path``.
        override: Entry-point-level request that beats the config.

    Returns:
        The resolved candidate count.

    Raises:
        ValueError: When best-of-N is requested but this policy family is not wired for it,
            or the count is invalid.
        TypeError: When the resolved critic does not satisfy :class:`ActionChunkCritic`.
    """
    requested: int | None = override
    if requested is None:
        requested = getattr(getattr(cfg, "policy", cfg), "n_candidates", None)
    n = 1 if requested is None else int(requested)
    if n < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n}.")

    if n == 1:
        # Return BEFORE the wiring probe below. `PreTrainedConfig.n_candidates` defaults to
        # `1` — an int, not `None` — so this is the path every unarmed serving process takes,
        # for every policy family. Probing here would abort startup of the gRPC / RoboCasa /
        # inference / benchmark entry points for pi0, pi05_mem, both pi07 families, cosmos3
        # and value, none of which asked for anything.
        #
        # Only write when the attribute already exists: creating it on an unwired family
        # would make a later `n > 1` call believe that family was wired.
        if hasattr(policy, "n_candidates"):
            policy.n_candidates = 1
        return 1

    policy_type = getattr(getattr(cfg, "policy", cfg), "type", type(policy).__name__)
    # A policy carrying recurrent rollout state cannot fan out candidates: the
    # chunks the losing candidates produced are never executed, so adopting
    # their state is wrong, and adopting the winner's makes the state depend on
    # a critic the training run never saw. Such a family may *inherit* a wired
    # `__init__` (pi05_ttt inherits `PI05Policy`'s, attribute included), so the
    # `hasattr` probe below cannot see it — check the opt-out explicitly, and
    # here rather than in the sampler, so the failure lands at startup like
    # every other configure-time failure this function exists to surface.
    if not getattr(policy, "supports_candidate_sampling", True):
        raise ValueError(
            f"n_candidates={n} was requested but policy type {policy_type!r} carries recurrent "
            "rollout state (test-time-training fast weights), so best-of-N is undefined for it: "
            "there is no correct answer for which candidate's state update the rollout should "
            "adopt. Use n_candidates=1."
        )
    # A family whose sampler has not been wired would accept the attribute and then never
    # read it — the silent no-op an operator would read as "the critic never fires". Refuse.
    if not hasattr(policy, "n_candidates"):
        raise ValueError(
            f"n_candidates={n} was requested but policy type {policy_type!r} does not expose "
            "`n_candidates`, i.e. its sampler is not wired for best-of-N candidate sampling."
        )

    resolved = critic
    if resolved is None:
        path = getattr(getattr(cfg, "policy", cfg), "action_chunk_critic_path", None)
        if path is None:
            raise ValueError(
                f"n_candidates={n} needs a critic to choose between the candidates, but "
                "action_chunk_critic_path is unset. Set it to "
                f"{MEDOID!r} for the built-in consensus selector."
            )
        resolved = load_critic(path, policy=policy)

    if not isinstance(resolved, ActionChunkCritic):
        raise TypeError(
            f"{type(resolved).__name__} does not satisfy the ActionChunkCritic protocol "
            "(missing `score_chunks`)."
        )

    if isinstance(resolved, nn.Module):
        if dtype is not None:
            # Routed rather than a blanket `.to(dtype)`: a critic with a SigLIP tower has
            # float32-pinned patch/position embeddings that a blanket cast would re-round.
            from opentau.policies.utils import to_dtype_preserving_siglip_float32

            to_dtype_preserving_siglip_float32(resolved, dtype=dtype, device=device)
        else:
            resolved.to(device)
        resolved.eval()

    _smoke_critic(resolved, n, policy, device)
    attach_critic(policy, resolved)
    policy.n_candidates = n
    logging.info("best-of-%d candidate sampling armed with %s", n, type(resolved).__name__)
    return n


def _smoke_critic(critic: Any, n: int, policy: nn.Module, device: torch.device | str) -> None:
    """Call the critic once on dummy inputs so a wrong shape fails at startup.

    ``runtime_checkable`` ``isinstance`` checks method *presence* only — never arity, never
    return shape — so without this a critic returning ``(B,)`` instead of ``(B, N)`` gets
    through configuration and fails on the first robot request.
    """
    config = policy.config
    action_dim = config.action_feature.shape[0]
    probe = torch.zeros(1, n, config.chunk_size, action_dim, device=device)
    # Distinct rows: an all-zeros probe makes every pairwise distance zero, which a consensus
    # critic would reject as degenerate rather than as the shape check this is.
    probe = probe + rearrange(torch.arange(n, device=device, dtype=probe.dtype), "n -> 1 n 1 1")
    row_mask = torch.ones(1, config.chunk_size, dtype=torch.bool, device=device)
    scores = critic.score_chunks(
        {},
        probe,
        row_mask=row_mask,
        dataset_index=torch.zeros(1, dtype=torch.long, device=device),
    )
    if not isinstance(scores, Tensor) or scores.shape != (1, n):
        raise TypeError(
            f"{type(critic).__name__}.score_chunks must return a (B, N) tensor; got "
            f"{tuple(scores.shape) if isinstance(scores, Tensor) else type(scores).__name__} "
            f"for B=1, N={n}."
        )
