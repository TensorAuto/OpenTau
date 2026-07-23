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

"""State / action column selection and delta-action mapping.

Two index spaces exist and confusing them is the easiest way to mistrain a model, so they are
named explicitly everywhere in this module:

**parquet space**
    Column indices of the dataset's raw on-disk ``state`` / ``action`` vectors. This is the space
    ``DatasetConfig.state_index``, ``DatasetConfig.action_index`` and
    ``DatasetConfig.delta_action_state_map`` are written in.

**post-index space**
    Positions within the vectors the dataset actually emits, i.e. after ``state_index`` /
    ``action_index`` have subset and permuted the raw columns. This is what the *policy* sees —
    by the time a batch reaches it the raw columns are long gone — so the policy-side
    ``delta_action_state_map`` (used to invert the delta at inference) lives in this space.

:func:`resolve_delta_map` is the single translation between them; both the dataset (which applies
the delta) and the mixture metadata (which publishes the policy-side map) go through it, so the
two cannot drift.
"""

import logging

import torch
from torch import Tensor

__all__ = [
    "add_chunk_start_state",
    "apply_column_index",
    "resolve_delta_map",
    "subtract_chunk_start_state",
]


def apply_column_index(value: Tensor, index: list[int] | None, *, what: str, who: str) -> Tensor:
    """Select and reorder the last axis of ``value`` by ``index`` (parquet space).

    Args:
        value: Tensor whose **last** axis is the feature axis (``(D,)``, ``(T, D)``,
            ``(chunk, D)``, ...).
        index: Parquet column indices to keep, in emission order. ``None`` returns ``value``
            unchanged (the common path — no copy, no cost).
        what: Feature name for error messages, e.g. ``"state"``.
        who: Dataset identity for error messages.

    Returns:
        ``value`` restricted to ``index`` along the last axis, in the given order.

    Raises:
        IndexError: If any index is out of range for ``value``'s last axis. Raised eagerly with
            the offending indices named, because a silent out-of-range would otherwise surface
            much later as a shape mismatch inside the policy.
    """
    if index is None:
        return value
    dim = value.shape[-1]
    out_of_range = [i for i in index if i >= dim]
    if out_of_range:
        raise IndexError(
            f"{who}: {what}_index refers to column(s) {out_of_range} but this dataset's raw "
            f"'{what}' has only {dim} column(s) (valid range 0..{dim - 1})."
        )
    return value.index_select(-1, torch.as_tensor(index, dtype=torch.long, device=value.device))


def resolve_delta_map(
    parquet_map: dict[int, int],
    action_index: list[int] | None,
    state_index: list[int] | None,
    *,
    who: str,
    action_dim: int | None = None,
    warn_unmapped: bool = True,
) -> dict[int, int]:
    """Translate a parquet-space delta map into post-index space.

    Args:
        parquet_map: ``{action_dim: state_dim}`` in **parquet space**.
        action_index: The dataset's ``action_index`` (parquet space), or ``None`` for identity.
        state_index: The dataset's ``state_index`` (parquet space), or ``None`` for identity.
        who: Dataset identity for messages.
        action_dim: Number of action columns the dataset emits *after* indexing. Used only to
            warn about kept-but-unmapped dims; pass ``None`` to skip that check when the width
            isn't known yet.
        warn_unmapped: Whether to warn when some kept action dims have no mapping.

    Returns:
        ``{action_pos: state_pos}`` in **post-index space**, ready to index the emitted vectors.

    Raises:
        ValueError: If a mapped action or state column was dropped by the corresponding index —
            silently ignoring it would train that dim absolute while the config says otherwise.
    """
    # parquet column -> emitted position. Identity when no index is configured.
    action_pos = {c: p for p, c in enumerate(action_index)} if action_index is not None else None
    state_pos = {c: p for p, c in enumerate(state_index)} if state_index is not None else None

    resolved: dict[int, int] = {}
    dropped_actions: list[int] = []
    dropped_states: list[tuple[int, int]] = []
    for a_col, s_col in sorted(parquet_map.items()):
        if action_pos is not None and a_col not in action_pos:
            dropped_actions.append(a_col)
            continue
        if state_pos is not None and s_col not in state_pos:
            dropped_states.append((a_col, s_col))
            continue
        resolved[action_pos[a_col] if action_pos is not None else a_col] = (
            state_pos[s_col] if state_pos is not None else s_col
        )

    if dropped_actions:
        raise ValueError(
            f"{who}: delta_action_state_map maps action column(s) {dropped_actions}, but "
            f"action_index={action_index} does not keep them. Either add them to action_index "
            "or remove them from the delta map."
        )
    if dropped_states:
        raise ValueError(
            f"{who}: delta_action_state_map references state column(s) "
            f"{[s for _, s in dropped_states]} (for action column(s) "
            f"{[a for a, _ in dropped_states]}), but state_index={state_index} does not keep "
            "them. A delta cannot be formed against a state column the dataset drops."
        )

    if warn_unmapped and action_dim is not None:
        unmapped = [p for p in range(action_dim) if p not in resolved]
        if unmapped:
            logging.warning(
                "%s: use_delta_joint_actions is on, but action dim(s) %s have no entry in "
                "delta_action_state_map, so they will train as ABSOLUTE actions (post-index "
                "positions; %d of %d dims are relative). This is intended for e.g. an absolute "
                "gripper — if any of these should be relative, add them to the map.",
                who,
                unmapped,
                len(resolved),
                action_dim,
            )
    return resolved


def _offset_actions(actions: Tensor, state: Tensor, delta_map: dict[int, int], sign: float) -> Tensor:
    """Add or subtract the chunk-start state on ``delta_map``'s action dims.

    Shared by :func:`subtract_chunk_start_state` and :func:`add_chunk_start_state` so the forward
    and inverse transforms cannot drift apart in shape handling — a mismatch there would show up
    only as subtly wrong actions at inference.

    Args:
        actions: ``(chunk, D_a)`` or ``(B, chunk, D_a)``.
        state: ``(D_s,)`` / ``(T, D_s)`` unbatched, or ``(B, D_s)`` / ``(B, T, D_s)`` batched.
            A history axis is resolved to its **last** (current) step, which is what the chunk is
            anchored to.
        delta_map: ``{action_pos: state_pos}`` in **post-index space**.
        sign: ``-1.0`` to make actions relative, ``+1.0`` to make them absolute.

    Returns:
        A new tensor; ``actions`` is not mutated.
    """
    if not delta_map:
        return actions
    # A history axis is present exactly when state carries one more axis than the "one vector per
    # chunk" baseline, i.e. when its rank matches the action tensor's.
    current = state[..., -1, :] if state.ndim == actions.ndim else state
    a_pos = torch.as_tensor(sorted(delta_map), dtype=torch.long, device=actions.device)
    s_pos = torch.as_tensor([delta_map[int(a)] for a in a_pos], dtype=torch.long, device=actions.device)
    out = actions.clone()
    # Insert the chunk axis so ONE state broadcasts across the whole horizon — the defining
    # property of this transform, mirroring openpi's `np.expand_dims(..., axis=-2)`.
    offset = current[..., s_pos].unsqueeze(-2).to(out.dtype)
    out[..., a_pos] = out[..., a_pos] + sign * offset
    return out


def subtract_chunk_start_state(actions: Tensor, state: Tensor, delta_map: dict[int, int]) -> Tensor:
    """Convert absolute actions to deltas against a single chunk-start state.

    Mirrors openpi's ``DeltaActions``: the *same* state — the one observed at the start of the
    chunk — is subtracted from every action in the horizon, so element ``k`` carries the
    cumulative displacement from the chunk start rather than a per-step increment. Only
    ``delta_map``'s action dims are touched; the rest keep absolute values.

    Args:
        actions: ``(chunk, D_a)`` or ``(B, chunk, D_a)`` absolute actions.
        state: The conditioning state; see :func:`_offset_actions` for accepted shapes.
        delta_map: ``{action_pos: state_pos}`` in **post-index space**.

    Returns:
        A new tensor; ``actions`` is not mutated.
    """
    return _offset_actions(actions, state, delta_map, -1.0)


def add_chunk_start_state(actions: Tensor, state: Tensor, delta_map: dict[int, int]) -> Tensor:
    """Inverse of :func:`subtract_chunk_start_state` (openpi's ``AbsoluteActions``).

    Args:
        actions: ``(chunk, D_a)`` or ``(B, chunk, D_a)`` delta actions as emitted by the policy.
        state: The state the chunk was conditioned on; see :func:`_offset_actions`.
        delta_map: ``{action_pos: state_pos}`` in **post-index space**.

    Returns:
        A new tensor with absolute actions on the mapped dims.
    """
    return _offset_actions(actions, state, delta_map, 1.0)
