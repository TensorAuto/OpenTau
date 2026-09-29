# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Range normalization and the joint-delta parameterization of the SO-101 recipe.

``normalize`` maps a channel's 1st / 99th percentile to -1 / 1 and clips: ``x_norm = clip(2 (x - q01) / span -
1, -clip, clip)`` with ``span = q99 - q01``, or 1 where the bounds coincide (a gripper that never moved).
``denormalize`` inverts it inside the clip range. ``deltas`` turns a window's absolute commands into the
per-frame differences the recipe trains on (the command before the window supplies the first one), and
``integrate`` adds predicted deltas onto the observed state. Both keep ``absolute_dims`` unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEGENERATE_SPAN = 1e-6


def bounds_tensors(stats: dict[str, list[float]], like: Tensor) -> tuple[Tensor, Tensor]:
    """``(q01, span)`` as tensors on ``like``'s device and dtype."""
    q01 = torch.as_tensor(stats["q01"], dtype=like.dtype, device=like.device)
    q99 = torch.as_tensor(stats["q99"], dtype=like.dtype, device=like.device)
    span = q99 - q01
    span = torch.where(span > DEGENERATE_SPAN, span, torch.ones_like(span))
    return q01, span


def normalize(x: Tensor, stats: dict[str, list[float]] | None, clip: float) -> Tensor:
    """``(..., D)`` in dataset units -> normalized; ``stats=None`` is the identity."""
    if stats is None:
        return x
    q01, span = bounds_tensors(stats, x)
    return (2.0 * (x - q01) / span - 1.0).clamp_(-clip, clip)


def denormalize(x: Tensor, stats: dict[str, list[float]] | None) -> Tensor:
    """Inverse of :func:`normalize` (exact inside the clip range)."""
    if stats is None:
        return x
    q01, span = bounds_tensors(stats, x)
    return (x + 1.0) * span / 2.0 + q01


def deltas(actions: Tensor, previous: Tensor, absolute_dims: tuple[int, ...]) -> Tensor:
    """``actions (B, K, D)`` and the command before the window ``previous (B, D)`` -> ``a[t] - a[t-1]`` per
    frame, with ``absolute_dims`` copied from ``actions`` unchanged."""
    if previous.shape != (actions.shape[0], actions.shape[2]):
        raise ValueError(
            f"previous command must be (B, D) = {(actions.shape[0], actions.shape[2])}, got {tuple(previous.shape)}"
        )
    shifted = torch.cat([previous[:, None], actions[:, :-1]], dim=1)
    out = actions - shifted
    if absolute_dims:
        dims = list(absolute_dims)
        out[..., dims] = actions[..., dims]
    return out


def integrate(targets: Tensor, state: Tensor, absolute_dims: tuple[int, ...]) -> Tensor:
    """``targets (K, D)`` deltas and the observed ``state (D,)`` -> absolute commands ``state + cumsum``;
    ``absolute_dims`` are taken from ``targets`` as they are."""
    if state.shape != (targets.shape[-1],):
        raise ValueError(f"state must be ({targets.shape[-1]},), got {tuple(state.shape)}")
    out = state[None] + torch.cumsum(targets, dim=0)
    if absolute_dims:
        dims = list(absolute_dims)
        out[..., dims] = targets[..., dims]
    return out
