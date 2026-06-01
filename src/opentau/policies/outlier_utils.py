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
"""Pure, collective-free detection of normalized state/action outliers.

Shared by the ``pi07`` and ``pi07_paligemma`` low-level policies. Detection runs
inside ``forward`` so it MUST stay rule-5 safe (see CLAUDE.md rule 5): it only
reads the batch and returns records — no logging, no cross-step dedup, and no
collectives — keeping the ``forward`` graph (and the collective counts under
FSDP / ZeRO) identical across ranks regardless of what local data a rank holds.

The cross-rank merge, cross-step dedup, and the actual ``logging.warning`` live
in the training loop (``log_outlier_records_distributed`` in
``opentau/scripts/train.py``), which runs on rank 0 after gathering offenders
from every rank — so the warning always reaches wandb regardless of which rank
held the offending sample.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from einops import rearrange, reduce
from torch import Tensor


class OutlierRecord(TypedDict):
    """One worst ``(source, key, dim)`` offender found in a local batch.

    ``value`` is the largest ``|normalized value|`` seen for this
    ``(source, key, dim)`` in the batch; ``source`` / ``episode`` / ``frame`` are
    the provenance of that worst sample (``None`` when the batch lacks the
    field). Only plain Python scalars are stored so the list survives the
    ``gather_object`` round-trip used to ship records to rank 0.
    """

    key: str
    source: object
    dim: int
    value: float
    episode: object
    frame: object


def _per_sample(value: object, i: int) -> object:
    """Pick the ``i``-th per-sample entry from a batch provenance field.

    Handles the list / tensor / scalar (or missing) forms the standard data
    format may use for ``source`` / ``episode_index`` / ``frame_index``.
    """
    if isinstance(value, list):
        return value[i]
    if torch.is_tensor(value):
        return int(value[i])
    return value


def _attended_steps_mask(key: str, t: Tensor, batch: dict[str, Tensor]) -> Tensor | None:
    """``(B, T)`` bool of timesteps the model attends to for ``key``, or ``None`` to scan all.

    Mirrors :meth:`_build_prefix_items`. For ``state`` the current (last) frame is always
    attended — including when ``obs_history_is_pad`` marks everything padded (the
    ``history_state_drop`` case, where the masked history is zeroed *after* normalization
    downstream) and when the mask is *absent*, where the model attends the last frame only
    (``state_mask = zeros; [:, -1] = True``). ``actions`` follows ``action_is_pad``. Returns
    ``None`` (scan every timestep) for a 2-D tensor, a shape-mismatched mask, or ``actions``
    without ``action_is_pad`` — keeping every existing caller/test unchanged.
    """
    if t.ndim < 3:
        return None
    if key == "state":
        pad = batch.get("obs_history_is_pad")
        if pad is not None and (pad.ndim != 2 or pad.shape[1] != t.shape[1]):
            return None  # shape mismatch -> scan all (back-compat; shouldn't happen in practice)
        if pad is None:
            # No mask: the model attends the current frame only, so the warning must too.
            keep = torch.zeros(t.shape[0], t.shape[1], dtype=torch.bool, device=t.device)
        else:
            keep = ~pad.bool()
        keep[:, -1] = True  # the current frame is always attended
        return keep
    if key == "actions":
        pad = batch.get("action_is_pad")
        if pad is None or pad.ndim != 2 or pad.shape[1] != t.shape[1]:
            return None
        return ~pad.bool()
    return None


def detect_state_action_outliers(batch: dict[str, Tensor], threshold: float | None) -> list[OutlierRecord]:
    """Return the worst ``(source, key, dim)`` outlier records in ``batch``.

    A normalized value far from unit scale almost always means bad normalization
    stats (e.g. near-zero std on a constant dim) or corrupt data. This finds the
    offending dims so the training loop can warn about them, recording the
    ``source`` / ``episode_index`` / ``frame_index`` (when present in the batch)
    to trace a poorly-normalized dim back to the dataset/frame to inspect.

    Pure and collective-free: it reads ``batch`` without mutating it, fires no
    collective, and does NO cross-step dedup or logging — so the ``forward``
    graph, and therefore the collective counts under FSDP / ZeRO, stay identical
    across ranks regardless of what any rank's local data contains (CLAUDE.md
    rule 5). The returned records are merged across ranks, deduped, and logged on
    rank 0 by ``log_outlier_records_distributed`` in the training loop.

    Args:
        batch: Training batch *after* input/target normalization. Reads
            ``state`` ``(B, [T,] D)`` and ``actions`` ``(B, chunk, D)``; the
            zero-padded tail dims never trigger. Timesteps the model does not
            attend to (padded history via ``obs_history_is_pad`` / padded action
            steps via ``action_is_pad``) are excluded so a masked-out frame can't
            trip the check — the current state frame is always kept.
        threshold: Absolute-value ceiling. ``None`` or ``<= 0`` disables the
            check entirely (returns ``[]`` with no device sync).

    Returns:
        One :class:`OutlierRecord` per offending ``(source, key, dim)`` in this
        batch (the worst ``|value|`` per tuple). Empty when disabled or clean.
    """
    if threshold is None or threshold <= 0:
        return []
    per_key: dict[str, tuple[Tensor, Tensor]] = {}
    for key in ("state", "actions"):
        t = batch.get(key)
        if t is None:
            continue
        t = t.detach().abs().float()
        # Ignore timesteps the model never attends to (padded history / padded action steps):
        # zero them so a masked-out frame can't trip the warning. The current state frame is
        # always kept (mirrors `_build_prefix_items`'s `state_mask[:, -1] = True`).
        keep = _attended_steps_mask(key, t, batch)
        if keep is not None:
            t = t * rearrange(keep, "b t -> b t 1").to(t.dtype)
        # (B, [T|chunk,] D) -> (B, D): max |value| per feature dim. The ``b ... d`` pattern
        # absorbs the optional middle axis (2-D state, 3-D state history, 3-D action chunk).
        per_dim_max = reduce(t, "b ... d -> b d", "max")
        per_key[key] = (per_dim_max, per_dim_max > threshold)
    if not per_key:
        return []
    # One device->host sync covering both tensors; skip the rest on the common
    # (no-outlier) path.
    if not bool(torch.cat([viol.flatten() for _, viol in per_key.values()]).any()):
        return []

    src = batch.get("source")
    ep = batch.get("episode_index")
    fr = batch.get("frame_index")

    records: list[OutlierRecord] = []
    for key, (per_dim_max, viol) in per_key.items():
        # (sample, dim) coordinates of every violating entry this step.
        coords = torch.nonzero(viol, as_tuple=False).tolist()
        if not coords:
            continue
        # Index on CPU once so the per-offender lookups below don't each sync.
        pdm = per_dim_max.detach().cpu()
        # Aggregate to the worst |value| per (source, key, dim) in this batch (a batch can hold
        # several samples from the same source/dim); keep the worst sample's provenance.
        worst_per_tup: dict[tuple[object, str, int], tuple[float, int]] = {}
        for s_i, d in coords:
            tup = (_per_sample(src, s_i), key, d)
            val = pdm[s_i, d].item()
            if tup not in worst_per_tup or val > worst_per_tup[tup][0]:
                worst_per_tup[tup] = (val, s_i)
        for (source, _key, dim), (val, s_i) in worst_per_tup.items():
            records.append(
                OutlierRecord(
                    key=key,
                    source=source,
                    dim=dim,
                    value=val,
                    episode=_per_sample(ep, s_i),
                    frame=_per_sample(fr, s_i),
                )
            )
    return records
