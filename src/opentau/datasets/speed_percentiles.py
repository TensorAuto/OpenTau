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
"""Per-task percentile bucketing for the ``speed`` optional key.

For each ``(dataset, task)`` pair we rank the episodes by length-in-frames
and bucket each episode's length into one of ``{0, 10, 20, ..., 100}``
based on where it falls in its task's distribution. Shorter episodes get
lower bucket labels (= "faster"); longer episodes get higher labels.

The 10 percentile boundaries (p5, p15, ..., p95) per task are persisted
to ``meta/speed_percentiles.jsonl`` so the compute happens at most once
per dataset on disk. Existence of the file is the sole gate; staleness
(after ``meta/episodes.jsonl`` is appended to) is detected by comparing
the on-disk per-task ``n_episodes`` total against the current load — a
mismatch logs a WARNING but the file is still trusted (delete it to
force recompute).

Tasks with fewer than ``MIN_EPISODES_FOR_PERCENTILES`` *distinct*
episode lengths are written with ``"percentiles": null`` and bucket
every episode to ``SPARSE_TASK_BUCKET`` (= 50). The distinct-count
threshold catches both small-N and degenerate (all-equal-length)
distributions in one rule.
"""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from opentau.datasets.utils import load_jsonlines, write_jsonlines
from opentau.utils.accelerate_utils import get_proc_accelerator

if TYPE_CHECKING:
    import datasets
    import torch

# Boundary percentiles, ascending. Length 10 by design: 11 buckets.
SPEED_PERCENTILES: tuple[int, ...] = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)
# Bucket labels, ascending. Index i is the label for "between p_{i-1} and
# p_i" (with index 0 = "below p5" and index 10 = "at or above p95").
SPEED_BUCKET_LABELS: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
# Persistence path under the dataset root. JSONL with one entry per task.
SPEED_PERCENTILES_PATH = "meta/speed_percentiles.jsonl"
# A task needs at least this many *distinct* episode lengths for percentile
# ranking to be meaningful (one-per-decile minimum).
MIN_EPISODES_FOR_PERCENTILES = 10
# Bucket label assigned to episodes whose task is too sparse to rank.
# Median-equivalent so the policy receives a neutral signal rather than a
# misleading "fast" or "slow" claim.
SPARSE_TASK_BUCKET = 50

# Module-level set of dataset roots for which we've already warned about a
# read-only meta/ directory. Suppresses log spam when many datasets in a
# mixture share the same read-only HF snapshot. Mirrors the
# ``_CONTROL_MODE_WARNED`` pattern in ``lerobot_dataset.py``.
_READONLY_WARNED: set[str] = set()

# Module-level set of unknown per-frame ``task_index`` integers we've
# already warned about. Keyed by the integer index, so a corrupt parquet
# pointing many episodes at a missing index logs once per distinct index
# per process instead of once per episode per rank.
_UNKNOWN_TASK_INDEX_WARNED: set[int] = set()


def compute_task_percentiles(
    episode_lengths_per_task: dict[int, list[int]],
) -> dict[int, list[float] | None]:
    """Compute p5..p95 of episode lengths for each task.

    Args:
        episode_lengths_per_task: Maps ``task_index`` to the list of
            episode lengths (in frames) for episodes belonging to that
            task.

    Returns:
        Dict mapping ``task_index`` to a length-10 list of floats
        ``[p5, p15, ..., p95]`` (ascending), or ``None`` for tasks with
        fewer than :data:`MIN_EPISODES_FOR_PERCENTILES` distinct lengths.
        The ``None`` sentinel covers both the small-N case (one or two
        episodes) and the degenerate all-equal-length case (a fixed-length
        synthetic dataset). Both produce zero-information rankings, so
        the bucket lookup falls back to :data:`SPARSE_TASK_BUCKET`.
    """
    out: dict[int, list[float] | None] = {}
    for task_idx, lengths in episode_lengths_per_task.items():
        if len(set(lengths)) < MIN_EPISODES_FOR_PERCENTILES:
            out[task_idx] = None
            continue
        pcts = np.percentile(lengths, SPEED_PERCENTILES)
        out[task_idx] = [float(p) for p in pcts]
    return out


def bucket_episode_length(length: int, percentiles: list[float] | None) -> int:
    """Map an episode length to its bucket label.

    Args:
        length: Episode length in frames.
        percentiles: The 10 percentile boundaries for this episode's task,
            or ``None`` when the task is too sparse to rank.

    Returns:
        One of :data:`SPEED_BUCKET_LABELS` (i.e. ``{0, 10, 20, ..., 100}``).
        Sparse tasks (``percentiles is None``) return
        :data:`SPARSE_TASK_BUCKET`.

    Tie semantics: ``length == p_X`` lands in the *upper* bucket, e.g.
    ``length == p25`` → 30. Uses ``np.searchsorted(side='right')`` so the
    bucket boundary intervals read as ``[p_{i-1}, p_i)`` with the lowest
    bucket being ``(-inf, p5)`` and the highest ``[p95, +inf)``.
    """
    if percentiles is None:
        return SPARSE_TASK_BUCKET
    idx = int(np.searchsorted(percentiles, length, side="right"))
    return SPEED_BUCKET_LABELS[idx]


def _group_lengths_by_task(
    episode_lengths: dict[int, int],
    episode_to_task_index: dict[int, int],
) -> dict[int, list[int]]:
    """Group ``episode_lengths`` values by their ``episode_to_task_index`` key."""
    by_task: dict[int, list[int]] = defaultdict(list)
    for ep_idx, task_idx in episode_to_task_index.items():
        by_task[task_idx].append(episode_lengths[ep_idx])
    return dict(by_task)


def episode_to_task_index_from_hf_dataset(
    hf_dataset: datasets.Dataset,
    episodes: list[int],
    episode_data_index: dict[str, torch.Tensor],
    epi2idx: dict[int, int],
    valid_task_indices: set[int],
) -> dict[int, int]:
    """Build ``{ep_idx: task_idx}`` by reading the per-frame parquet.

    Each row of the per-episode parquet carries an authoritative integer
    ``task_index`` column (written by :meth:`LeRobotDataset._save_episode_table`
    and consumed by :meth:`LeRobotDataset.__getitem__`). Resolving the
    per-episode task this way bypasses any drift between
    ``episodes.jsonl::tasks[0]`` (a string) and ``tasks.jsonl::task`` (the
    string keyed by ``task_index``) — a paraphrased ``tasks.jsonl`` no
    longer breaks the lookup.

    Only the *first* row of each episode is read; the parquet's
    ``task_index`` is constant within an episode by construction.

    Episodes whose resolved ``task_index`` is not present in
    ``valid_task_indices`` (i.e. the index points to a task that ``tasks.jsonl``
    doesn't define — genuine metadata corruption, not paraphrasing) are
    skipped with a deduped WARNING. Skipped episodes are still trained on;
    they just fall back to ``SPARSE_TASK_BUCKET`` in the downstream
    speed-bucket lookup, which already tolerates a missing
    ``episode_to_task_index`` entry.

    Args:
        hf_dataset: The dataset's per-frame HF ``Dataset`` (parquet-backed).
        episodes: Selected episode indices (the dataset's ``self.episodes``).
        episode_data_index: ``{"from": Tensor, "to": Tensor}`` giving the
            start/end row in ``hf_dataset`` for each selected episode.
        epi2idx: Maps ``episode_index`` to its position in
            ``episode_data_index["from"]``.
        valid_task_indices: All ``task_index`` values defined in
            ``tasks.jsonl`` (i.e. ``set(meta.task_to_task_index.values())``).
            Used solely to detect parquet rows that point at a task the
            metadata doesn't know about.
    """
    if not episodes:
        return {}
    start_indices = [int(episode_data_index["from"][epi2idx[ep]].item()) for ep in episodes]
    task_indices = hf_dataset.with_format("arrow").select(start_indices)["task_index"].to_pylist()
    out: dict[int, int] = {}
    for ep, task_idx in zip(episodes, task_indices, strict=True):
        task_idx = int(task_idx)
        if task_idx not in valid_task_indices:
            if task_idx not in _UNKNOWN_TASK_INDEX_WARNED:
                _UNKNOWN_TASK_INDEX_WARNED.add(task_idx)
                logging.warning(
                    "Episode parquet rows reference task_index=%d which is not defined "
                    "in tasks.jsonl; episode(s) using it fall back to the sparse speed "
                    "bucket. This indicates corrupt dataset metadata.",
                    task_idx,
                )
            continue
        out[ep] = task_idx
    return out


def _atomic_write_jsonlines(rows: list[dict], path: Path) -> None:
    """Write ``rows`` to ``path`` atomically via tmp file + ``os.replace``.

    Defends against partial-write corruption when two processes (e.g.
    parallel training jobs sharing one dataset, or a stale ``__init__``
    overlapping a fresh one) race on the same file. Within a single
    distributed run we additionally rank-gate the write (see
    :func:`load_or_compute_speed_percentiles`); the atomic rename is the
    fallback for cases where rank-gating doesn't apply.

    Uses a per-writer UUID-suffixed tmp path so two concurrent writers
    don't truncate each other's tmp file before the rename. ``os.replace``
    is atomic on POSIX and on Windows for same-filesystem renames, so
    once the tmp file is fully written the swap into ``path`` is the
    point-of-commit; readers see either the previous file or the new
    one, never a half-written one.
    """
    tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    try:
        write_jsonlines(rows, tmp_path)
        os.replace(tmp_path, path)
    finally:
        # If write_jsonlines raised after creating the file, or os.replace
        # failed, leave no orphan behind.
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def load_or_compute_speed_percentiles(
    root: Path,
    episode_lengths: dict[int, int],
    episode_to_task_index: dict[int, int],
    task_to_task_index: dict[str, int],
) -> dict[int, list[float] | None]:
    """Return per-task percentile lookup for the dataset rooted at ``root``.

    If ``root / meta/speed_percentiles.jsonl`` already exists, load it
    verbatim — staleness is accepted by design (a WARNING is logged when
    the on-disk file was computed from *fewer* episodes than the current
    load is using, i.e. episodes were appended since; subset loads, where
    the current load uses fewer episodes than on-disk, are silent because
    the on-disk percentiles are a more robust sample). The file is
    trusted in both cases — delete it to force a recompute. Otherwise
    compute the percentiles from ``episode_lengths`` and persist the
    file.

    Append-only migration: when the on-disk file is missing rows for
    tasks that *do* have resolved episodes in this load, percentiles are
    computed for the missing tasks only and appended to the file (logged
    at WARNING so the rewrite is visible). Existing rows are preserved
    verbatim. This recovers datasets whose existing percentile file was
    written before :func:`episode_to_task_index_from_hf_dataset` — those
    files were missing rows for tasks whose ``episodes.jsonl::tasks[0]``
    string didn't match ``tasks.jsonl``, so every episode of those tasks
    bucketed to ``SPARSE_TASK_BUCKET``. The append-only design avoids
    clobbering existing percentiles with subset-derived samples when the
    current load is a mixture / subset; force a full recompute by
    deleting the file.

    Distributed-safe: every rank computes the percentiles in-memory
    (the result is deterministic from the inputs), but only the main
    process writes the file; the ``Accelerator.wait_for_everyone()``
    barrier afterwards ensures non-main ranks don't sail past the
    function while the file is still mid-rename. Each rank returns its
    own in-memory copy rather than re-reading the just-written file.
    The write itself is atomic (per-writer UUID-suffixed tmp file +
    ``os.replace``) so two independent processes (e.g. concurrent
    training jobs sharing a dataset root) can't corrupt each other's
    output even outside the rank-gated case.

    Persistence schema (one line per task)::

        {"task_index": 0, "task": "pick up the red block",
         "n_episodes": 47, "percentiles": [120.0, 145.0, ..., 340.0]}
        {"task_index": 3, "task": "open the drawer",
         "n_episodes": 2, "percentiles": null}

    On read-only roots (write raises ``OSError`` / ``PermissionError``)
    the in-memory dict is still returned and a warning is logged once
    per root.

    Args:
        root: Dataset root directory (the parent of ``meta/``).
        episode_lengths: ``{ep_idx: length_in_frames}`` — typically
            ``LeRobotDataset.episode_lengths`` so the percentile compute
            and the per-episode bucket pre-fill share one source.
        episode_to_task_index: ``{ep_idx: task_idx}`` — typically built
            via :func:`episode_to_task_index_from_hf_dataset`.
        task_to_task_index: ``{task_string: task_index}`` — used only to
            denormalize the task string into each row for human
            inspection.

    Returns:
        Dict mapping ``task_index`` to its 10 percentile boundaries (or
        ``None`` for sparse tasks). Indexed lookup uses
        :func:`bucket_episode_length`.
    """
    path = Path(root) / SPEED_PERCENTILES_PATH
    acc = get_proc_accelerator()
    distributed = acc is not None and acc.num_processes > 1
    is_main_or_solo = (not distributed) or acc.is_main_process

    # NB: the barrier at the end of this function must run on every code path,
    # not just the compute path. Otherwise a rank that arrives *after* rank 0
    # has finished writing the file takes the early-return branch (file now
    # exists), skips the barrier, and silently desyncs the collective counter
    # for every subsequent collective in the mixture-load loop — manifesting
    # as a NCCL hang at a much later (and entirely unrelated) sync point.
    try:
        existing_rows: list[dict] = []
        if path.is_file():
            existing_rows = load_jsonlines(path)
            on_disk_indices = {int(row["task_index"]) for row in existing_rows}
            expected_indices = set(episode_to_task_index.values())
            missing = expected_indices - on_disk_indices
            if not missing:
                if is_main_or_solo:
                    # ``episode_to_task_index`` covers only episodes that
                    # were both (a) resolved by the per-frame ``task_index``
                    # lookup and (b) selected for this load. The directional
                    # check below warns only when the current load has *more*
                    # episodes than the on-disk file was computed from
                    # (i.e. episodes were appended since the file was
                    # written). Subset loads (current uses fewer episodes
                    # than on-disk) are silent — the on-disk percentiles
                    # were computed from a larger, more robust sample, so
                    # using them is if anything an improvement.
                    on_disk_total = sum(int(row.get("n_episodes", 0)) for row in existing_rows)
                    current_total = len(episode_to_task_index)
                    if on_disk_total < current_total:
                        logging.warning(
                            "Stale %s: on-disk percentiles were computed from %d episodes, "
                            "but current load has %d. New episodes may have been appended "
                            "since the file was written. Using on-disk percentiles as-is; "
                            "delete the file to force recompute.",
                            path,
                            on_disk_total,
                            current_total,
                        )
                return {int(row["task_index"]): row["percentiles"] for row in existing_rows}

            # Append-only migration: compute percentiles for the *missing*
            # tasks only, then merge with existing rows. Recomputing every
            # task from the current load would clobber existing rows that
            # were computed from a larger, more robust episode set when
            # the current load is a subset (the dominant case for
            # mixture training). The user can still force a full recompute
            # by deleting the file.
            if is_main_or_solo:
                logging.warning(
                    "Adding %d missing task(s) %s to %s. Existing rows preserved; "
                    "delete the file to force a full recompute.",
                    len(missing),
                    sorted(missing),
                    path,
                )
            task_indices_to_compute = missing
        else:
            # File doesn't exist — fresh compute over every resolved task.
            task_indices_to_compute = set(episode_to_task_index.values())

        # Compute percentiles for the target task set only.
        by_task: dict[int, list[int]] = defaultdict(list)
        for ep_idx, task_idx in episode_to_task_index.items():
            if task_idx in task_indices_to_compute:
                by_task[task_idx].append(episode_lengths[ep_idx])
        index_to_task = {idx: task for task, idx in task_to_task_index.items()}
        new_percentiles = compute_task_percentiles(dict(by_task))
        new_rows = [
            {
                "task_index": task_idx,
                "task": index_to_task.get(task_idx, ""),
                "n_episodes": len(by_task.get(task_idx, [])),
                "percentiles": new_percentiles[task_idx],
            }
            for task_idx in sorted(new_percentiles)
        ]
        rows_to_write = existing_rows + new_rows

        if is_main_or_solo:
            try:
                _atomic_write_jsonlines(rows_to_write, path)
            except (OSError, PermissionError) as e:
                root_key = str(root)
                if root_key not in _READONLY_WARNED:
                    _READONLY_WARNED.add(root_key)
                    logging.warning(
                        "Could not write speed percentiles to %s (%s); using in-memory "
                        "values for this run. The compute will repeat on every load until "
                        "the file can be written.",
                        path,
                        e,
                    )

        # Merge existing on-disk percentiles with newly-computed ones.
        merged = {int(row["task_index"]): row["percentiles"] for row in existing_rows}
        merged.update(new_percentiles)
        return merged
    finally:
        if distributed:
            acc.wait_for_everyone()
