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
per dataset on disk. Existence of the file is the sole gate — staleness
(after ``meta/episodes.jsonl`` is appended to) is accepted; delete the
file to force a recompute.

Tasks with fewer than ``MIN_EPISODES_FOR_PERCENTILES`` *distinct*
episode lengths are written with ``"percentiles": null`` and bucket
every episode to ``SPARSE_TASK_BUCKET`` (= 50). The distinct-count
threshold catches both small-N and degenerate (all-equal-length)
distributions in one rule.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from opentau.datasets.utils import load_jsonlines, write_jsonlines

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
# mixture share the same read-only HF snapshot.
_READONLY_WARNED: set[str] = set()


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


def _episode_lengths_per_task(
    episodes: dict[int, dict],
    task_to_task_index: dict[str, int],
) -> dict[int, list[int]]:
    """Group episode lengths by task_index.

    Episodes whose ``tasks`` field is a multi-element list silently use
    ``tasks[0]`` — the codebase assumes an N-to-1 episode-to-task
    relationship even though the field is structurally a list.
    """
    by_task: dict[int, list[int]] = defaultdict(list)
    for ep_info in episodes.values():
        tasks = ep_info.get("tasks") or []
        if not tasks:
            continue
        task_idx = task_to_task_index[tasks[0]]
        by_task[task_idx].append(int(ep_info["length"]))
    return dict(by_task)


def load_or_compute_speed_percentiles(
    root: Path,
    episodes: dict[int, dict],
    task_to_task_index: dict[str, int],
) -> dict[int, list[float] | None]:
    """Return per-task percentile lookup for the dataset rooted at ``root``.

    If ``root / meta/speed_percentiles.jsonl`` already exists, load it
    verbatim — staleness is accepted by design (delete the file to force
    a recompute). Otherwise compute the percentiles from ``episodes`` and
    persist the file.

    Persistence schema (one line per task)::

        {"task_index": 0, "task": "pick up the red block",
         "n_episodes": 47, "percentiles": [120.0, 145.0, ..., 340.0]}
        {"task_index": 3, "task": "open the drawer",
         "n_episodes": 2, "percentiles": null}

    On read-only roots (write raises ``OSError`` / ``PermissionError``)
    the dict is still returned in-memory and a warning is logged once
    per root.

    Args:
        root: Dataset root directory (the parent of ``meta/``).
        episodes: ``LeRobotDatasetMetadata.episodes``: ``{ep_idx: ep_info}``.
        task_to_task_index: ``LeRobotDatasetMetadata.task_to_task_index``:
            ``{task_string: task_index}``.

    Returns:
        Dict mapping ``task_index`` to its 10 percentile boundaries (or
        ``None`` for sparse tasks). Indexed lookup uses
        :func:`bucket_episode_length`.
    """
    path = Path(root) / SPEED_PERCENTILES_PATH
    by_task = _episode_lengths_per_task(episodes, task_to_task_index)
    index_to_task = {idx: task for task, idx in task_to_task_index.items()}

    if path.is_file():
        rows = load_jsonlines(path)
        return {int(row["task_index"]): row["percentiles"] for row in rows}

    percentiles = compute_task_percentiles(by_task)
    rows = [
        {
            "task_index": task_idx,
            "task": index_to_task.get(task_idx, ""),
            "n_episodes": len(by_task.get(task_idx, [])),
            "percentiles": percentiles[task_idx],
        }
        for task_idx in sorted(percentiles)
    ]
    try:
        write_jsonlines(rows, path)
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
    return percentiles
