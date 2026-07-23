#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Portions of this file (the `RunningStats` histogram quantile estimator) are adapted from
# openpi (https://github.com/Physical-Intelligence/openpi), `src/openpi/shared/normalize.py`,
# Copyright Physical Intelligence, licensed under the Apache License, Version 2.0 — the same
# license as this project. Keeping the estimator identical is deliberate: it makes the
# quantiles we compute here directly comparable to an openpi-produced `norm_stats.json`.
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

"""Delta-action normalization statistics, computed on the fly and cached to disk.

Why this exists: when ``use_delta_joint_actions`` is on, each action in a chunk has the
chunk-start state subtracted, so the action distribution the policy trains against is *not* the
one recorded in the dataset's ``meta/stats.json`` (which describes absolute, per-frame actions).
Reusing those stats misscales every mapped dim — in practice by 2-5x, since a joint's
displacement over a chunk is far smaller than its absolute range.

Recomputing them is unavoidably more work than reading a file: a chunk of length ``H`` turns each
frame into ``H`` delta values, so the pass is ``H`` times the data volume of a per-frame stats
pass. Two things keep that affordable:

* **No video decode.** Only the numeric ``state`` / ``action`` columns are read, straight out of
  parquet. Video is never touched, which is what makes an ``H``-fold pass tractable at all.
* **Disk cache.** Results are written under each dataset's own root, keyed by a hash of
  everything that affects them (column indices, delta map, chunk length, resample settings, ...),
  so the cost is paid once per configuration rather than once per run.

Distribution: the compute is rank-0-only, and other ranks wait by **polling the filesystem**
rather than sitting in a collective — see :func:`load_or_compute_delta_action_stats`.
"""

import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "RunningStats",
    "compute_delta_action_stats",
    "delta_stats_cache_key",
    "load_or_compute_delta_action_stats",
]

# Subdirectory under a dataset's root where cache files land. Sits beside `meta/stats.json` and
# `meta/speed_percentiles.jsonl`, so a dataset carries its derived stats with it.
CACHE_SUBDIR = Path("meta") / "delta_action_stats"

# Schema version of the cache payload. Bump when the on-disk layout or the compute semantics
# change in a way that invalidates existing files; stale files are then ignored rather than
# silently reused.
CACHE_VERSION = 1

# How long a non-main rank waits for rank 0's compute before giving up. Generous because the
# pass is O(frames x chunk_size) over every dataset in the mixture and runs once per config.
POLL_TIMEOUT_S = 3 * 60 * 60
POLL_INTERVAL_S = 5.0
POLL_LOG_EVERY_S = 120.0

_READONLY_WARNED: set[str] = set()


class RunningStats:
    """Streaming mean/std/min/max plus histogram-based quantiles.

    Adapted from openpi's ``openpi.shared.normalize.RunningStats`` (Apache-2.0). Quantiles come
    from a fixed 5000-bin per-dimension histogram whose edges are rebuilt whenever the observed
    range grows, so memory is constant regardless of how many samples stream through — necessary
    here because the delta pass produces ``frames x chunk_size`` values per dimension.

    Rebinning redistributes existing counts by their **old left edges**, which smears the
    distribution a little every time the range grows. With randomly-ordered input that costs
    about one bin of accuracy, but when the observed range grows monotonically across updates —
    which is what happens reading a dataset episode by episode — the error compounds badly
    (measured at ~46 bins on a monotonically-widening stream).

    ``bounds`` removes that failure mode: pre-seeding the edges with the data's true range means
    the range never grows, so ``_adjust_histograms`` never fires and the estimate stays within
    one bin. Callers that know the range up front (via a cheap min/max pre-pass) should pass it.
    Constructed **without** ``bounds`` this class behaves exactly like openpi's, so it can still
    be used to reproduce openpi's numbers directly.
    """

    def __init__(self, num_quantile_bins: int = 5000, bounds: tuple[np.ndarray, np.ndarray] | None = None):
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = num_quantile_bins
        self._bounds = bounds

    def update(self, batch: np.ndarray) -> None:
        """Fold a batch of vectors into the running statistics.

        Args:
            batch: Array whose **last** axis is the feature axis; every leading axis is treated
                as a batch axis. A ``(frames, chunk, dim)`` delta tensor therefore contributes
                ``frames * chunk`` samples, which is exactly the intent.
        """
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            # With `bounds`, seed the edges from the caller's known range so they never need
            # adjusting; otherwise seed from this first batch, exactly as openpi does.
            lo, hi = self._bounds if self._bounds is not None else (self._min, self._max)
            self._bin_edges = [
                np.linspace(lo[i] - 1e-10, hi[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    f"vector length {vector_length} does not match the initialized length {self._mean.size}."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            changed = bool(np.any(new_max > self._max) or np.any(new_min < self._min))
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            # Fixed edges from `bounds` already cover the data, so there is nothing to adjust —
            # and skipping it is precisely what avoids the compounding rebinning error.
            if changed and self._bounds is None:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )
        self._update_histograms(batch)

    def _adjust_histograms(self) -> None:
        """Rebuild bin edges over the widened range, redistributing existing counts."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: Sequence[float]) -> list[np.ndarray]:
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[min(int(idx), len(edges) - 1)])
            results.append(np.array(q_values))
        return results

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Return the accumulated stats in OpenTau's per-feature stats layout.

        Returns:
            ``{"mean", "std", "min", "max", "q01", "q99", "count"}``, all ``float32`` except
            ``count`` (``int64``, shape ``(1,)`` to match `compute_stats`' convention).

        Raises:
            ValueError: If fewer than two samples were accumulated.
        """
        if self._count < 2:
            raise ValueError(f"cannot compute statistics from {self._count} sample(s).")
        variance = self._mean_of_squares - self._mean**2
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return {
            "mean": self._mean.astype(np.float32),
            "std": np.sqrt(np.maximum(0, variance)).astype(np.float32),
            "min": self._min.astype(np.float32),
            "max": self._max.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "count": np.array([self._count], dtype=np.int64),
        }


def delta_stats_cache_key(
    *,
    state_index: list[int] | None,
    action_index: list[int] | None,
    delta_map: dict[int, int] | None,
    chunk_offsets: Sequence[float],
    vector_resample_strategy: str,
    episodes: Sequence[int] | None,
    excluded_episodes: Sequence[int] | None,
    fps: float | None,
    revision: str | None,
) -> str:
    """Hash everything that changes the computed stats into a short cache key.

    The dataset's *root* supplies identity (the cache lives under it), so the key only has to
    separate configurations of the same dataset: a different chunk length, delta map, column
    selection or episode subset must not collide. ``revision`` is folded in when known so a
    re-pinned dataset does not reuse the previous revision's numbers.

    Args:
        state_index: ``DatasetConfig.state_index`` (parquet space) or ``None``.
        action_index: ``DatasetConfig.action_index`` (parquet space) or ``None``.
        delta_map: Post-index ``{action_pos: state_pos}`` map, or ``None``.
        chunk_offsets: Per-chunk frame offsets, i.e. the action horizon in index space.
        vector_resample_strategy: ``"nearest"`` or ``"linear"``.
        episodes: Selected episode indices, or ``None`` for all.
        excluded_episodes: Dropped episode indices, or ``None``.
        fps: Dataset frame rate.
        revision: Dataset revision when pinned.

    Returns:
        A 12-character hex digest.
    """
    payload = {
        "version": CACHE_VERSION,
        "state_index": list(state_index) if state_index is not None else None,
        "action_index": list(action_index) if action_index is not None else None,
        # Sorted int-keyed pairs: dict ordering must not perturb the key.
        "delta_map": sorted((int(a), int(s)) for a, s in (delta_map or {}).items()),
        # Rounded so float noise in the offsets can't fragment the cache.
        "chunk_offsets": [round(float(o), 6) for o in chunk_offsets],
        "vector_resample_strategy": vector_resample_strategy,
        "episodes": sorted(int(e) for e in episodes) if episodes is not None else None,
        "excluded_episodes": (
            sorted(int(e) for e in excluded_episodes) if excluded_episodes is not None else None
        ),
        "fps": round(float(fps), 6) if fps is not None else None,
        "revision": revision,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _gather_chunks(
    values: np.ndarray,
    frame_rows: np.ndarray,
    offsets: np.ndarray,
    ep_start: int,
    ep_end: int,
    strategy: str,
) -> np.ndarray:
    """Gather ``(len(frame_rows), len(offsets), dim)`` chunks, clipped to the episode.

    Mirrors ``BaseDataset._get_query_indices_soft`` + ``_query_hf_dataset_soft``: soft indices are
    clipped into ``[ep_start, ep_end - 1]`` (frames near the episode end repeat the last row, and
    training marks those padded rather than dropping them, so they are kept here too), then
    resolved to rows by rounding or linear interpolation.

    Args:
        values: ``(rows, dim)`` column data for the whole file.
        frame_rows: Absolute row indices of the chunk anchors.
        offsets: Soft frame offsets defining the horizon.
        ep_start: First row of the episode.
        ep_end: One past the last row of the episode.
        strategy: ``"nearest"`` or ``"linear"``.

    Returns:
        The gathered chunk tensor.

    Raises:
        ValueError: On an unknown resample strategy.
    """
    soft = frame_rows[:, None] + offsets[None, :]
    soft = np.clip(soft, ep_start, ep_end - 1)
    if strategy == "nearest":
        return values[np.rint(soft).astype(np.int64)]
    if strategy == "linear":
        floor = np.floor(soft).astype(np.int64)
        frac = soft - floor
        ceil = np.clip(floor + (frac > 0.0), ep_start, ep_end - 1)
        lo = values[floor]
        hi = values[ceil]
        return lo + frac[..., None] * (hi - lo)
    raise ValueError(f"Unsupported vector_resample_strategy: {strategy!r}. Use 'linear' or 'nearest'.")


def _process_parquet_file(task: dict[str, Any]) -> dict[str, dict[str, np.ndarray]] | None:
    """Accumulate one parquet file's contribution. Runs inside a worker process.

    Returns ``None`` when the file yields nothing usable (missing, unreadable, or no selected
    episode) so the caller can skip it without special-casing empty accumulators.
    """
    import pyarrow.parquet as pq  # lazy: keep workers torch-free and fast to spawn

    path = task["path"]
    if not os.path.exists(path):
        logging.warning("delta stats: missing parquet %s; skipping.", path)
        return None
    try:
        table = pq.read_table(path, columns=[task["state_col"], task["action_col"], "episode_index"])
    except Exception as e:  # noqa: BLE001 - one bad shard must not sink the dataset
        logging.warning("delta stats: skip %s: %r", path, e)
        return None

    states = np.asarray(table.column(task["state_col"]).to_pylist(), dtype=np.float64)
    actions = np.asarray(table.column(task["action_col"]).to_pylist(), dtype=np.float64)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    if states.ndim != 2 or actions.ndim != 2 or len(states) == 0:
        logging.warning(
            "delta stats: skip %s: unexpected shapes state=%s action=%s",
            path,
            states.shape,
            actions.shape,
        )
        return None

    if task["state_index"] is not None:
        states = states[:, task["state_index"]]
    if task["action_index"] is not None:
        actions = actions[:, task["action_index"]]

    selected = task["episodes"]
    offsets = np.asarray(task["chunk_offsets"], dtype=np.float64)
    delta_map = {int(a): int(s) for a, s in task["delta_map"].items()}
    a_pos = np.array(sorted(delta_map), dtype=np.int64)
    s_pos = np.array([delta_map[int(a)] for a in a_pos], dtype=np.int64)

    block = int(task["block_frames"])
    # Rows are ordered by ascending episode_index (asserted at dataset load), so contiguous runs
    # of equal values delimit episodes.
    breaks = np.flatnonzero(np.diff(episodes)) + 1
    spans = [
        (int(lo), int(hi))
        for lo, hi in zip(
            np.concatenate(([0], breaks)), np.concatenate((breaks, [len(episodes)])), strict=True
        )
        if selected is None or int(episodes[lo]) in selected
    ]
    if not spans:
        return None

    def _iter_delta_chunks():
        """Yield ``(frames, horizon, dim)`` delta-action blocks, episode by episode."""
        for ep_start, ep_end in spans:
            rows = np.arange(ep_start, ep_end, dtype=np.int64)
            for begin in range(0, len(rows), block):
                frame_rows = rows[begin : begin + block]
                chunk = _gather_chunks(actions, frame_rows, offsets, ep_start, ep_end, task["strategy"])
                if a_pos.size:
                    # One state per chunk — the anchor frame's — broadcast across the horizon.
                    # This is the defining property of openpi's `DeltaActions`.
                    chunk[:, :, a_pos] -= states[frame_rows][:, None, s_pos]
                yield chunk

    # Pass 1: the delta range, so the histogram edges can be fixed up front. Costs a few percent
    # of the histogram pass (measured) and buys back the accuracy that repeated rebinning would
    # otherwise cost on an episode-ordered stream, where the range grows monotonically.
    lo = np.full(actions.shape[1], np.inf)
    hi = np.full(actions.shape[1], -np.inf)
    for chunk in _iter_delta_chunks():
        flat = chunk.reshape(-1, chunk.shape[-1])
        lo = np.minimum(lo, flat.min(axis=0))
        hi = np.maximum(hi, flat.max(axis=0))

    # Pass 2: accumulate against those fixed edges.
    action_acc = RunningStats(bounds=(lo, hi))
    for chunk in _iter_delta_chunks():
        action_acc.update(chunk)

    state_rows = np.concatenate([np.arange(lo_, hi_, dtype=np.int64) for lo_, hi_ in spans])
    state_values = states[state_rows]
    state_acc = RunningStats(bounds=(state_values.min(axis=0), state_values.max(axis=0)))
    state_acc.update(state_values)

    if action_acc._count < 2 or state_acc._count < 2:
        return None
    return {"actions": action_acc.get_statistics(), "state": state_acc.get_statistics()}


def _merge_running(parts: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    """Combine per-file statistics into one set via the existing weighted aggregator.

    Reuses :func:`opentau.datasets.compute_stats.aggregate_feature_stats` so pooling here obeys
    the same weighted-mean/variance rules as every other stats path — including the all-or-none
    quantile requirement, which holds trivially since every part is produced by
    :class:`RunningStats`.
    """
    from opentau.datasets.compute_stats import aggregate_feature_stats

    merged: dict[str, dict[str, np.ndarray]] = {}
    for feature in ("actions", "state"):
        entries = [p[feature] for p in parts if feature in p]
        if not entries:
            continue
        merged[feature] = entries[0] if len(entries) == 1 else aggregate_feature_stats(entries)
    return merged


def compute_delta_action_stats(
    *,
    parquet_paths: Sequence[str],
    state_col: str,
    action_col: str,
    state_index: list[int] | None,
    action_index: list[int] | None,
    delta_map: dict[int, int],
    chunk_offsets: Sequence[float],
    strategy: str,
    episodes: set[int] | None,
    max_workers: int = 1,
    block_frames: int = 4096,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute delta-action (and state) stats for one dataset from its parquet files.

    Args:
        parquet_paths: Deduplicated data-file paths for this dataset.
        state_col: Raw state column name.
        action_col: Raw action column name.
        state_index: Parquet-space state column selection, or ``None``.
        action_index: Parquet-space action column selection, or ``None``.
        delta_map: Post-index ``{action_pos: state_pos}`` map. May be empty, in which case the
            actions are accumulated as-is (still chunked, which is what training sees).
        chunk_offsets: Soft frame offsets defining the action horizon.
        strategy: ``vector_resample_strategy``.
        episodes: Selected episode indices, or ``None`` for all.
        max_workers: Process-pool width. ``1`` runs inline, which keeps stack traces readable
            and avoids pool overhead for a single small file.
        block_frames: Chunk anchors processed per gather, bounding peak memory at roughly
            ``block_frames x len(chunk_offsets) x dim`` float64 values.

    Returns:
        ``{"actions": {...}, "state": {...}}`` in OpenTau's per-feature stats layout.

    Raises:
        ValueError: If no file produced usable samples.
    """
    tasks = [
        {
            "path": str(p),
            "state_col": state_col,
            "action_col": action_col,
            "state_index": state_index,
            "action_index": action_index,
            "delta_map": delta_map,
            "chunk_offsets": list(chunk_offsets),
            "strategy": strategy,
            "episodes": episodes,
            "block_frames": block_frames,
        }
        for p in parquet_paths
    ]
    if not tasks:
        raise ValueError("compute_delta_action_stats: no parquet paths given.")

    if max_workers <= 1 or len(tasks) == 1:
        parts = [_process_parquet_file(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            parts = list(pool.map(_process_parquet_file, tasks))

    usable = [p for p in parts if p]
    if not usable:
        raise ValueError(
            "compute_delta_action_stats: every parquet file was empty, unreadable, or excluded "
            f"by the episode selection ({len(tasks)} file(s) tried)."
        )
    return _merge_running(usable)


def _serialize(stats: dict[str, dict[str, np.ndarray]]) -> str:
    payload = {
        "version": CACHE_VERSION,
        "stats": {
            feature: {name: np.asarray(value).tolist() for name, value in feat.items()}
            for feature, feat in stats.items()
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def _deserialize(text: str) -> dict[str, dict[str, np.ndarray]] | None:
    """Parse a cache file, returning ``None`` for anything unusable (stale, truncated, corrupt)."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    out: dict[str, dict[str, np.ndarray]] = {}
    for feature, feat in payload.get("stats", {}).items():
        out[feature] = {
            name: np.asarray(value, dtype=np.int64 if name == "count" else np.float32)
            for name, value in feat.items()
        }
    return out or None


def _atomic_write_text(text: str, path: Path) -> None:
    """Write ``text`` to ``path`` via a per-writer tmp file + ``os.replace``.

    Mirrors ``speed_percentiles._atomic_write_jsonlines``: the UUID suffix keeps two concurrent
    writers from truncating each other's tmp file, and ``os.replace`` makes the swap the
    point-of-commit, so a reader sees either the old file or the new one but never a partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def load_or_compute_delta_action_stats(
    *,
    root: Path,
    cache_key: str,
    compute_kwargs: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Return cached delta-action stats for a dataset, computing them once if absent.

    **Distributed behavior.** Only the main process computes. Other ranks wait by polling the
    filesystem, *not* by entering a collective: dataset construction runs after ``Accelerator()``
    with no raised ``InitProcessGroupKwargs`` timeout, so parking every other rank in
    ``wait_for_everyone()`` while rank 0 spends minutes on an ``O(frames x chunk)`` pass would
    trip NCCL's 30-minute watchdog and abort the run. Polling keeps zero collectives in flight,
    so the watchdog has nothing to time out.

    A single short ``wait_for_everyone()`` runs afterwards in a ``finally``, so **every** path —
    cache hit, fresh compute, or read-only fallback — reaches the same barrier. Skipping it on an
    early return would desync the collective counter and surface as a hang at an unrelated sync
    point much later (the lesson recorded in ``speed_percentiles.py``).

    Args:
        root: Dataset root; the cache lives at ``root / meta / delta_action_stats / <key>.json``.
        cache_key: Digest from :func:`delta_stats_cache_key`.
        compute_kwargs: Forwarded verbatim to :func:`compute_delta_action_stats`.

    Returns:
        ``{"actions": {...}, "state": {...}}``.

    Raises:
        TimeoutError: If a non-main rank waits past :data:`POLL_TIMEOUT_S`, which in practice
            means rank 0 died mid-compute.
    """
    # Imported lazily: only this function needs the accelerator. Keeping it off the module's
    # import path means the pool workers (which import this module to unpickle
    # `_process_parquet_file`) don't pay for `accelerate`, and the estimator stays usable from
    # lightweight contexts such as the openpi parity harness.
    from opentau.utils.accelerate_utils import get_proc_accelerator

    path = Path(root) / CACHE_SUBDIR / f"{cache_key}.json"
    acc = get_proc_accelerator()
    distributed = acc is not None and acc.num_processes > 1
    is_main_or_solo = (not distributed) or acc.is_main_process

    try:
        if path.is_file():
            cached = _deserialize(path.read_text())
            if cached is not None:
                logging.info("delta stats: cache hit %s", path)
                return cached
            logging.warning("delta stats: ignoring unreadable/stale cache %s; recomputing.", path)

        if not is_main_or_solo:
            return _wait_for_main(path)

        logging.info("delta stats: computing (cache miss) -> %s", path)
        started = time.monotonic()
        stats = compute_delta_action_stats(**compute_kwargs)
        logging.info("delta stats: computed in %.1fs -> %s", time.monotonic() - started, path)
        try:
            _atomic_write_text(_serialize(stats), path)
        except (OSError, PermissionError) as e:
            root_key = str(root)
            if root_key not in _READONLY_WARNED:
                _READONLY_WARNED.add(root_key)
                logging.warning(
                    "delta stats: could not write %s (%s); using in-memory values for this run. "
                    "The compute will repeat on every load until the path is writable.",
                    path,
                    e,
                )
        return stats
    finally:
        if distributed:
            acc.wait_for_everyone()


def _wait_for_main(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Block until rank 0 publishes ``path``, polling the filesystem.

    Deliberately does not use a collective — see :func:`load_or_compute_delta_action_stats`.

    Raises:
        TimeoutError: If the file never appears within :data:`POLL_TIMEOUT_S`.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_S
    last_log = 0.0
    while time.monotonic() < deadline:
        if path.is_file():
            cached = _deserialize(path.read_text())
            if cached is not None:
                return cached
            # Present but unparsable: rank 0 may be mid-rename, or wrote a stale version.
            # Keep waiting rather than racing it.
        waited = POLL_TIMEOUT_S - (deadline - time.monotonic())
        if waited - last_log >= POLL_LOG_EVERY_S:
            last_log = waited
            logging.info("delta stats: waiting %.0fs for the main process to publish %s", waited, path)
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"delta stats: timed out after {POLL_TIMEOUT_S}s waiting for {path}. The main process "
        "most likely failed during the delta-action stats computation — check its log for the "
        "original traceback."
    )
