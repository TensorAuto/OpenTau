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
"""Tests for the per-task percentile bucketing in :mod:`opentau.datasets.speed_percentiles`.

Pin the ranking semantics, the sparse-task fallback, the on-disk
persistence contract, and the distributed-write safety so future edits
to the formula or the JSONL schema can't silently change `speed_raw`
distributions.
"""

from __future__ import annotations

import contextlib
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from opentau.datasets.speed_percentiles import (
    MIN_EPISODES_FOR_PERCENTILES,
    SPARSE_TASK_BUCKET,
    SPEED_BUCKET_LABELS,
    SPEED_PERCENTILES,
    SPEED_PERCENTILES_PATH,
    bucket_episode_length,
    compute_task_percentiles,
    episode_to_task_index_from_episodes,
    load_or_compute_speed_percentiles,
)
from opentau.datasets.utils import load_jsonlines, write_jsonlines


def _fake_meta(task_to_lengths: dict[str, list[int]]):
    """Build the inputs the loader expects from a `{task: [lengths]}` map.

    Returns ``(episode_lengths, episode_to_task_index, task_to_task_index)``
    — the same triple a real ``LeRobotDataset.__init__`` would assemble.
    """
    episode_lengths: dict[int, int] = {}
    episodes: dict[int, dict] = {}
    task_to_task_index = {task: i for i, task in enumerate(task_to_lengths)}
    next_ep = 0
    for task, lengths in task_to_lengths.items():
        for length in lengths:
            episode_lengths[next_ep] = length
            episodes[next_ep] = {"episode_index": next_ep, "tasks": [task], "length": length}
            next_ep += 1
    e2t = episode_to_task_index_from_episodes(episodes, task_to_task_index)
    return episode_lengths, e2t, task_to_task_index


class TestComputeTaskPercentiles:
    def test_well_populated_returns_ten_ascending_floats(self):
        # 100 distinct lengths well above the threshold.
        lengths = list(range(100, 200))
        out = compute_task_percentiles({0: lengths})
        assert out[0] is not None
        assert len(out[0]) == len(SPEED_PERCENTILES) == 10
        assert all(isinstance(p, float) for p in out[0])
        assert out[0] == sorted(out[0])

    def test_single_episode_task_returns_none(self):
        out = compute_task_percentiles({0: [123]})
        assert out[0] is None

    def test_two_episode_task_returns_none(self):
        out = compute_task_percentiles({0: [100, 200]})
        assert out[0] is None

    def test_below_threshold_distinct_returns_none(self):
        # 100 episodes but only 9 distinct lengths → still sparse.
        lengths = ([10, 20, 30, 40, 50, 60, 70, 80, 90] * 12)[:100]
        assert len(set(lengths)) < MIN_EPISODES_FOR_PERCENTILES
        out = compute_task_percentiles({0: lengths})
        assert out[0] is None

    def test_all_equal_lengths_returns_none(self):
        out = compute_task_percentiles({0: [200] * 100})
        assert out[0] is None

    def test_per_task_independence(self):
        out = compute_task_percentiles(
            {
                0: list(range(100, 200)),  # well populated
                1: [42],  # single-episode sparse
                2: [10] * 50,  # all-equal sparse
            }
        )
        assert out[0] is not None
        assert out[1] is None
        assert out[2] is None


class TestEpisodeToTaskIndex:
    def test_single_task_episodes(self):
        episodes = {
            0: {"tasks": ["taskA"], "length": 100},
            1: {"tasks": ["taskB"], "length": 200},
        }
        t2i = {"taskA": 0, "taskB": 1}
        assert episode_to_task_index_from_episodes(episodes, t2i) == {0: 0, 1: 1}

    def test_multi_task_uses_first(self):
        episodes = {0: {"tasks": ["taskA", "taskB"], "length": 100}}
        t2i = {"taskA": 0, "taskB": 1}
        assert episode_to_task_index_from_episodes(episodes, t2i) == {0: 0}

    def test_empty_tasks_skipped(self):
        episodes = {
            0: {"tasks": [], "length": 100},
            1: {"length": 200},
            2: {"tasks": ["taskA"], "length": 300},
        }
        t2i = {"taskA": 0}
        assert episode_to_task_index_from_episodes(episodes, t2i) == {2: 0}


class TestBucketEpisodeLength:
    @pytest.fixture
    def percentiles(self) -> list[float]:
        # Hand-picked round numbers so boundary tests are unambiguous.
        return [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]

    def test_below_p5_returns_zero(self, percentiles):
        assert bucket_episode_length(0, percentiles) == 0
        assert bucket_episode_length(49, percentiles) == 0

    def test_at_p5_exactly_returns_ten(self, percentiles):
        # Tie at p5 lands in the upper bucket per searchsorted(side='right').
        assert bucket_episode_length(50, percentiles) == 10

    def test_at_p95_exactly_returns_one_hundred(self, percentiles):
        assert bucket_episode_length(500, percentiles) == 100

    def test_above_p95_returns_one_hundred(self, percentiles):
        assert bucket_episode_length(99999, percentiles) == 100

    @pytest.mark.parametrize(
        "length, expected",
        [
            (49, 0),
            (50, 10),
            (75, 10),
            (100, 20),
            (150, 30),
            (200, 40),
            (250, 50),
            (300, 60),
            (350, 70),
            (400, 80),
            (450, 90),
            (500, 100),
            (501, 100),
        ],
    )
    def test_boundary_walk(self, percentiles, length, expected):
        assert bucket_episode_length(length, percentiles) == expected

    def test_sparse_fallback_returns_median_bucket(self):
        assert bucket_episode_length(123, None) == SPARSE_TASK_BUCKET == 50

    def test_returns_python_int(self, percentiles):
        # `_emit_optional_keys` later wraps with `int(raw)`; numpy ints
        # would slip through but a plain Python int is the contract.
        out = bucket_episode_length(75, percentiles)
        assert type(out) is int

    def test_label_set_invariant(self, percentiles):
        # Every possible length must map to one of the 11 declared labels.
        for length in (0, 49, 50, 100, 250, 500, 9999):
            assert bucket_episode_length(length, percentiles) in SPEED_BUCKET_LABELS


class TestLoadOrComputeSpeedPercentiles:
    def test_writes_file_when_absent(self, tmp_path):
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        path = tmp_path / SPEED_PERCENTILES_PATH
        assert path.is_file()
        assert out[0] is not None and len(out[0]) == 10

        # Round-trip: bucket using both the in-memory dict and what was
        # written to disk yields the same answer.
        rows = load_jsonlines(path)
        on_disk = {row["task_index"]: row["percentiles"] for row in rows}
        for length in (50, 150, 250, 350, 500):
            assert bucket_episode_length(length, out[0]) == bucket_episode_length(length, on_disk[0])

    def test_existing_file_is_loaded_verbatim(self, tmp_path):
        # Pre-write a hand-crafted file with values that compute would
        # never produce — proves we read it without re-computing.
        path = tmp_path / SPEED_PERCENTILES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(
            [
                {
                    "task_index": 0,
                    "task": "taskA",
                    "n_episodes": 1000,
                    "percentiles": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                }
            ],
            path,
        )

        # Episodes that match n_episodes exactly so no stale-warning fires.
        el, e2t, t2i = _fake_meta({"taskA": list(range(1000, 2000))})
        out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        # The hand-written percentiles are returned, not what compute
        # would have produced from the episodes.
        assert out[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    def test_round_trip_preserves_buckets(self, tmp_path):
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})

        first = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        # Second call hits the file path.
        second = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        for length in (50, 100, 150, 175, 200, 250):
            assert bucket_episode_length(length, first[0]) == bucket_episode_length(length, second[0])

    def test_sparse_task_persisted_with_null_percentiles(self, tmp_path):
        el, e2t, t2i = _fake_meta({"taskA": [42]})
        out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert out[0] is None

        rows = load_jsonlines(tmp_path / SPEED_PERCENTILES_PATH)
        assert rows[0]["task_index"] == 0
        assert rows[0]["n_episodes"] == 1
        assert rows[0]["percentiles"] is None

        # Reload reproduces the None.
        out2 = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert out2[0] is None
        # And the bucket lookup falls back to SPARSE_TASK_BUCKET.
        assert bucket_episode_length(42, out2[0]) == SPARSE_TASK_BUCKET

    def test_stale_file_logs_warning_but_still_used(self, tmp_path, caplog):
        # First write: 100 episodes for taskA.
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        # Second load with a different episode count — file should still
        # be used, but a WARNING fires.
        el2, e2t2, t2i2 = _fake_meta({"taskA": list(range(100, 250))})  # 150 episodes
        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, el2, e2t2, t2i2)
        assert out[0] is not None
        assert any("Stale" in rec.message and "speed_percentiles" in rec.message for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]

    def test_no_stale_warning_when_totals_match(self, tmp_path, caplog):
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        with caplog.at_level("WARNING"):
            load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert not any("Stale" in rec.message for rec in caplog.records)

    def test_concurrent_writes_produce_valid_file(self, tmp_path):
        # Multiple threads racing on the same root must not corrupt the
        # file. Per-writer UUID-suffixed tmp paths + os.replace are what
        # make this safe outside of the rank-gated path. Heterogeneous
        # task sets across workers force the bucket-index path to vary
        # per call so a torn write would be visible as a partial row
        # set, not just identical content overwriting itself.
        many_tasks = {f"task{i}": list(range(100 + i * 5, 200 + i * 5)) for i in range(5)}
        el, e2t, t2i = _fake_meta(many_tasks)

        def worker(_):
            return load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(16)))

        # All workers must agree on every task's percentiles.
        first = results[0]
        for r in results[1:]:
            assert r == first

        # The on-disk file must be parseable, complete, and consistent
        # with the in-memory result. A torn write would surface as a
        # JSON parse error, a wrong row count, or a percentile mismatch.
        rows = load_jsonlines(tmp_path / SPEED_PERCENTILES_PATH)
        assert len(rows) == len(many_tasks)
        from_disk = {int(row["task_index"]): row["percentiles"] for row in rows}
        assert from_disk == first
        for row in rows:
            assert row["n_episodes"] == 100

        # And no leftover .tmp files.
        leftovers = list((tmp_path / "meta").glob("speed_percentiles.jsonl.*.tmp"))
        assert leftovers == []

    def test_falls_back_to_in_memory_on_readonly_root(self, tmp_path, caplog, monkeypatch):
        # Make the meta/ dir read-only so write_jsonlines raises.
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        meta_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        # Force the write attempt to fail even if permissions don't (CI
        # often runs as root and ignores chmod). We patch the atomic
        # writer in the speed_percentiles module to always raise.
        from opentau.datasets import speed_percentiles as sp

        def _raise(*a, **kw):
            raise PermissionError("read-only test")

        monkeypatch.setattr(sp, "_atomic_write_jsonlines", _raise)
        # Reset the per-root warn-once cache so caplog sees the message.
        monkeypatch.setattr(sp, "_READONLY_WARNED", set())

        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        assert out[0] is not None and len(out[0]) == 10
        assert any("speed percentiles" in rec.message for rec in caplog.records)
        # Restore so tmp_path cleanup can remove the dir.
        with contextlib.suppress(OSError):
            meta_dir.chmod(stat.S_IRWXU)

    def test_handles_missing_task_in_lookup_defensively(self, tmp_path):
        # A hand-edited percentile file might omit a task that exists in
        # episodes; the consumer (lerobot_dataset's pre-bucket pass) uses
        # `.get(task_idx)` to fall back to the sparse bucket.
        path = tmp_path / SPEED_PERCENTILES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines([], path)

        out = load_or_compute_speed_percentiles(tmp_path, {}, {}, {})
        assert out == {}
        # `.get(missing_idx)` returns None → bucket 50.
        assert bucket_episode_length(123, out.get(99)) == SPARSE_TASK_BUCKET


class TestLiberoSnapshot:
    """Sanity check against the well-formed `physical-intelligence/libero`
    distribution: every task should be well populated (>= 10 distinct
    lengths) and each yields 10 ascending percentiles spanning a sensible
    frame range. Skips when the dataset isn't cached locally.
    """

    def test_compute_smoke(self):
        lerobot_root = os.environ.get("LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
        ep_path = os.path.join(lerobot_root, "physical-intelligence/libero/meta/episodes.jsonl")
        if not os.path.isfile(ep_path):
            pytest.skip(f"no local copy of physical-intelligence/libero at {ep_path}")
        import json
        from collections import defaultdict

        by_task: dict[str, list[int]] = defaultdict(list)
        with open(ep_path) as f:
            for line in f:
                d = json.loads(line)
                assert len(d["tasks"]) == 1
                by_task[d["tasks"][0]].append(d["length"])
        # Re-key by integer task_index for compute_task_percentiles.
        as_idx = {i: by_task[t] for i, t in enumerate(by_task)}
        out = compute_task_percentiles(as_idx)
        for idx, pcts in out.items():
            assert pcts is not None, f"task {idx} should be well populated"
            assert pcts == sorted(pcts)
            assert all(p > 0 for p in pcts)
            # Every observed length lands in a valid bucket label.
            for length in as_idx[idx]:
                assert bucket_episode_length(length, pcts) in SPEED_BUCKET_LABELS
