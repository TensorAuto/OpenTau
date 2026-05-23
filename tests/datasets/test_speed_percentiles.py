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
    episode_to_task_index_from_hf_dataset,
    load_or_compute_speed_percentiles,
)
from opentau.datasets.utils import load_jsonlines, write_jsonlines


def _fake_meta(task_to_lengths: dict[str, list[int]]):
    """Build the inputs the loader expects from a `{task: [lengths]}` map.

    Returns ``(episode_lengths, episode_to_task_index, task_to_task_index)``
    — the same triple a real ``LeRobotDataset.__init__`` would assemble.
    """
    episode_lengths: dict[int, int] = {}
    e2t: dict[int, int] = {}
    task_to_task_index = {task: i for i, task in enumerate(task_to_lengths)}
    next_ep = 0
    for task, lengths in task_to_lengths.items():
        for length in lengths:
            episode_lengths[next_ep] = length
            e2t[next_ep] = task_to_task_index[task]
            next_ep += 1
    return episode_lengths, e2t, task_to_task_index


def _fake_hf_dataset(per_episode_data: list[tuple[int, int, int]]):
    """Build a fake hf_dataset + episode_data_index + epi2idx.

    Args:
        per_episode_data: list of ``(episode_index, length, task_index)`` tuples,
            one per episode, in the order they should appear in the parquet.

    Returns ``(hf_dataset, episodes, episode_data_index, epi2idx)`` — the same
    four objects that ``LeRobotDataset.__init__`` passes to
    :func:`episode_to_task_index_from_hf_dataset`.
    """
    import torch
    from datasets import Dataset

    task_indices_col: list[int] = []
    episode_index_col: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    epi2idx: dict[int, int] = {}
    pos = 0
    for i, (ep, length, task_idx) in enumerate(per_episode_data):
        starts.append(pos)
        task_indices_col.extend([task_idx] * length)
        episode_index_col.extend([ep] * length)
        pos += length
        ends.append(pos)
        epi2idx[ep] = i
    ds = Dataset.from_dict(
        {
            "task_index": task_indices_col,
            "episode_index": episode_index_col,
        }
    )
    episode_data_index = {
        "from": torch.tensor(starts),
        "to": torch.tensor(ends),
    }
    episodes = [ep for ep, _, _ in per_episode_data]
    return ds, episodes, episode_data_index, epi2idx


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


class TestEpisodeToTaskIndexFromHfDataset:
    """Pin the parquet-keyed lookup that replaced the old string-keyed one.

    The previous implementation matched ``episodes.jsonl::tasks[0]`` against
    ``tasks.jsonl::task``; any paraphrase in either file broke the match
    and forced every affected episode to ``SPARSE_TASK_BUCKET``. The new
    implementation reads the authoritative integer ``task_index`` from the
    per-frame parquet, so paraphrases are inert and the warning fires only
    when the parquet itself references an index that ``tasks.jsonl``
    doesn't define (genuine metadata corruption).
    """

    @pytest.fixture(autouse=True)
    def _reset_warn_cache(self, monkeypatch):
        # The dedup set is module-level, so without reset a test that warns
        # would suppress the warning in the next test for the same index.
        from opentau.datasets import speed_percentiles as sp

        monkeypatch.setattr(sp, "_UNKNOWN_TASK_INDEX_WARNED", set())

    def test_happy_path_resolves_every_episode(self):
        ds, episodes, edi, epi2idx = _fake_hf_dataset(
            [
                (0, 50, 0),
                (1, 100, 1),
                (2, 75, 0),
            ]
        )
        out = episode_to_task_index_from_hf_dataset(
            hf_dataset=ds,
            episodes=episodes,
            episode_data_index=edi,
            epi2idx=epi2idx,
            valid_task_indices={0, 1},
        )
        assert out == {0: 0, 1: 1, 2: 0}

    def test_unknown_task_index_skipped_with_warning(self, caplog):
        # Episode 1 points at task_index=99 which isn't in tasks.jsonl
        # (valid_task_indices). It's dropped; resolvable episodes pass.
        ds, episodes, edi, epi2idx = _fake_hf_dataset(
            [
                (0, 50, 0),
                (1, 100, 99),
                (2, 75, 0),
            ]
        )
        with caplog.at_level("WARNING"):
            out = episode_to_task_index_from_hf_dataset(
                hf_dataset=ds,
                episodes=episodes,
                episode_data_index=edi,
                epi2idx=epi2idx,
                valid_task_indices={0, 1},
            )
        assert out == {0: 0, 2: 0}
        bad_msgs = [r.message for r in caplog.records if "task_index=99" in r.message]
        assert len(bad_msgs) == 1

    def test_repeated_unknown_index_dedupes_warning(self, caplog):
        # Three episodes all reference the same bad index — the deduper
        # collapses to a single warning per distinct index per process.
        ds, episodes, edi, epi2idx = _fake_hf_dataset(
            [
                (0, 50, 99),
                (1, 100, 99),
                (2, 75, 99),
            ]
        )
        with caplog.at_level("WARNING"):
            out = episode_to_task_index_from_hf_dataset(
                hf_dataset=ds,
                episodes=episodes,
                episode_data_index=edi,
                epi2idx=epi2idx,
                valid_task_indices={0, 1},
            )
        assert out == {}
        bad_msgs = [r.message for r in caplog.records if "task_index=99" in r.message]
        assert len(bad_msgs) == 1

    def test_empty_episodes_returns_empty_without_touching_dataset(self):
        # When no episodes are selected, the function must not even peek
        # at the dataset — important for tiny / lazy hf_dataset stubs.
        ds, _, edi, epi2idx = _fake_hf_dataset([(0, 50, 0)])
        out = episode_to_task_index_from_hf_dataset(
            hf_dataset=ds,
            episodes=[],
            episode_data_index=edi,
            epi2idx=epi2idx,
            valid_task_indices={0},
        )
        assert out == {}

    def test_paraphrased_tasks_jsonl_does_not_warn(self, caplog):
        # The whole point of this rewrite: even when tasks.jsonl uses a
        # paraphrased string for what episodes.jsonl calls something else,
        # the parquet's task_index is authoritative — no warning fires
        # because no string comparison happens.
        ds, episodes, edi, epi2idx = _fake_hf_dataset(
            [
                (0, 50, 0),
                (1, 100, 1),
            ]
        )
        with caplog.at_level("WARNING"):
            out = episode_to_task_index_from_hf_dataset(
                hf_dataset=ds,
                episodes=episodes,
                episode_data_index=edi,
                epi2idx=epi2idx,
                valid_task_indices={0, 1},
            )
        assert out == {0: 0, 1: 1}
        assert not any("task_index" in r.message for r in caplog.records)


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

    def test_stale_warning_fires_when_episodes_appended(self, tmp_path, caplog):
        # First write: 100 episodes for taskA.
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        # Second load has *more* episodes for the same task — the on-disk
        # percentiles were computed from fewer episodes than the current
        # load is using (genuine staleness: new data was appended). File
        # is still trusted, but a WARNING fires so the user knows the
        # percentile distribution may not reflect the current dataset.
        el2, e2t2, t2i2 = _fake_meta({"taskA": list(range(100, 250))})  # 150 episodes
        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, el2, e2t2, t2i2)
        assert out[0] is not None
        assert any("Stale" in rec.message and "speed_percentiles" in rec.message for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]

    def test_no_stale_warning_for_subset_load(self, tmp_path, caplog):
        # First write: 150 episodes for taskA.
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 250))})
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        # Second load is a subset — current has *fewer* episodes than
        # the on-disk file was computed from. This is the dominant case
        # for heterogeneous mixture training (a 10% mix loads ~45 of
        # 457 episodes per dataset). The on-disk percentiles are
        # already a more robust sample than a recompute on the subset
        # would produce, so no warning fires.
        el2, e2t2, t2i2 = _fake_meta({"taskA": list(range(100, 150))})  # 50 episodes
        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, el2, e2t2, t2i2)
        assert out[0] is not None
        assert not any("Stale" in rec.message for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]

    def test_no_stale_warning_when_totals_match(self, tmp_path, caplog):
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200))})
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        with caplog.at_level("WARNING"):
            load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert not any("Stale" in rec.message for rec in caplog.records)

    def test_clean_regeneration_after_file_deletion(self, tmp_path, caplog):
        # Workflow: user deletes meta/speed_percentiles.jsonl across the
        # mixture and lets the next run rebuild it from scratch. After
        # deletion, the loader must take the compute path, write a fresh
        # file based on whatever the current load can resolve, and not
        # log spurious staleness or migration warnings on this run or
        # the next.
        el, e2t, t2i = _fake_meta({"taskA": list(range(100, 200)), "taskB": list(range(50, 150))})
        path = tmp_path / SPEED_PERCENTILES_PATH

        # Initial write.
        load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert path.is_file()

        # Simulate the user's "rm meta/speed_percentiles.jsonl" step.
        path.unlink()

        caplog.clear()
        with caplog.at_level("INFO"):
            out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        # File was regenerated.
        assert path.is_file()
        # Both tasks present with valid percentiles.
        assert set(out.keys()) == {0, 1}
        assert all(out[k] is not None and len(out[k]) == 10 for k in out)
        # No noise: no Recomputing-migration log, no Stale warning.
        assert not any("Recomputing" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]
        assert not any("Stale" in r.message for r in caplog.records), [r.message for r in caplog.records]

        # And the regenerated file is self-consistent on the next load.
        caplog.clear()
        with caplog.at_level("INFO"):
            out2 = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert out2 == out
        assert not any("Recomputing" in r.message or "Stale" in r.message for r in caplog.records)

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

    def test_migration_adds_missing_tasks_and_preserves_existing(self, tmp_path, caplog):
        # Simulates the state left by the pre-fix code: tasks.jsonl was
        # paraphrased relative to episodes.jsonl, the string-keyed lookup
        # only resolved a subset of tasks, and the resulting percentile
        # file is missing rows for every drifted task. The new loader
        # detects the missing tasks and *appends* fresh rows for them,
        # leaving existing rows untouched (so subset loads don't clobber
        # full-set percentiles).
        el, e2t, t2i = _fake_meta(
            {
                "taskA": list(range(100, 200)),  # 100 episodes
                "taskB": list(range(50, 150)),  # 100 more episodes — index 1
            }
        )
        path = tmp_path / SPEED_PERCENTILES_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pre-write a file with only taskA — taskB (task_index=1) is
        # missing, simulating the buggy first-pass output. Use deliberately
        # round percentiles that compute would never produce so we can
        # tell whether they survived the migration.
        sentinel_percentiles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        write_jsonlines(
            [
                {
                    "task_index": 0,
                    "task": "taskA",
                    "n_episodes": 100,
                    "percentiles": sentinel_percentiles,
                }
            ],
            path,
        )

        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)

        # Both tasks present in the returned dict.
        assert set(out.keys()) == {0, 1}
        # taskA's pre-existing sentinel percentiles are preserved — this
        # is the key behavior the reviewer flagged. Recomputing would
        # have replaced them with [108.95, ...] (np.percentile of 100..200).
        assert out[0] == sentinel_percentiles
        # taskB was computed fresh from the current load.
        assert out[1] is not None
        # WARNING-level migration log fired with the expected text.
        warn_msgs = [r.message for r in caplog.records if "Adding 1 missing task" in r.message]
        assert warn_msgs, [r.message for r in caplog.records]
        # The on-disk file now has both rows; taskA's row is unchanged.
        rows = load_jsonlines(path)
        assert {int(r["task_index"]) for r in rows} == {0, 1}
        task_a_row = next(r for r in rows if r["task_index"] == 0)
        assert task_a_row["percentiles"] == sentinel_percentiles
        assert task_a_row["n_episodes"] == 100  # also preserved

        # Second call takes the fast read path — no migration, no Stale.
        caplog.clear()
        with caplog.at_level("WARNING"):
            load_or_compute_speed_percentiles(tmp_path, el, e2t, t2i)
        assert not any("Adding" in r.message for r in caplog.records)
        assert not any("Stale" in r.message for r in caplog.records)

    def test_migration_on_subset_load_does_not_clobber_existing(self, tmp_path, caplog):
        # The scenario the reviewer pinned: a pre-fix file was written
        # with full-dataset percentiles for taskA (n=100), but the new
        # load is a 10% subset that resolves both taskA (subset n=10)
        # *and* a previously-drifted taskB. The migration must add a
        # taskB row from the subset, but the high-fidelity taskA
        # percentiles must survive untouched — otherwise the next full
        # load would inherit subset-sized samples for taskA.
        full_lengths = list(range(100, 200))  # 100 distinct lengths
        subset_lengths = full_lengths[:10]
        full_el, full_e2t, t2i = _fake_meta({"taskA": full_lengths, "taskB": list(range(50, 150))})
        # First write: full-dataset taskA only (taskB drifted under old code).
        load_or_compute_speed_percentiles(
            tmp_path,
            {ep: full_lengths[ep] for ep in range(len(full_lengths))},
            dict.fromkeys(range(len(full_lengths)), 0),
            {"taskA": 0, "taskB": 1},
        )
        path = tmp_path / SPEED_PERCENTILES_PATH
        full_task_a = next(r for r in load_jsonlines(path) if r["task_index"] == 0)["percentiles"]

        # Now simulate a subset load that resolves both taskA and taskB.
        subset_el = {}
        subset_e2t = {}
        next_ep = 0
        for length in subset_lengths:
            subset_el[next_ep] = length
            subset_e2t[next_ep] = 0
            next_ep += 1
        for length in list(range(50, 60)):  # 10 subset episodes for taskB
            subset_el[next_ep] = length
            subset_e2t[next_ep] = 1
            next_ep += 1

        with caplog.at_level("WARNING"):
            out = load_or_compute_speed_percentiles(tmp_path, subset_el, subset_e2t, t2i)

        # taskA's full-dataset percentiles must survive.
        assert out[0] == full_task_a, "subset-load migration clobbered full-dataset taskA percentiles"
        # taskB row added (sparse because only 10 subset episodes).
        assert 1 in out
        # On-disk file has both, taskA row unchanged.
        rows = load_jsonlines(path)
        on_disk_task_a = next(r for r in rows if r["task_index"] == 0)
        assert on_disk_task_a["percentiles"] == full_task_a
        assert on_disk_task_a["n_episodes"] == len(full_lengths)


class TestLiberoSnapshot:
    """Sanity check against the well-formed `TensorAuto/libero`
    distribution: every task should be well populated (>= 10 distinct
    lengths) and each yields 10 ascending percentiles spanning a sensible
    frame range. Skips when the dataset isn't cached locally.
    """

    def test_compute_smoke(self):
        lerobot_root = os.environ.get("LEROBOT_HOME") or os.path.expanduser("~/.cache/huggingface/lerobot")
        ep_path = os.path.join(lerobot_root, "TensorAuto/libero/meta/episodes.jsonl")
        if not os.path.isfile(ep_path):
            pytest.skip(f"no local copy of TensorAuto/libero at {ep_path}")
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
