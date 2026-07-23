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

"""On-the-fly delta-action statistics and their disk cache."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from opentau.datasets.delta_action_stats import (
    CACHE_SUBDIR,
    RunningStats,
    compute_delta_action_stats,
    delta_stats_cache_key,
    load_or_compute_delta_action_stats,
)

# Sized so the 1st percentile is well-determined: N_EP * T_PER_EP * HORIZON = 7200 delta samples
# puts ~72 in the tail the q01 assertion probes. At a few hundred samples the 1st percentile is
# only the ~6th order statistic, and `np.quantile`'s interpolation vs. a histogram bin edge
# disagree by far more than either estimator's actual error.
T_PER_EP = 300
N_EP = 4
DIM_A = 4
DIM_S = 6
HORIZON = 6


def _write_dataset(root, seed=0):
    """Write a single-file parquet dataset of smooth per-episode trajectories."""
    rng = np.random.default_rng(seed)
    states, actions, episodes = [], [], []
    for e in range(N_EP):
        walk = np.cumsum(rng.normal(size=(T_PER_EP, DIM_S)) * 0.1, axis=0) + rng.normal(size=DIM_S)
        states.append(walk)
        # First DIM_A-1 action dims track the state (so deltas are small) but are not identical
        # to it — a commanded target differs from the measured position. Without that offset the
        # k=0 element of every chunk would be exactly 0.0, producing a point mass that no
        # histogram quantile can localize, which is a fixture artifact rather than real data.
        tracked = walk[:, : DIM_A - 1] + rng.normal(size=(T_PER_EP, DIM_A - 1)) * 0.02
        act = np.concatenate([tracked, rng.uniform(0, 1, size=(T_PER_EP, 1))], axis=1)
        actions.append(act)
        episodes.append(np.full(T_PER_EP, e))
    states, actions, episodes = map(np.concatenate, (states, actions, episodes))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.parquet"
    pq.write_table(
        pa.table(
            {
                "observation.state": [r.tolist() for r in states],
                "action": [r.tolist() for r in actions],
                "episode_index": episodes.tolist(),
            }
        ),
        path,
    )
    return path, states, actions


def _kwargs(path, **overrides):
    base = {
        "parquet_paths": [str(path)],
        "state_col": "observation.state",
        "action_col": "action",
        "state_index": None,
        "action_index": None,
        "delta_map": {i: i for i in range(DIM_A - 1)},
        "chunk_offsets": np.arange(HORIZON, dtype=np.float64),
        "strategy": "nearest",
        "episodes": None,
        "max_workers": 1,
    }
    base.update(overrides)
    return base


def _brute_force_deltas(states, actions, dim, offsets, delta_map):
    """Reference implementation: explicit per-frame loop with episode clipping."""
    out = []
    for e in range(N_EP):
        lo, hi = e * T_PER_EP, (e + 1) * T_PER_EP
        for t in range(lo, hi):
            idx = np.clip(t + offsets, lo, hi - 1).astype(int)
            vals = actions[idx, dim]
            if dim in delta_map:
                vals = vals - states[t, delta_map[dim]]
            out.append(vals)
    return np.concatenate(out)


class TestRunningStats:
    def test_quantiles_match_numpy_within_one_bin(self):
        """The 5000-bin histogram tracks exact quantiles to ~one bin width of the range."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100_000, 3)) * [1.0, 5.0, 0.1] + [0.0, -3.0, 2.0]
        acc = RunningStats()
        for block in np.array_split(x, 13):  # streamed, to exercise histogram rebinning
            acc.update(block)
        stats = acc.get_statistics()
        for q, name in ((0.01, "q01"), (0.99, "q99")):
            exact = np.quantile(x, q, axis=0)
            bin_width = (x.max(axis=0) - x.min(axis=0)) / 5000
            assert np.all(np.abs(stats[name] - exact) < 10 * bin_width)

    def test_mean_std_min_max_are_exact(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5_000, 2))
        acc = RunningStats()
        for block in np.array_split(x, 7):
            acc.update(block)
        stats = acc.get_statistics()
        np.testing.assert_allclose(stats["mean"], x.mean(0), atol=1e-5)
        np.testing.assert_allclose(stats["std"], x.std(0), atol=1e-5)
        np.testing.assert_allclose(stats["min"], x.min(0), atol=1e-6)
        np.testing.assert_allclose(stats["max"], x.max(0), atol=1e-6)
        assert stats["count"].tolist() == [5_000]

    def test_counts_every_element_of_a_chunk_axis(self):
        """A (frames, chunk, dim) update contributes frames*chunk samples, not frames."""
        acc = RunningStats()
        acc.update(np.zeros((10, 4, 2)))
        assert acc.get_statistics()["count"].tolist() == [40]

    def test_bounds_prevent_rebinning_error_on_a_widening_stream(self):
        """Fixed edges keep the estimate accurate when the range grows monotonically.

        Rebinning redistributes counts by their old left edges, so a stream whose observed range
        widens on every update — exactly what reading a dataset episode by episode looks like —
        smears the histogram badly. Passing the true range up front means the edges never move.
        Regression test: without `bounds` this data lands ~40 bins off.
        """
        rng = np.random.default_rng(0)
        x = np.sort(rng.normal(size=(50_000, 1)), axis=0)  # ascending => worst case
        exact = np.quantile(x, 0.01)
        bin_width = (x.max() - x.min()) / 5000

        unbounded = RunningStats()
        bounded = RunningStats(bounds=(x.min(axis=0), x.max(axis=0)))
        for block in np.array_split(x, 200):
            unbounded.update(block)
            bounded.update(block)

        bounded_err = abs(float(bounded.get_statistics()["q01"][0]) - exact)
        unbounded_err = abs(float(unbounded.get_statistics()["q01"][0]) - exact)
        assert bounded_err < 2 * bin_width
        assert unbounded_err > 10 * bin_width  # the failure mode `bounds` exists to avoid

    def test_without_bounds_matches_openpi_behavior(self):
        """No `bounds` => edges seeded from the first batch and adjusted on growth, as upstream."""
        acc = RunningStats()
        acc.update(np.array([[0.0], [1.0]]))
        first_edges = acc._bin_edges[0].copy()
        acc.update(np.array([[-5.0], [5.0]]))  # widen the range
        assert not np.array_equal(acc._bin_edges[0], first_edges)

    def test_too_few_samples_raises(self):
        acc = RunningStats()
        acc.update(np.zeros((1, 2)))
        with pytest.raises(ValueError, match="1 sample"):
            acc.get_statistics()

    def test_dimension_mismatch_raises(self):
        acc = RunningStats()
        acc.update(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="does not match"):
            acc.update(np.zeros((4, 3)))


class TestComputeDeltaActionStats:
    def test_matches_brute_force_reference(self, tmp_path):
        path, states, actions = _write_dataset(tmp_path)
        kw = _kwargs(path)
        got = compute_delta_action_stats(**kw)["actions"]
        offsets = kw["chunk_offsets"]
        for dim in range(DIM_A):
            ref = _brute_force_deltas(states, actions, dim, offsets, kw["delta_map"])
            # Mean/std are exact, so they pin that the sample SETS are identical.
            assert abs(float(got["mean"][dim]) - ref.mean()) < 1e-4
            assert abs(float(got["std"][dim]) - ref.std()) < 1e-4
            # Quantiles are histogram estimates. Two independent sources of slack: the bin width,
            # and how tightly this many samples pins the quantile at all (the spread of the
            # reference quantile across a +/-0.5% window around the target).
            bin_slack = 10 * (ref.max() - ref.min()) / 5000
            for q, name in ((0.01, "q01"), (0.99, "q99")):
                sample_slack = abs(np.quantile(ref, q + 0.005) - np.quantile(ref, q - 0.005))
                tol = max(bin_slack, sample_slack, 1e-3)
                assert abs(float(got[name][dim]) - np.quantile(ref, q)) < tol

    def test_unmapped_dim_keeps_absolute_scale(self, tmp_path):
        """One buffer can mix relative and absolute dims — the gripper case."""
        path, _, _ = _write_dataset(tmp_path)
        got = compute_delta_action_stats(**_kwargs(path))["actions"]
        gripper = DIM_A - 1
        # Absolute uniform[0,1] gripper: band stays near [0, 1].
        assert float(got["q01"][gripper]) > -0.1
        assert float(got["q99"][gripper]) < 1.1
        # Mapped dims are displacements, so their band straddles zero and is far tighter.
        assert float(got["q01"][0]) < 0 < float(got["q99"][0])
        assert (got["q99"][0] - got["q01"][0]) < (got["q99"][gripper] - got["q01"][gripper])

    def test_action_count_is_frames_times_horizon(self, tmp_path):
        """Every element of every chunk is a sample — the H-fold cost this module exists for."""
        path, _, _ = _write_dataset(tmp_path)
        stats = compute_delta_action_stats(**_kwargs(path))
        assert stats["actions"]["count"].tolist() == [N_EP * T_PER_EP * HORIZON]
        # State is per-frame, not chunked.
        assert stats["state"]["count"].tolist() == [N_EP * T_PER_EP]

    def test_episode_selection_is_honored(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        stats = compute_delta_action_stats(**_kwargs(path, episodes={0}))
        assert stats["actions"]["count"].tolist() == [T_PER_EP * HORIZON]

    def test_chunks_clip_at_episode_boundaries(self, tmp_path):
        """A chunk never reads across into the next episode; it repeats the last frame.

        Training clips and marks those rows padded rather than dropping them, so the stats must
        include them or they would describe a different sample set than the model sees.
        """
        path, states, actions = _write_dataset(tmp_path)
        # A horizon longer than an episode forces clipping on every anchor.
        offsets = np.arange(T_PER_EP + 10, dtype=np.float64)
        kw = _kwargs(path, chunk_offsets=offsets, episodes={1})
        got = compute_delta_action_stats(**kw)["actions"]
        ref = _brute_force_deltas(states, actions, 0, offsets, kw["delta_map"])
        ref = ref[T_PER_EP * len(offsets) : 2 * T_PER_EP * len(offsets)]  # episode 1 only
        assert abs(float(got["mean"][0]) - ref.mean()) < 1e-4

    def test_column_index_is_applied_before_the_delta(self, tmp_path):
        path, states, actions = _write_dataset(tmp_path)
        kw = _kwargs(path, action_index=[2, 0], state_index=[1, 0], delta_map={0: 1})
        got = compute_delta_action_stats(**kw)["actions"]
        # post-index action pos 0 == raw col 2; post-index state pos 1 == raw col 0.
        ref = _brute_force_deltas(states, actions, 2, kw["chunk_offsets"], {2: 0})
        assert abs(float(got["mean"][0]) - ref.mean()) < 1e-4

    def test_linear_and_nearest_strategies_both_run(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        offsets = np.arange(HORIZON, dtype=np.float64) + 0.5  # fractional -> strategies diverge
        near = compute_delta_action_stats(**_kwargs(path, chunk_offsets=offsets))["actions"]
        lin = compute_delta_action_stats(**_kwargs(path, chunk_offsets=offsets, strategy="linear"))["actions"]
        assert not np.allclose(near["mean"], lin["mean"])

    def test_unknown_strategy_raises(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        with pytest.raises(ValueError, match="vector_resample_strategy"):
            compute_delta_action_stats(**_kwargs(path, strategy="cubic"))

    def test_no_usable_files_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty, unreadable, or excluded"):
            compute_delta_action_stats(**_kwargs(tmp_path / "missing.parquet"))

    def test_multiprocess_matches_inline(self, tmp_path):
        """The pool path must agree with the inline path exactly."""
        path, _, _ = _write_dataset(tmp_path)
        inline = compute_delta_action_stats(**_kwargs(path, max_workers=1))["actions"]
        pooled = compute_delta_action_stats(**_kwargs(path, parquet_paths=[str(path)], max_workers=4))[
            "actions"
        ]
        np.testing.assert_allclose(inline["mean"], pooled["mean"])


class TestCacheKey:
    def _key(self, **overrides):
        base = {
            "state_index": None,
            "action_index": None,
            "delta_map": {0: 0},
            "chunk_offsets": [0.0, 1.0],
            "vector_resample_strategy": "nearest",
            "episodes": None,
            "excluded_episodes": None,
            "fps": 20.0,
            "revision": None,
        }
        base.update(overrides)
        return delta_stats_cache_key(**base)

    def test_stable_across_calls(self):
        assert self._key() == self._key()

    def test_insensitive_to_dict_and_list_ordering(self):
        """Reordering the map must not fragment the cache."""
        a = self._key(delta_map={0: 1, 2: 3})
        b = self._key(delta_map={2: 3, 0: 1})
        assert a == b

    @pytest.mark.parametrize(
        "override",
        [
            {"chunk_offsets": [0.0, 1.0, 2.0]},
            {"delta_map": {0: 1}},
            {"state_index": [0, 1]},
            {"action_index": [1, 0]},
            {"vector_resample_strategy": "linear"},
            {"episodes": [0, 1]},
            {"excluded_episodes": [3]},
            {"fps": 30.0},
            {"revision": "v2"},
        ],
    )
    def test_every_input_that_changes_the_stats_changes_the_key(self, override):
        assert self._key(**override) != self._key()


class TestLoadOrCompute:
    def test_computes_then_serves_from_cache(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        kw = _kwargs(path)
        key = "testkey_hit"
        first = load_or_compute_delta_action_stats(root=tmp_path, cache_key=key, compute_kwargs=kw)
        assert (tmp_path / CACHE_SUBDIR / f"{key}.json").is_file()
        # Point the recompute at a nonexistent file: a cache miss would now raise, so returning
        # matching stats proves the second call never recomputed.
        second = load_or_compute_delta_action_stats(
            root=tmp_path, cache_key=key, compute_kwargs=_kwargs(tmp_path / "gone.parquet")
        )
        np.testing.assert_allclose(first["actions"]["q01"], second["actions"]["q01"])
        np.testing.assert_allclose(first["state"]["mean"], second["state"]["mean"])

    def test_distinct_keys_do_not_collide(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        a = load_or_compute_delta_action_stats(root=tmp_path, cache_key="key_a", compute_kwargs=_kwargs(path))
        b = load_or_compute_delta_action_stats(
            root=tmp_path,
            cache_key="key_b",
            compute_kwargs=_kwargs(path, chunk_offsets=np.arange(2.0)),
        )
        assert not np.allclose(a["actions"]["count"], b["actions"]["count"])

    def test_corrupt_cache_is_recomputed(self, tmp_path):
        """A truncated or half-written file must not poison the run."""
        path, _, _ = _write_dataset(tmp_path)
        key = "testkey_corrupt"
        cache_file = tmp_path / CACHE_SUBDIR / f"{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{not valid json")
        stats = load_or_compute_delta_action_stats(root=tmp_path, cache_key=key, compute_kwargs=_kwargs(path))
        assert stats["actions"]["count"].tolist() == [N_EP * T_PER_EP * HORIZON]

    def test_stale_version_is_ignored(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        key = "testkey_stale"
        cache_file = tmp_path / CACHE_SUBDIR / f"{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text('{"version": 0, "stats": {"actions": {"mean": [9.0]}}}')
        stats = load_or_compute_delta_action_stats(root=tmp_path, cache_key=key, compute_kwargs=_kwargs(path))
        assert float(stats["actions"]["mean"][0]) != 9.0

    def test_readonly_root_still_returns_stats(self, tmp_path, monkeypatch):
        """An unwritable dataset root degrades to recompute-every-load, never a crash."""
        path, _, _ = _write_dataset(tmp_path)

        def _boom(*_a, **_k):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr("opentau.datasets.delta_action_stats._atomic_write_text", _boom)
        stats = load_or_compute_delta_action_stats(
            root=tmp_path, cache_key="testkey_readonly", compute_kwargs=_kwargs(path)
        )
        assert stats["actions"]["count"].tolist() == [N_EP * T_PER_EP * HORIZON]
