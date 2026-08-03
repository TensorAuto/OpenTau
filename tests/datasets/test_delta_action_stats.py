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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from opentau.datasets.delta_action_stats import (
    CACHE_SUBDIR,
    RunningStats,
    _allocate_row_budgets,
    _merge_running,
    _select_anchor_rows,
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


class TestRowCap:
    """`max_rows` bounds the O(frames x horizon) pass without biasing the sample."""

    def test_cap_is_a_hard_bound_on_anchor_rows(self):
        """Never more than the cap, for every span layout and every cap."""
        layouts = [
            [(0, 100)],
            [(0, 10), (10, 20), (20, 30)],
            [(0, 1), (1, 2), (2, 3), (3, 400)],
            [(5, 7), (100, 101)],
        ]
        for spans in layouts:
            total = sum(hi - lo for lo, hi in spans)
            for cap in range(1, total + 3):
                kept = sum(len(rows) for rows, _, _ in _select_anchor_rows(spans, cap))
                assert kept <= cap, f"spans={spans} cap={cap} kept={kept}"

    def test_uncapped_keeps_every_row(self):
        spans = [(0, 4), (4, 9)]
        assert [rows.tolist() for rows, _, _ in _select_anchor_rows(spans, None)] == [
            [0, 1, 2, 3],
            [4, 5, 6, 7, 8],
        ]

    def test_cap_at_or_above_the_row_count_keeps_every_row(self):
        spans = [(0, 4), (4, 9)]
        for cap in (9, 10, 10_000):
            selected = _select_anchor_rows(spans, cap)
            assert sum(len(rows) for rows, _, _ in selected) == 9

    def test_stride_phase_continues_across_episode_boundaries(self):
        """Anchors must not restart at frame 0 of every episode.

        A per-span restart would anchor every episode at its first frame, over-sampling the
        episode-start transient (which `_gather_chunks` already treats specially by clipping).
        """
        spans = [(0, 10), (10, 20), (20, 30)]
        selected = _select_anchor_rows(spans, 10)  # stride 3
        assert [rows.tolist() for rows, _, _ in selected] == [
            [0, 3, 6, 9],
            [12, 15, 18],
            [21, 24, 27],
        ]
        starts = [int(rows[0]) - lo for rows, lo, _ in selected]
        assert len(set(starts)) > 1, "every episode was anchored at the same in-episode phase"

    def test_true_episode_bounds_survive_subsampling(self):
        """Clipping still uses the real episode extent, not the sampled rows' extent."""
        selected = _select_anchor_rows([(0, 100), (100, 200)], 4)
        assert [(lo, hi) for _, lo, hi in selected] == [(0, 100), (100, 200)]

    def test_cap_bounds_the_accumulated_sample_count(self, tmp_path):
        path, _, _ = _write_dataset(tmp_path)
        cap = 200
        stats = compute_delta_action_stats(**_kwargs(path, max_rows=cap))
        assert stats["state"]["count"].tolist()[0] <= cap
        assert stats["actions"]["count"].tolist()[0] <= cap * HORIZON
        # And it really did bind — the uncapped pass sees far more.
        assert stats["state"]["count"].tolist()[0] < N_EP * T_PER_EP

    def test_cap_above_the_dataset_size_is_a_no_op(self, tmp_path):
        """A cap the dataset never reaches must produce bit-identical stats to no cap."""
        path, _, _ = _write_dataset(tmp_path)
        uncapped = compute_delta_action_stats(**_kwargs(path))["actions"]
        capped = compute_delta_action_stats(**_kwargs(path, max_rows=10 * N_EP * T_PER_EP))["actions"]
        for stat in ("mean", "std", "min", "max", "q01", "q99", "count"):
            np.testing.assert_array_equal(uncapped[stat], capped[stat])

    def test_cap_samples_the_whole_dataset_not_a_prefix(self, tmp_path):
        """The tail of the dataset must still reach the stats.

        Truncating to the first N rows is the obvious wrong implementation and would be invisible
        in a mean/std check on smooth data. Here the LAST episode is shifted far away, so its
        contribution shows up in `max` — a prefix sample would miss it entirely.
        """
        rng = np.random.default_rng(7)
        states, actions, episodes = [], [], []
        for e in range(N_EP):
            walk = np.cumsum(rng.normal(size=(T_PER_EP, DIM_S)) * 0.1, axis=0)
            act = np.concatenate([walk[:, : DIM_A - 1], rng.uniform(0, 1, size=(T_PER_EP, 1))], axis=1)
            if e == N_EP - 1:
                act = act + 50.0  # only the final episode reaches this scale
            states.append(walk)
            actions.append(act)
            episodes.append(np.full(T_PER_EP, e))
        states, actions, episodes = map(np.concatenate, (states, actions, episodes))
        path = tmp_path / "data.parquet"
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
        got = compute_delta_action_stats(**_kwargs(path, max_rows=40))["actions"]
        assert float(got["max"][0]) > 40.0

    def test_capped_mean_tracks_the_uncapped_mean(self, tmp_path):
        """A uniform stride is an unbiased sample, so a generous cap barely moves the mean."""
        path, _, _ = _write_dataset(tmp_path)
        full = compute_delta_action_stats(**_kwargs(path))["actions"]
        capped = compute_delta_action_stats(**_kwargs(path, max_rows=N_EP * T_PER_EP // 2))["actions"]
        for dim in range(DIM_A):
            spread = float(full["std"][dim])
            assert abs(float(capped["mean"][dim]) - float(full["mean"][dim])) < 0.25 * spread

    def test_every_episode_still_contributes_under_a_loose_cap(self):
        """Striding across spans keeps whole episodes from dropping out."""
        selected = _select_anchor_rows([(e * T_PER_EP, (e + 1) * T_PER_EP) for e in range(N_EP)], 100)
        assert len(selected) == N_EP

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_invalid_cap_raises(self, tmp_path, bad):
        """`True` is in the list because `bool` is an `int` subclass.

        A bare `>= 1` check lets `max_rows=True` through as 1, which does not raise — it caps the
        pass at one anchor row and silently returns stats fitted to two samples (a measured std
        of 0.13 where the true value is 0.99). The config layer rejects bools, but this function
        is exported and the module is documented as usable directly, so guard it here too.
        """
        path, _, _ = _write_dataset(tmp_path)
        with pytest.raises(ValueError, match="max_rows must be >= 1"):
            compute_delta_action_stats(**_kwargs(path, max_rows=bad))

    def test_multiprocess_matches_inline_under_a_cap(self, tmp_path):
        """The cap must be applied identically on both execution paths."""
        path, _, _ = _write_dataset(tmp_path)
        inline = compute_delta_action_stats(**_kwargs(path, max_rows=137, max_workers=1))["actions"]
        pooled = compute_delta_action_stats(**_kwargs(path, max_rows=137, max_workers=4))["actions"]
        np.testing.assert_allclose(inline["mean"], pooled["mean"])
        np.testing.assert_array_equal(inline["count"], pooled["count"])


class TestRowBudgetAllocation:
    """Splitting a dataset-wide cap across a multi-file dataset."""

    def _write(self, path, n_rows, n_ep=2):
        rng = np.random.default_rng(0)
        episodes = np.repeat(np.arange(n_ep), n_rows // n_ep)
        n = len(episodes)
        pq.write_table(
            pa.table(
                {
                    "observation.state": [r.tolist() for r in rng.normal(size=(n, DIM_S))],
                    "action": [r.tolist() for r in rng.normal(size=(n, DIM_A))],
                    "episode_index": episodes.tolist(),
                }
            ),
            path,
        )
        return str(path)

    def test_uncapped_gives_every_file_no_budget(self, tmp_path):
        paths = [self._write(tmp_path / f"{i}.parquet", 100) for i in range(3)]
        assert _allocate_row_budgets(paths, None) == [None, None, None]

    def test_cap_larger_than_the_dataset_does_not_bind(self, tmp_path):
        paths = [self._write(tmp_path / f"{i}.parquet", 100) for i in range(3)]
        assert _allocate_row_budgets(paths, 10_000) == [None, None, None]

    def test_budgets_track_file_sizes(self, tmp_path):
        """Proportional, not even — `_merge_running` weights each file by what it contributes,
        so an even split would over-weight a small shard in the pooled mean."""
        big = self._write(tmp_path / "big.parquet", 900)
        small = self._write(tmp_path / "small.parquet", 100)
        budgets = _allocate_row_budgets([big, small], 100)
        assert sum(budgets) <= 100
        assert budgets[0] == 90 and budgets[1] == 10

    def test_unreadable_file_does_not_starve_the_rest(self, tmp_path):
        """A footer we can't read is sized at the mean, so the readable files keep sane budgets."""
        good = self._write(tmp_path / "good.parquet", 1000)
        missing = str(tmp_path / "nope.parquet")
        budgets = _allocate_row_budgets([good, missing], 100)
        assert budgets[0] >= 40
        assert all(b >= 2 for b in budgets)

    def test_budgets_track_selected_rows_under_an_episode_filter(self, tmp_path):
        """Sizes count the rows the run will *use*, not every row in the file.

        Two same-size files, but the episode filter keeps 1 of 10 episodes in the first and all
        10 in the second. Sizing by the parquet footer would call them equal and hand each half
        the cap, over-weighting the barely-selected file 10x in the pooled merge.
        """
        a = self._write(tmp_path / "a.parquet", 1000, n_ep=10)  # episodes 0-9
        b = self._write(tmp_path / "b.parquet", 1000, n_ep=10)  # episodes 0-9
        # `episodes` is matched on the value in the column, so this keeps 1/10 of each file...
        budgets = _allocate_row_budgets([a, b], 110, episodes={0})
        assert budgets == [55, 55], "equal selections must still split equally"
        # ...whereas an asymmetric selection has to shift the split. Rebuild `b` with a single
        # episode index so every one of its rows is selected.
        b_all = self._write(tmp_path / "b_all.parquet", 1000, n_ep=1)  # all rows are episode 0
        budgets = _allocate_row_budgets([a, b_all], 110, episodes={0})
        assert sum(budgets) <= 110
        assert budgets[1] > 5 * budgets[0], f"expected a ~10:1 split by selected rows, got {budgets}"

    def test_every_file_clears_the_two_sample_minimum(self, tmp_path):
        """A tiny cap must not silently drop whole shards below `RunningStats`' minimum."""
        paths = [self._write(tmp_path / f"{i}.parquet", 200) for i in range(8)]
        assert all(b >= 2 for b in _allocate_row_budgets(paths, 4))

    def test_multi_file_dataset_respects_the_cap_end_to_end(self, tmp_path):
        paths = [self._write(tmp_path / f"{i}.parquet", 600) for i in range(3)]
        stats = compute_delta_action_stats(**_kwargs(paths[0], parquet_paths=paths, max_rows=300))
        assert stats["state"]["count"].tolist()[0] <= 300

    def _write_at(self, path, n_rows, centre):
        """A file whose rows all sit near `centre`, so its share of the pool is visible."""
        rng = np.random.default_rng(int(centre))
        eps = np.repeat(np.arange(2), n_rows // 2)
        n = len(eps)
        pq.write_table(
            pa.table(
                {
                    "observation.state": [r.tolist() for r in rng.normal(size=(n, DIM_S)) * 0.1 + centre],
                    "action": [r.tolist() for r in rng.normal(size=(n, DIM_A)) * 0.1 + centre],
                    "episode_index": eps.tolist(),
                }
            ),
            path,
        )
        return str(path)

    def test_capping_preserves_the_ratio_between_files(self, tmp_path):
        """A 9:1 pair of files must still pool 9:1 after the cap.

        `_merge_running` pools the per-file results weighted by the counts they contribute, so
        the budget split *is* the mixture ratio within a dataset. An even split — the obvious
        alternative — would re-weight a 9:1 dataset to 1:1 and drag the pooled mean most of the
        way to the small file's distribution.
        """
        big = self._write_at(tmp_path / "big.parquet", 9000, 0.0)
        small = self._write_at(tmp_path / "small.parquet", 1000, 10.0)
        # No delta map and a 1-step horizon: the action stats are then just the raw rows, so the
        # pooled mean reads directly as each file's share of the sample.
        kw = {"delta_map": {}, "chunk_offsets": np.array([0.0])}

        full = compute_delta_action_stats(**_kwargs(big, parquet_paths=[big, small], **kw))["actions"]
        capped = compute_delta_action_stats(**_kwargs(big, parquet_paths=[big, small], max_rows=500, **kw))[
            "actions"
        ]

        # True pooled mean is 0.9 * 0 + 0.1 * 10 == 1.0.
        assert abs(float(full["mean"][0]) - 1.0) < 0.1
        assert abs(float(capped["mean"][0]) - float(full["mean"][0])) < 0.1

        # Pin the counterfactual: had the budgets been split evenly, the pool would land ~5.0.
        even = _merge_running(
            [
                compute_delta_action_stats(**_kwargs(big, parquet_paths=[big], max_rows=250, **kw)),
                compute_delta_action_stats(**_kwargs(small, parquet_paths=[small], max_rows=250, **kw)),
            ]
        )
        assert abs(float(even["actions"]["mean"][0]) - 1.0) > 3.0


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
            {"max_rows": 100_000},
        ],
    )
    def test_every_input_that_changes_the_stats_changes_the_key(self, override):
        assert self._key(**override) != self._key()

    def test_an_absent_cap_keeps_the_pre_cap_digest(self):
        """Uncapped configs must keep the key they had before `max_rows` existed.

        Their cache files hold the expensive all-rows result; folding an always-present
        `max_rows: null` into the payload would invalidate every one of them on upgrade.
        """
        assert self._key(max_rows=None) == self._key()
        # Pinned against the digest this input produced before `max_rows` existed. Any future
        # payload change that would orphan existing cache files has to break this line first.
        assert self._key() == "f4c5d10ec6db"

    def test_distinct_caps_do_not_collide(self):
        assert self._key(max_rows=1000) != self._key(max_rows=2000)


class TestFactoryWiring:
    """`DatasetMixtureConfig.delta_stats_max_rows` has to reach BOTH consumers.

    Threading it into the compute but not the cache key would serve a previously-cached uncapped
    (or differently-capped) result forever; threading it into the key but not the compute would
    recompute the identical numbers under a new key. Neither failure raises.
    """

    def _fake_dataset(self, root, action_col="action"):
        """A stand-in dataset whose delta-timestamp keys match its own name map.

        ``resolve_delta_timestamps`` keys its output by RAW on-disk column names, so a fake
        whose ``_get_name_map`` reports ``action_col`` must key ``delta_timestamps_params``
        by ``action_col`` too. Keying it by the ``"actions"`` alias instead would model no
        real dataset and would hide a wrong lookup in the code under test.

        Args:
            root: Dataset root the fake metadata should report.
            action_col: Raw on-disk name of the action column.

        Returns:
            A ``SimpleNamespace`` exposing the attributes ``_compute_or_load_delta_stats`` reads.
        """
        meta = SimpleNamespace(
            root=root,
            episodes=[0, 1],
            get_data_file_path=lambda ep: Path("data.parquet"),
        )
        return SimpleNamespace(
            meta=meta,
            episodes=None,
            delta_timestamps_params=[{action_col: [0.0, 0.05]}],
            fps=20.0,
            vector_resample_strategy="nearest",
            delta_action_state_map={0: 0},
            _get_name_map=lambda: {"state": "observation.state", "actions": action_col},
        )

    def _run(self, tmp_path, monkeypatch, max_rows, action_col="action"):
        from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
        from opentau.datasets import factory

        seen = {}

        def _capture(*, root, cache_key, compute_kwargs):
            seen.update(cache_key=cache_key, compute_kwargs=compute_kwargs)
            return {"actions": {}, "state": {}}

        monkeypatch.setattr(factory, "load_or_compute_delta_action_stats", _capture)
        factory._compute_or_load_delta_stats(
            self._fake_dataset(tmp_path, action_col=action_col),
            DatasetConfig(repo_id="fake/ds", use_delta_joint_actions=True, delta_action_state_map={0: 0}),
            SimpleNamespace(
                num_workers=2, dataset_mixture=DatasetMixtureConfig(delta_stats_max_rows=max_rows)
            ),
        )
        return seen

    def test_cap_reaches_the_compute(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, 1234)["compute_kwargs"]["max_rows"] == 1234

    def test_cap_reaches_the_cache_key(self, tmp_path, monkeypatch):
        capped = self._run(tmp_path, monkeypatch, 1234)["cache_key"]
        other = self._run(tmp_path, monkeypatch, 5678)["cache_key"]
        uncapped = self._run(tmp_path, monkeypatch, None)["cache_key"]
        assert len({capped, other, uncapped}) == 3

    def test_default_is_uncapped(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, monkeypatch, None)["compute_kwargs"]["max_rows"] is None

    # `delta_timestamps_params` is keyed by the dataset's RAW action column, so the horizon
    # lookup must go through the name map. Hard-coding the `"actions"` alias raised
    # `KeyError: 'actions'` at dataset build for every delta run on a repo whose column is
    # named `action` — which is the LeRobot standard, and most of the registered mappings.
    @pytest.mark.parametrize("action_col", ["action", "actions"])
    def test_action_horizon_is_read_under_the_raw_column_name(self, tmp_path, monkeypatch, action_col):
        chunk_offsets = self._run(tmp_path, monkeypatch, None, action_col=action_col)["compute_kwargs"][
            "chunk_offsets"
        ]
        # [0.0s, 0.05s] at 20 fps -> frame offsets [0, 1].
        np.testing.assert_allclose(chunk_offsets, [0.0, 1.0])

    def test_raw_column_name_does_not_change_the_cache_key(self, tmp_path, monkeypatch):
        """Same horizon, different on-disk spelling -> same stats, so the same key.

        The key is built from `chunk_offsets`, not the column name; a rename that reached the
        key would orphan every cached result for no numerical reason.
        """
        assert (
            self._run(tmp_path, monkeypatch, None, action_col="action")["cache_key"]
            == self._run(tmp_path, monkeypatch, None, action_col="actions")["cache_key"]
        )


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

    def test_readonly_root_multi_rank_fails_fast_instead_of_hanging(self, tmp_path, monkeypatch):
        """A read-only root in a multi-rank run must raise, not deadlock.

        Rank 0's in-memory fallback would return while the other ranks poll forever for a file
        that never appears; rank 0's barrier then trips the NCCL watchdog. Every rank sees the
        same read-only filesystem, so the fail-fast is uniform.
        """
        path, _, _ = _write_dataset(tmp_path)

        class _FakeAcc:
            num_processes = 4
            is_main_process = True

            def wait_for_everyone(self):
                pass

        monkeypatch.setattr("opentau.utils.accelerate_utils.get_proc_accelerator", lambda: _FakeAcc())
        monkeypatch.setattr("opentau.datasets.delta_action_stats._cache_dir_writable", lambda _d: False)
        with pytest.raises(RuntimeError, match="not writable.*rank"):
            load_or_compute_delta_action_stats(
                root=tmp_path, cache_key="testkey_ro_dist", compute_kwargs=_kwargs(path)
            )
