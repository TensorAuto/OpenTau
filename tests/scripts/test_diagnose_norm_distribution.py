#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the streaming normalized-distribution diagnostic.

Covers the pure, CPU-only pieces that need no real dataset: the histogram bin
layout, the ``DimStats`` accumulator (vs brute-force numpy), merge
associativity, NaN/Inf handling, and the normalization formulas. A parquet
round-trip exercises the worker end-to-end without opentau metadata.
"""

import numpy as np
import pytest

from opentau.scripts.diagnose_norm_distribution import (
    BIN_EDGES,
    EPS,
    N_BINS,
    DimStats,
    _normalize,
    make_bin_edges,
)


class TestBinEdges:
    def test_strictly_increasing(self):
        assert np.all(np.diff(BIN_EDGES) > 0)

    def test_symmetric_about_zero(self):
        finite = BIN_EDGES[np.isfinite(BIN_EDGES)]
        np.testing.assert_allclose(finite, -finite[::-1], atol=1e-9)

    def test_open_catchalls_and_count(self):
        assert BIN_EDGES[0] == -np.inf and BIN_EDGES[-1] == np.inf
        assert len(BIN_EDGES) == N_BINS + 1
        assert make_bin_edges().shape == BIN_EDGES.shape


class TestNormalize:
    def test_mean_std(self):
        x = np.array([[2.0, 4.0]])
        stat = {"mean": np.array([1.0, 2.0]), "std": np.array([1.0, 2.0])}
        z = _normalize(x, stat, "MEAN_STD")
        np.testing.assert_allclose(z, [[1.0 / (1 + EPS), 2.0 / (2 + EPS)]])

    def test_min_max(self):
        x = np.array([[0.5]])
        stat = {"min": np.array([0.0]), "max": np.array([1.0])}
        z = _normalize(x, stat, "MIN_MAX")
        np.testing.assert_allclose(z, [[0.5 / (1 + EPS) * 2 - 1]], atol=1e-6)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            _normalize(np.zeros((1, 1)), {}, "NOPE")


def _brute(x, z, k):
    finite = np.isfinite(x) & np.isfinite(z)
    xf = np.where(finite, x, np.nan)
    zf = np.where(finite, z, np.nan)
    return {
        "count": finite.sum(0),
        "mean": np.nanmean(xf, 0),
        "xmin": np.nanmin(xf, 0),
        "xmax": np.nanmax(xf, 0),
        "zmean": np.nanmean(zf, 0),
        "n_beyond10": (np.abs(zf) > 10).sum(0),
        "n_nonfinite": (~finite).sum(0),
    }


class TestDimStats:
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 4)) * rng.uniform(0.1, 5, size=4)
        stat = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}
        z = _normalize(x, stat, "MEAN_STD")
        acc = DimStats(4)
        acc.update(x, z)
        b = _brute(x, z, 10)
        np.testing.assert_array_equal(acc.count, b["count"])
        np.testing.assert_allclose(acc.data_mean(), b["mean"], rtol=1e-10)
        np.testing.assert_allclose(acc.xmin, b["xmin"])
        np.testing.assert_allclose(acc.xmax, b["xmax"])
        np.testing.assert_allclose(acc.z_mean(), b["zmean"], rtol=1e-10)
        # data_std vs numpy population std
        np.testing.assert_allclose(acc.data_std(), x.std(0), rtol=1e-8)
        # every finite sample lands in exactly one bin
        np.testing.assert_array_equal(acc.hist.sum(1), b["count"])
        np.testing.assert_array_equal(acc.n_beyond[:, 2], b["n_beyond10"])

    def test_merge_is_associative(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 3))
        stat = {"mean": np.zeros(3), "std": np.ones(3), "min": np.full(3, -3.0), "max": np.full(3, 3.0)}
        z = _normalize(x, stat, "MEAN_STD")
        whole = DimStats(3)
        whole.update(x, z)
        a, b = DimStats(3), DimStats(3)
        a.update(x[:111], z[:111])
        b.update(x[111:], z[111:])
        merged = a.merge(b)
        np.testing.assert_array_equal(whole.count, merged.count)
        np.testing.assert_array_equal(whole.hist, merged.hist)
        np.testing.assert_allclose(whole.data_mean(), merged.data_mean(), rtol=1e-12)
        np.testing.assert_allclose(whole.xmax, merged.xmax)
        np.testing.assert_array_equal(whole.n_beyond, merged.n_beyond)

    def test_nonfinite_excluded(self):
        x = np.array([[1.0, np.nan], [2.0, np.inf], [3.0, 4.0]])
        stat = {"mean": np.zeros(2), "std": np.ones(2), "min": np.zeros(2), "max": np.ones(2)}
        z = _normalize(x, stat, "MEAN_STD")
        acc = DimStats(2)
        acc.update(x, z)
        np.testing.assert_array_equal(acc.count, [3, 1])
        np.testing.assert_array_equal(acc.n_nonfinite, [0, 2])
        np.testing.assert_allclose(acc.data_mean(), [2.0, 4.0])

    def test_dim_mismatch_raises(self):
        acc = DimStats(2)
        with pytest.raises(ValueError):
            acc.update(np.zeros((5, 3)), np.zeros((5, 3)))


@pytest.mark.slow
def test_worker_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from opentau.scripts.diagnose_norm_distribution import _process_dataset

    rng = np.random.default_rng(2)
    state = rng.normal(size=(40, 3))
    action = rng.normal(size=(40, 2))
    path = tmp_path / "episode_000000.parquet"
    pq.write_table(pa.table({"observation.state": list(state), "action": list(action)}), path)

    sstat = {"mean": state.mean(0), "std": state.std(0), "min": state.min(0), "max": state.max(0)}
    astat = {"mean": action.mean(0), "std": action.std(0), "min": action.min(0), "max": action.max(0)}
    task = {
        "head_key": "franka::joint",
        "parquet_paths": [str(path)],
        "frame_stride": 1,
        "norm_mode": "MEAN_STD",
        "state": ("observation.state", 3, sstat),
        "actions": ("action", 2, astat),
    }
    out = _process_dataset(task)
    assert set(out) == {("franka::joint", "state"), ("franka::joint", "actions")}
    sacc = out[("franka::joint", "state")]
    assert int(sacc.count.sum()) == 40 * 3
    # normalizing by the data's own mean/std => z ~ mean 0, std 1
    np.testing.assert_allclose(sacc.z_mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(sacc.z_std(), 1.0, atol=1e-6)


@pytest.mark.slow
def test_plot_writes_histograms_and_summary(tmp_path):
    """The figures are one per-head histogram grid + a global z_std summary (no more
    heatmap). Smoke-test that both land on disk for a synthetic one-head mixture."""
    pytest.importorskip("matplotlib")

    from opentau.scripts.diagnose_norm_distribution import _finalize_report, _plot

    rng = np.random.default_rng(3)
    merged, meta = {}, {}
    for feat, dim in (("state", 3), ("actions", 2)):
        x = rng.normal(size=(200, dim))
        stat = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}
        acc = DimStats(dim)
        acc.update(x, _normalize(x, stat, "MEAN_STD"))
        merged[("franka::ee", feat)] = acc
        meta[feat] = stat
    head_info = {"franka::ee": {"robot_type": "franka", "control_mode": "ee", "meta": meta}}

    rows = _finalize_report(merged, head_info, ["franka::ee"])["rows"]
    _plot(merged, head_info, ["franka::ee"], rows, tmp_path)

    assert (tmp_path / "head__franka__ee.png").exists()
    assert (tmp_path / "summary_zstd_distribution.png").exists()
