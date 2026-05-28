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

"""Tests for ``opentau.scripts.fit_fast_tokenizer`` helpers.

End-to-end ``main()`` runs are slow (build mixture, drain dataloader,
fit BPE), so this file targets the small helpers whose behaviour can
flip silently on a config edit -- especially the mixed-frequency
``action_freq=None`` branch added alongside the ``feat(datasets):
mixed-frequency training`` work, which the manual sampler must now
respect by substituting each dataset's native fps (same convention as
``factory.resolve_delta_timestamps``).
"""

from __future__ import annotations

import numpy as np

from opentau.configs.default import DatasetConfig
from opentau.scripts.fit_fast_tokenizer import (
    _normalize_chunks,
    _normalize_chunks_per_head,
    _pool_per_head_stats,
    _sample_chunks_for_dataset_manual,
)
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID


class TestSampleChunksManualActionFreq:
    """``_sample_chunks_for_dataset_manual`` accepts ``action_freq=None``.

    Before mixed-frequency training landed, the manual sampler required
    a positive float ``action_freq``. ``main`` called ``float(...)`` on
    the mixture-level value, which crashes on ``None``. The fix
    substitutes each dataset's native fps inside the worker so the
    chunk is ``chunk_size`` consecutive native frames -- matching what
    ``factory.resolve_delta_timestamps`` does at training time.
    """

    def _run(self, root, action_freq, n_chunks=8, chunk_size=10, seed=0):
        cfg = DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(root))
        return _sample_chunks_for_dataset_manual((0, cfg, n_chunks, chunk_size, action_freq, seed))

    def test_none_action_freq_matches_explicit_native_fps(self, tmp_path, lerobot_dataset_factory):
        """``action_freq=None`` produces the same chunks as ``action_freq=native_fps``.

        The substitution is the *only* difference between the two call
        styles -- everything downstream of ``target_dt = 1.0 / action_freq``
        is identical -- so on a single 30 fps dataset, with the same seed,
        the two paths must yield byte-equivalent chunks. Re-using one root
        (the worker is read-only on the dataset) keeps the comparison from
        being polluted by any non-determinism in the factory's synthetic
        data generation.
        """
        # Materialize a 30 fps synthetic dataset on disk via the shared
        # fixture; the worker later constructs a fresh ``LeRobotDatasetMetadata``
        # from ``root`` and reads the parquet directly.
        root = tmp_path / "ds"
        lerobot_dataset_factory(root=root, total_episodes=3, total_frames=150)

        idx_none, chunks_none, err_none = self._run(root, action_freq=None)
        assert err_none is None, f"action_freq=None path errored: {err_none}"
        assert idx_none == 0
        assert len(chunks_none) == 8, "expected n_chunks=8 chunks back"
        for c in chunks_none:
            assert c.shape == (10, 6), f"unexpected chunk shape {c.shape}"

        _, chunks_native, err_native = self._run(root, action_freq=float(DEFAULT_FPS))
        assert err_native is None
        assert len(chunks_native) == len(chunks_none)
        # ``assert_array_equal`` (not ``assert_allclose``): empirically, with
        # both calls hitting the same parquet and the same rng-seeded episode /
        # start picks, the interp's target timestamps land on native frame
        # boundaries closely enough that ``scipy.interpolate.interp1d`` returns
        # bit-equal output. A future divergence would surface here immediately.
        for c_none, c_native in zip(chunks_none, chunks_native, strict=True):
            np.testing.assert_array_equal(c_none, c_native)


class TestPoolPerHeadStats:
    """``_pool_per_head_stats`` pools per-dataset action stats per norm key.

    These cases pin the contract that ``_aggregate_stats_per_head`` relies on
    -- the script normalizes each chunk with the *pooled* head's
    (min, max), and that pool MUST match what the training policy's
    ``DatasetMixtureMetadata._build_norm_heads`` computes from the same raw
    stats. If this drifts the fit normalization stops matching the
    training-time normalization and the BPE chunk-length distribution at
    inference will diverge from what the fit script saw -- exactly the
    failure mode that motivated this code path.
    """

    def test_two_datasets_one_head_pool_min_max(self):
        """Two datasets sharing a (rt, cm) get the elementwise nanmin/nanmax."""
        action_dim = 4
        raw_min = [
            np.array([-1.0, -2.0, -3.0, np.nan], dtype=np.float64),
            np.array([-0.5, -3.0, -2.0, -1.0], dtype=np.float64),
        ]
        raw_max = [
            np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64),
            np.array([2.0, 1.5, 4.0, 1.0], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotA::ee"]

        per_ds_min, per_ds_max, per_head_min = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)

        # One head, two datasets -- both datasets get the same pooled (min, max).
        assert len(per_head_min) == 1
        assert "robotA::ee" in per_head_min
        np.testing.assert_array_equal(per_ds_min[0], per_ds_min[1])
        np.testing.assert_array_equal(per_ds_max[0], per_ds_max[1])
        # Pooled values: elementwise nanmin / nanmax across the two datasets.
        np.testing.assert_array_equal(per_ds_min[0], np.array([-1.0, -3.0, -3.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 2.0, 4.0, 1.0], dtype=np.float32))

    def test_distinct_heads_kept_separate(self):
        """Different (rt, cm) pairs do NOT pool, even if dims overlap."""
        action_dim = 3
        raw_min = [
            np.array([-1.0, -1.0, -1.0], dtype=np.float64),
            np.array([-10.0, -10.0, -10.0], dtype=np.float64),
        ]
        raw_max = [
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([10.0, 10.0, 10.0], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotB::joint"]

        per_ds_min, per_ds_max, per_head_min = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)

        assert len(per_head_min) == 2
        # Each dataset keeps its own stats verbatim (singleton heads).
        np.testing.assert_array_equal(per_ds_min[0], np.array([-1.0, -1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_min[1], np.array([-10.0, -10.0, -10.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([1.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[1], np.array([10.0, 10.0, 10.0], dtype=np.float32))

    def test_failed_load_falls_back_to_minus_one_one(self):
        """A dataset whose stats failed to load gets the [-1, 1] sentinel."""
        action_dim = 2
        raw_min = [np.array([-2.0, -2.0], dtype=np.float64), None]
        raw_max = [np.array([2.0, 2.0], dtype=np.float64), None]
        per_ds_key = ["robotA::ee", None]

        per_ds_min, per_ds_max, per_head_min = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)

        assert len(per_head_min) == 1
        np.testing.assert_array_equal(per_ds_min[0], np.array([-2.0, -2.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_min[1], np.array([-1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[1], np.array([1.0, 1.0], dtype=np.float32))

    def test_all_nan_dim_filled_with_minus_one_one(self):
        """A dim no head member has stats for gets the [-1, 1] fallback."""
        action_dim = 3
        # Two datasets share a head; both are NaN on dim 2 (e.g. zero-pad slot).
        raw_min = [
            np.array([-1.0, -1.0, np.nan], dtype=np.float64),
            np.array([-2.0, -2.0, np.nan], dtype=np.float64),
        ]
        raw_max = [
            np.array([1.0, 1.0, np.nan], dtype=np.float64),
            np.array([2.0, 2.0, np.nan], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotA::ee"]

        per_ds_min, per_ds_max, _ = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)
        # Pooled real dims; dim 2 falls back to [-1, 1].
        np.testing.assert_array_equal(per_ds_min[0], np.array([-2.0, -2.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 2.0, 1.0], dtype=np.float32))


class TestNormalizeChunksPerHead:
    """``_normalize_chunks_per_head`` normalizes each dataset's slice
    independently using that dataset's (min, max).

    ``_sample_via_manual`` concatenates per-dataset chunk buckets in
    mixture-config order, so the helper relies on ``per_dataset_chunks``
    (the count per dataset) to know which row range belongs to each
    dataset. This test pins that contract: chunks from dataset 0 get
    normalized with ``per_ds_min[0]/max[0]``, chunks from dataset 1 with
    ``per_ds_min[1]/max[1]``, etc., regardless of how datasets with zero
    chunks are interleaved.
    """

    def test_per_dataset_slice_normalization(self):
        action_dim = 2
        chunk_size = 3
        # 3 datasets, the middle one had 0 chunks (degenerate, common
        # outcome of very-small total_chunks budgets after rounding).
        per_dataset_chunks = [2, 0, 1]
        # Dataset 0: chunks all equal to its raw max, so after [-1, 1]
        # normalization they should land at +1 on every dim.
        # Dataset 2: chunks all equal to its raw min, so should land at -1.
        per_ds_min = [
            np.array([-2.0, -2.0], dtype=np.float32),  # ds 0
            np.array([-1.0, -1.0], dtype=np.float32),  # ds 1 (unused, 0 chunks)
            np.array([-5.0, -10.0], dtype=np.float32),  # ds 2
        ]
        per_ds_max = [
            np.array([2.0, 2.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([5.0, 10.0], dtype=np.float32),
        ]

        # Build the stacked chunks accordingly.
        stacked = np.zeros((3, chunk_size, action_dim), dtype=np.float32)
        stacked[0:2] = 2.0  # ds 0 at raw_max
        stacked[2] = np.array([-5.0, -10.0], dtype=np.float32)  # ds 2 at raw_min

        out = _normalize_chunks_per_head(stacked, per_dataset_chunks, per_ds_min, per_ds_max)

        # First two rows -> all +1 (ds 0 at its raw_max)
        np.testing.assert_allclose(out[0:2], 1.0, atol=1e-7)
        # Third row -> all -1 (ds 2 at its raw_min)
        np.testing.assert_allclose(out[2], -1.0, atol=1e-7)

    def test_matches_global_when_all_datasets_share_one_head(self):
        """Per-head normalization with one shared (rt, cm) === global."""
        rng = np.random.default_rng(0)
        action_dim = 4
        n_chunks = 7
        chunk_size = 5
        stacked = rng.uniform(-1, 1, size=(n_chunks, chunk_size, action_dim)).astype(np.float32)

        # Two datasets that share the same pooled stats.
        per_dataset_chunks = [3, 4]
        shared_min = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
        shared_max = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

        out_per_head = _normalize_chunks_per_head(
            stacked, per_dataset_chunks, [shared_min, shared_min], [shared_max, shared_max]
        )
        out_global = _normalize_chunks(stacked, shared_min, shared_max)

        # Bit-exact: both paths apply the same affine transform.
        np.testing.assert_array_equal(out_per_head, out_global)
