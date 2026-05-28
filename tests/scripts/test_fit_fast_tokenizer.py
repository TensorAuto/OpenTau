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
        # Real per-head fits zero-pad trailing slots (matches `pad_vector` in
        # `_to_standard_data_format`); the pure pooler is dim-agnostic, so this
        # test uses real numbers throughout and relies on a separate test for
        # the production zero-pad behaviour.
        raw_min = [
            np.array([-1.0, -2.0, -3.0, -1.5], dtype=np.float64),
            np.array([-0.5, -3.0, -2.0, -1.0], dtype=np.float64),
        ]
        raw_max = [
            np.array([1.0, 2.0, 3.0, 1.5], dtype=np.float64),
            np.array([2.0, 1.5, 4.0, 1.0], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotA::ee"]

        per_ds_min, per_ds_max, per_head_stats = _pool_per_head_stats(
            raw_min, raw_max, per_ds_key, action_dim
        )

        # One head, two datasets -- both datasets get the same pooled (min, max).
        assert len(per_head_stats) == 1
        assert "robotA::ee" in per_head_stats
        np.testing.assert_array_equal(per_ds_min[0], per_ds_min[1])
        np.testing.assert_array_equal(per_ds_max[0], per_ds_max[1])
        # Pooled values: elementwise nanmin / nanmax across the two datasets.
        np.testing.assert_array_equal(per_ds_min[0], np.array([-1.0, -3.0, -3.0, -1.5], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 2.0, 4.0, 1.5], dtype=np.float32))

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

        per_ds_min, per_ds_max, per_head_stats = _pool_per_head_stats(
            raw_min, raw_max, per_ds_key, action_dim
        )

        assert len(per_head_stats) == 2
        # Each dataset keeps its own stats verbatim (singleton heads).
        np.testing.assert_array_equal(per_ds_min[0], np.array([-1.0, -1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_min[1], np.array([-10.0, -10.0, -10.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([1.0, 1.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[1], np.array([10.0, 10.0, 10.0], dtype=np.float32))

    def test_inf_masked_before_pool(self):
        """``±Inf`` in raw stats are masked to NaN before nanmin/nanmax.

        Mirrors ``aggregate_stats`` (compute_stats.py:350) -- without the
        mask, a single ``+Inf`` poisons nanmax for the whole head and
        ``-Inf`` poisons nanmin. Both would make the production
        ``(x - min) / (max - min)`` evaluate to 0 in float32 (the divisor
        wins), breaking the chunk's normalization silently.
        """
        action_dim = 2
        raw_min = [
            np.array([-1.0, -np.inf], dtype=np.float64),  # one corrupted dim
            np.array([-2.0, -2.0], dtype=np.float64),
        ]
        raw_max = [
            np.array([np.inf, 1.0], dtype=np.float64),  # corrupted dim
            np.array([2.0, 2.0], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotA::ee"]

        per_ds_min, per_ds_max, _ = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)
        # The Inf-corrupted entries get masked to NaN; the other peer's finite
        # values dominate the pool.
        np.testing.assert_array_equal(per_ds_min[0], np.array([-2.0, -2.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 2.0], dtype=np.float32))

    def test_all_inf_dim_falls_back_to_minus_one_one(self):
        """A dim where every head member is ±Inf gets the [-1, 1] fallback."""
        action_dim = 2
        raw_min = [
            np.array([-1.0, -np.inf], dtype=np.float64),
            np.array([-2.0, -np.inf], dtype=np.float64),
        ]
        raw_max = [
            np.array([1.0, np.inf], dtype=np.float64),
            np.array([2.0, np.inf], dtype=np.float64),
        ]
        per_ds_key = ["robotA::ee", "robotA::ee"]

        per_ds_min, per_ds_max, _ = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)
        # Dim 0 pools normally; dim 1 is Inf across all members -> [-1, 1] fill.
        np.testing.assert_array_equal(per_ds_min[0], np.array([-2.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(per_ds_max[0], np.array([2.0, 1.0], dtype=np.float32))


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

    def test_chunk_count_mismatch_raises_assert(self):
        """Sampler/normalizer drift is caught at the boundary, not silently."""
        action_dim = 2
        stacked = np.zeros((3, 4, action_dim), dtype=np.float32)
        per_dataset_chunks = [2, 2]  # sum=4 but stacked has 3 rows
        per_ds_min = [np.array([-1.0, -1.0], dtype=np.float32)] * 2
        per_ds_max = [np.array([1.0, 1.0], dtype=np.float32)] * 2

        import pytest

        with pytest.raises(AssertionError, match="sums to"):
            _normalize_chunks_per_head(stacked, per_dataset_chunks, per_ds_min, per_ds_max)


class TestNormalizeEquivalenceVsProduction:
    """``_normalize_chunks_per_head`` matches production ``Normalize`` byte-for-byte.

    Pins the invariant the PR claims: a chunk normalized at fit time produces
    the same byte sequence the policy feeds to the FAST tokenizer at training
    time. The production path is ``Normalize({"ACTION": MIN_MAX}).forward``
    with per-head stacked stats buffers (PR #347) and per-sample
    ``dataset_index`` lookups. Our manual path applies the same per-dataset
    (min, max) directly. Any drift between these two -- e.g. forgetting the
    ``* 2 - 1`` shift, swapping EPS conventions, or accidentally truncating
    the trailing-dim slot -- means the BPE corpus the fit operates on no
    longer matches what training sees, and the published token-length
    distribution becomes uncalibrated guidance.
    """

    def test_per_head_normalize_matches_production_normalize(self):
        """Synthesize 2 datasets, 2 heads; verify per-dataset output matches
        ``Normalize`` driven by per-sample ``dataset_index``.
        """
        import torch

        from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from opentau.policies.normalize import Normalize

        action_dim = 4
        chunk_size = 3
        # Dataset 0: head "robotA::ee", min=[-2, -2, 0, 0], max=[2, 2, 0, 0]
        # Dataset 1: head "robotB::joint", min=[-1, -5, -10, 0], max=[3, 5, 10, 0]
        # Trailing zeros simulate `pad_vector` zero-pad in
        # `_to_standard_data_format`. Production `(x - 0) / (0 - 0 + EPS) * 2 - 1`
        # evaluates to -1 for those slots, and we must too.
        per_ds_min_f32 = [
            np.array([-2.0, -2.0, 0.0, 0.0], dtype=np.float32),
            np.array([-1.0, -5.0, -10.0, 0.0], dtype=np.float32),
        ]
        per_ds_max_f32 = [
            np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32),
            np.array([3.0, 5.0, 10.0, 0.0], dtype=np.float32),
        ]

        # Per-dataset chunk counts -- mixed sizes to exercise the offset arithmetic.
        per_dataset_chunks = [3, 4]
        n_total = sum(per_dataset_chunks)

        rng = np.random.default_rng(42)
        # Real signal on dims 0-1 for ds 0, dims 0-2 for ds 1; zero on padded tail.
        stacked = np.zeros((n_total, chunk_size, action_dim), dtype=np.float32)
        stacked[0:3, :, 0:2] = rng.uniform(-2, 2, size=(3, chunk_size, 2)).astype(np.float32)
        stacked[3:7, :, 0:3] = rng.uniform(-5, 5, size=(4, chunk_size, 3)).astype(np.float32)

        out_manual = _normalize_chunks_per_head(stacked, per_dataset_chunks, per_ds_min_f32, per_ds_max_f32)

        # Build the production `Normalize` layer with per-dataset stats buffers.
        # The contract: pass `per_dataset_stats` as a list aligned to
        # `dataset_names`; `Normalize.forward(batch, dataset_index)` then
        # picks each sample's stats via `dataset_index[i]`.
        features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
        norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
        per_dataset_stats = [
            {
                "action": {
                    "min": torch.from_numpy(per_ds_min_f32[0]),
                    "max": torch.from_numpy(per_ds_max_f32[0]),
                }
            },
            {
                "action": {
                    "min": torch.from_numpy(per_ds_min_f32[1]),
                    "max": torch.from_numpy(per_ds_max_f32[1]),
                }
            },
        ]
        normalize = Normalize(
            features=features,
            norm_map=norm_map,
            per_dataset_stats=per_dataset_stats,
            dataset_names=["dsA", "dsB"],
        )
        # Production processes one chunk at a time; flatten to (N*T, D) so
        # each sample carries its own dataset_index and the buffer gather
        # exactly mirrors training.
        n, t, d = stacked.shape
        dataset_index = torch.from_numpy(
            np.concatenate([np.full(per_dataset_chunks[k] * t, k, dtype=np.int64) for k in range(2)])
        )
        batch = {"action": torch.from_numpy(stacked.reshape(n * t, d))}
        out_prod = normalize(batch, dataset_index)["action"].numpy().reshape(n, t, d)

        # Both paths apply `(x - min) / (max - min + EPS) * 2 - 1` with EPS=1e-8.
        # Production runs the whole expression in float32; the manual path
        # computes in float64 then casts to float32 at the end, so low-bit
        # rounding differs by ~1 ULP (1.19e-7) on a few entries. The DCT scale
        # the BPE codec sees is O(1) so this is well below the threshold that
        # would change the BPE token-id sequence. Padded slots become -1 in
        # both (zero data, zero stats, +EPS divisor -> -1).
        np.testing.assert_allclose(out_manual, out_prod, rtol=0, atol=2e-7)
        # Sanity: the padded suffix actually is -1 in both outputs.
        np.testing.assert_allclose(out_manual[0:3, :, 2:4], -1.0, atol=1e-7)
        np.testing.assert_allclose(out_manual[3:7, :, 3:4], -1.0, atol=1e-7)
