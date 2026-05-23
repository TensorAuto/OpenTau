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
from opentau.scripts.fit_fast_tokenizer import _sample_chunks_for_dataset_manual
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
        # ``assert_allclose`` (not ``assert_array_equal``) because the
        # interpolator targets ``t0 + k*(1/fps)`` floats which are not
        # bit-equal to the native timestamps for general ``t0``; the
        # rounding error stays well below 1e-6 in float32.
        for c_none, c_native in zip(chunks_none, chunks_native, strict=True):
            np.testing.assert_allclose(c_none, c_native, rtol=1e-6, atol=1e-6)
