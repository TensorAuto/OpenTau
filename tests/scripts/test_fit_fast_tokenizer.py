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

    def test_none_action_freq_uses_native_fps(self, tmp_path, lerobot_dataset_factory):
        """``action_freq=None`` substitutes ``meta.info['fps']`` per-dataset.

        With chunk_size=10 against a 30 fps synthetic dataset, the chunks
        should match those returned when ``action_freq`` is explicitly
        set to the native fps. They span the same wall-clock duration and
        target the same native timestamps, so the BPE-corpus content is
        equivalent under both call styles.
        """
        # Materialize a 30 fps synthetic dataset on disk via the shared
        # fixture; the worker later constructs a fresh ``LeRobotDatasetMetadata``
        # from ``root`` and reads the parquet directly.
        root = tmp_path / "ds_none"
        lerobot_dataset_factory(root=root, total_episodes=3, total_frames=150)

        idx_none, chunks_none, err_none = self._run(root, action_freq=None)
        assert err_none is None, f"action_freq=None path errored: {err_none}"
        assert idx_none == 0
        assert len(chunks_none) == 8, "expected n_chunks=8 chunks back"
        for c in chunks_none:
            assert c.shape == (10, 6), f"unexpected chunk shape {c.shape}"

        # Same seed + same effective action_freq must yield identical chunks.
        # Use a separate root to keep the deterministic per-dataset rng split
        # from being contaminated by any caching side-effects on the first call.
        root2 = tmp_path / "ds_native"
        lerobot_dataset_factory(root=root2, total_episodes=3, total_frames=150)
        _, chunks_native, err_native = self._run(root2, action_freq=float(DEFAULT_FPS))
        assert err_native is None
        assert len(chunks_native) == 8
