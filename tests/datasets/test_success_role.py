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

"""Tests for the standard ``mistake`` / ``success`` feature-mapping roles.

Covers ``LeRobotDataset._resolve_episode_success`` (the resolution chain for
success-polarity signals: mapped frame column -> mapped episode-metadata key
-> per-episode ``episodes_stats`` mean aggregate -> legacy per-episode
``success`` key), ``LeRobotDataset._attach_mistake_raw`` (deriving
``mistake_raw`` with the mistake-polarity column taking precedence over an
inverted ``success`` role), and the episode-constancy counting over
``episodes_stats``. The stats-shaped tests go through the real
``load_episodes_v30`` / ``load_episodes_stats_v30`` loaders so the helpers are
exercised against loader-emitted structures, not hand-built dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import packaging.version
import pyarrow as pa
import pytest
import torch

from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from opentau.datasets.utils import load_episodes_stats_v30, load_episodes_v30

resolve = LeRobotDataset._resolve_episode_success

_TEST_MAPPING_KEY = "_tests/success_role_dummy"


@pytest.fixture(autouse=True)
def _clean_mapping_registration():
    """Remove the test's mapping key from the process-global registry."""
    yield
    DATA_FEATURES_NAME_MAPPING.pop(_TEST_MAPPING_KEY, None)


def _make_ds(
    mapping: dict[str, str],
    episodes_stats: dict | None = None,
    version: str = "v3.0",
) -> LeRobotDataset:
    """Minimal LeRobotDataset carrying just what _attach_mistake_raw reads."""
    DATA_FEATURES_NAME_MAPPING[_TEST_MAPPING_KEY] = mapping
    ds = object.__new__(LeRobotDataset)
    ds.repo_id = _TEST_MAPPING_KEY
    ds.control_mode = None
    ds.meta = SimpleNamespace(_version=packaging.version.parse(version), episodes_stats=episodes_stats or {})
    return ds


def _v30_episodes_table(success_stats: list[tuple[float, float, float]]) -> pa.Table:
    """Build a v3.0 episodes-metadata Arrow table with flattened stats columns.

    ``success_stats`` holds one ``(min, max, mean)`` triple per episode for a
    column named ``ok`` — the same flattened ``stats/{feature}/{stat}`` layout
    ``_read_v30_episodes_arrow`` produces from the metadata shards.
    """
    n = len(success_stats)
    return pa.table(
        {
            "episode_index": pa.array(range(n), type=pa.int64()),
            "length": pa.array([10] * n, type=pa.int64()),
            "tasks": pa.array([["task"]] * n, type=pa.list_(pa.string())),
            "stats/ok/min": pa.array([[lo] for lo, _, _ in success_stats], type=pa.list_(pa.float64())),
            "stats/ok/max": pa.array([[hi] for _, hi, _ in success_stats], type=pa.list_(pa.float64())),
            "stats/ok/mean": pa.array([[mu] for _, _, mu in success_stats], type=pa.list_(pa.float64())),
        }
    )


class TestResolveEpisodeSuccess:
    def test_mapped_frame_column_bool_tensor(self):
        nm = {"success": "is_episode_successful"}
        assert resolve({"is_episode_successful": torch.tensor(True)}, {}, nm) is True
        assert resolve({"is_episode_successful": torch.tensor([False])}, {}, nm) is False

    def test_mapped_frame_column_plain_int(self):
        nm = {"success": "task_success"}
        assert resolve({"task_success": 1}, {}, nm) is True
        assert resolve({"task_success": 0}, {}, nm) is False

    def test_mapped_episode_meta_key(self):
        nm = {"success": "is_episode_successful"}
        assert resolve({}, {"is_episode_successful": True}, nm) is True
        assert resolve({}, {"is_episode_successful": 0}, nm) is False

    def test_episode_stats_mean_fallback_uses_loader_shape(self):
        nm = {"success": "ok"}
        stats = load_episodes_stats_v30(None, episodes_table=_v30_episodes_table([(1, 1, 1), (0, 0, 0)]))
        assert resolve({}, {}, nm, episode_stats=stats[0]) is True
        assert resolve({}, {}, nm, episode_stats=stats[1]) is False

    def test_frame_column_beats_episode_meta_and_stats(self):
        nm = {"success": "flag"}
        info = {"flag": True}
        stats = {"flag": {"mean": np.array([1.0])}}
        assert resolve({"flag": torch.tensor(False)}, info, nm, episode_stats=stats) is False

    def test_legacy_success_key_without_mapping(self):
        assert resolve({}, {"success": False}, {}) is False
        assert resolve({}, {"success": True}, {}) is True

    def test_mapped_role_beats_legacy_success_key(self):
        nm = {"success": "flag"}
        assert resolve({"flag": torch.tensor(False)}, {"success": True}, nm) is False

    def test_mapped_but_absent_falls_back_to_legacy_key(self):
        nm = {"success": "absent_col"}
        assert resolve({}, {"success": False}, nm) is False

    def test_none_when_unlabeled(self):
        assert resolve({}, {}, {}) is None
        assert resolve({}, {}, {"success": "absent_col"}) is None


class TestEpisodeStatsFor:
    def test_returns_entry_for_v21_plus(self):
        stats = {0: {"ok": {"mean": np.array([1.0])}}}
        ds = _make_ds({}, episodes_stats=stats, version="v3.0")
        assert ds._episode_stats_for(0) is stats[0]
        assert ds._episode_stats_for(99) is None

    def test_v20_backward_compatible_stats_are_ignored(self):
        # v2.0 episodes_stats replicate the *global* aggregate per episode —
        # not an episode-level signal, so the lookup must return None.
        stats = {0: {"ok": {"mean": np.array([0.5])}}}
        ds = _make_ds({}, episodes_stats=stats, version="v2.0")
        assert ds._episode_stats_for(0) is None


class TestAttachMistakeRaw:
    def test_success_column_inverts_to_mistake(self):
        ds = _make_ds({"success": "is_episode_successful"})
        item = {"is_episode_successful": torch.tensor(False)}
        assert ds._attach_mistake_raw(item, {}, 0) is False
        assert item["mistake_raw"] == 1

        item = {"is_episode_successful": torch.tensor(True)}
        assert ds._attach_mistake_raw(item, {}, 0) is True
        assert item["mistake_raw"] == 0

    def test_mapped_mistake_column(self):
        ds = _make_ds({"mistake": "is_mistake"})
        item = {"is_mistake": torch.tensor(1)}
        ds._attach_mistake_raw(item, {}, 0)
        assert item["mistake_raw"] == 1

    def test_mistake_column_wins_over_success_role(self):
        ds = _make_ds({"success": "ok_flag", "mistake": "seg_mistake"})
        item = {"seg_mistake": torch.tensor(1), "ok_flag": torch.tensor(True)}
        assert ds._attach_mistake_raw(item, {}, 0) is True
        assert item["mistake_raw"] == 1

    def test_literal_mistake_column_needs_no_mapping(self):
        ds = _make_ds({})
        item = {"mistake": torch.tensor(1)}
        ds._attach_mistake_raw(item, {}, 0)
        assert item["mistake_raw"] == 1

    def test_episode_stats_aggregate_derives_mistake(self):
        stats = load_episodes_stats_v30(None, episodes_table=_v30_episodes_table([(0, 0, 0)]))
        ds = _make_ds({"success": "ok"}, episodes_stats=stats)
        item = {}
        assert ds._attach_mistake_raw(item, {}, 0) is False
        assert item["mistake_raw"] == 1

    def test_v20_stats_do_not_derive_mistake(self):
        stats = {0: {"ok": {"mean": np.array([0.0])}}}
        ds = _make_ds({"success": "ok"}, episodes_stats=stats, version="v2.0")
        item = {}
        assert ds._attach_mistake_raw(item, {}, 0) is None
        assert "mistake_raw" not in item

    def test_unlabeled_attaches_nothing(self):
        ds = _make_ds({})
        item = {}
        assert ds._attach_mistake_raw(item, {}, 0) is None
        assert "mistake_raw" not in item


class TestCountSuccessRoleVaryingEpisodes:
    count = staticmethod(LeRobotDataset._count_success_role_varying_episodes)

    def _loader_stats(self, triples):
        return load_episodes_stats_v30(None, episodes_table=_v30_episodes_table(triples))

    def test_constant_episodes_count_zero(self):
        stats = self._loader_stats([(1, 1, 1), (0, 0, 0)])
        assert self.count(stats, "ok") == (0, 2)

    def test_varying_episode_detected(self):
        stats = self._loader_stats([(0, 1, 0.5), (1, 1, 1)])
        assert self.count(stats, "ok") == (1, 2)

    def test_missing_column_is_skipped_not_flagged(self):
        stats = self._loader_stats([(1, 1, 1)])
        assert self.count(stats, "other_col") == (0, 0)

    def test_episodes_metadata_records_have_no_stats(self):
        # `load_episodes_v30` strips every `stats/*` column from the episode
        # records; the count must run on `episodes_stats`, not `episodes`.
        table = _v30_episodes_table([(0, 1, 0.5)])
        episodes = load_episodes_v30(None, episodes_table=table)
        assert not any(k.startswith("stats/") for k in episodes[0])
        stats = load_episodes_stats_v30(None, episodes_table=table)
        assert self.count(stats, "ok") == (1, 1)
