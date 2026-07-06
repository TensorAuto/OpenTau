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
-> v3.0 flattened stats aggregate -> legacy per-episode ``success`` key) and
``LeRobotDataset._attach_mistake_raw`` (deriving ``mistake_raw`` with the
mistake-polarity column taking precedence over an inverted ``success`` role).
"""

from __future__ import annotations

import torch

from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

resolve = LeRobotDataset._resolve_episode_success

_TEST_MAPPING_KEY = "_tests/success_role_dummy"


def _make_ds(mapping: dict[str, str]) -> LeRobotDataset:
    """Minimal LeRobotDataset carrying just what _attach_mistake_raw reads."""
    DATA_FEATURES_NAME_MAPPING[_TEST_MAPPING_KEY] = mapping
    ds = object.__new__(LeRobotDataset)
    ds.repo_id = _TEST_MAPPING_KEY
    ds.control_mode = None
    return ds


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

    def test_mapped_stats_mean_fallback(self):
        nm = {"success": "is_episode_successful"}
        assert resolve({}, {"stats/is_episode_successful/mean": [1.0]}, nm) is True
        assert resolve({}, {"stats/is_episode_successful/mean": [0.0]}, nm) is False

    def test_frame_column_beats_episode_meta_and_stats(self):
        nm = {"success": "flag"}
        info = {"flag": True, "stats/flag/mean": [1.0]}
        assert resolve({"flag": torch.tensor(False)}, info, nm) is False

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


class TestAttachMistakeRaw:
    def test_success_column_inverts_to_mistake(self):
        ds = _make_ds({"success": "is_episode_successful"})
        item = {"is_episode_successful": torch.tensor(False)}
        assert ds._attach_mistake_raw(item, {}) is False
        assert item["mistake_raw"] == 1

        item = {"is_episode_successful": torch.tensor(True)}
        assert ds._attach_mistake_raw(item, {}) is True
        assert item["mistake_raw"] == 0

    def test_mapped_mistake_column(self):
        ds = _make_ds({"mistake": "is_mistake"})
        item = {"is_mistake": torch.tensor(1)}
        ds._attach_mistake_raw(item, {})
        assert item["mistake_raw"] == 1

    def test_mistake_column_wins_over_success_role(self):
        ds = _make_ds({"success": "ok_flag", "mistake": "seg_mistake"})
        item = {"seg_mistake": torch.tensor(1), "ok_flag": torch.tensor(True)}
        assert ds._attach_mistake_raw(item, {}) is True
        assert item["mistake_raw"] == 1

    def test_literal_mistake_column_needs_no_mapping(self):
        ds = _make_ds({})
        item = {"mistake": torch.tensor(1)}
        ds._attach_mistake_raw(item, {})
        assert item["mistake_raw"] == 1

    def test_stats_aggregate_derives_mistake(self):
        ds = _make_ds({"success": "is_episode_successful"})
        item = {}
        assert ds._attach_mistake_raw(item, {"stats/is_episode_successful/mean": [0.0]}) is False
        assert item["mistake_raw"] == 1

    def test_unlabeled_attaches_nothing(self):
        ds = _make_ds({})
        item = {}
        assert ds._attach_mistake_raw(item, {}) is None
        assert "mistake_raw" not in item
