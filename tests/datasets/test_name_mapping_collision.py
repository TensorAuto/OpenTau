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

"""Tests for per-instance feature-name mapping resolution.

Two mixture entries sharing a ``repo_id`` and ``control_mode`` while declaring
different ``data_features_name_mapping`` values (e.g. two camera views of the
same repo) used to clobber each other in the process-global
``DATA_FEATURES_NAME_MAPPING`` registry (last-wins), so both dataset instances
silently read the last entry's columns. Covers ``BaseDataset._get_name_map``
(per-instance mapping wins, registry fallback, strict behavior), the
end-to-end collision repro through the real ``LeRobotDataset`` constructor,
and the config-time warning when conflicting registrations share a key.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opentau.configs.default import DatasetConfig
from opentau.datasets.dataset_mixture import DatasetMixtureMetadata
from opentau.datasets.factory import make_dataset, resolve_delta_timestamps
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

_TEST_KEYS = [
    "_tests/name_mapping_dummy",
    "_tests/name_mapping_dummy::joint",
    "_tests/collision_repo",
    "_tests/collision_repo::joint",
    "_tests/collision_repo::ee",
]


@pytest.fixture(autouse=True)
def _clean_mapping_registration():
    """Remove this module's keys from the process-global registry."""
    yield
    for key in _TEST_KEYS:
        DATA_FEATURES_NAME_MAPPING.pop(key, None)


def _make_ds(instance_mapping: dict[str, str] | None) -> LeRobotDataset:
    ds = object.__new__(LeRobotDataset)
    ds.repo_id = "_tests/name_mapping_dummy"
    ds.control_mode = None
    if instance_mapping is not None:
        ds._data_features_name_mapping = instance_mapping
    return ds


class TestGetNameMap:
    def test_instance_mapping_wins_over_registry(self):
        DATA_FEATURES_NAME_MAPPING["_tests/name_mapping_dummy"] = {"camera0": "registry_cam"}
        ds = _make_ds({"camera0": "instance_cam"})
        assert ds._get_name_map()["camera0"] == "instance_cam"

    def test_registry_fallback_when_no_instance_mapping(self):
        DATA_FEATURES_NAME_MAPPING["_tests/name_mapping_dummy"] = {"camera0": "registry_cam"}
        ds = _make_ds(None)
        assert ds._get_name_map()["camera0"] == "registry_cam"

    def test_strict_raises_without_any_mapping(self):
        ds = _make_ds(None)
        with pytest.raises(KeyError):
            ds._get_name_map()

    def test_non_strict_returns_empty_without_any_mapping(self):
        ds = _make_ds(None)
        assert ds._get_name_map(strict=False) == {}


class TestSameRepoEntriesKeepOwnMappings:
    def test_two_instances_resolve_their_own_camera(self, tmp_path, lerobot_dataset_factory):
        """The collision repro: same repo_id, same (absent) control mode,
        different camera0 — each instance must read its own column."""
        map_a = {"camera0": "exterior_1_left", "state": "observation.state", "actions": "action"}
        map_b = {"camera0": "exterior_2_left", "state": "observation.state", "actions": "action"}
        ds_a = lerobot_dataset_factory(
            root=tmp_path / "a", standardize=False, data_features_name_mapping=map_a
        )
        ds_b = lerobot_dataset_factory(
            root=tmp_path / "b", standardize=False, data_features_name_mapping=map_b
        )
        assert ds_a._get_name_map()["camera0"] == "exterior_1_left"
        assert ds_b._get_name_map()["camera0"] == "exterior_2_left"

    def test_val_clone_shares_the_instance_mapping(self, tmp_path, lerobot_dataset_factory):
        mapping = {"camera0": "exterior_1_left", "state": "observation.state", "actions": "action"}
        ds = lerobot_dataset_factory(root=tmp_path, standardize=False, data_features_name_mapping=mapping)
        clone = ds.shallow_copy_with_dropout(enable_dropout=False)
        assert clone._get_name_map()["camera0"] == "exterior_1_left"


class TestConfigRegistrationWarning:
    def test_conflicting_same_key_registration_warns(self):
        DatasetConfig(
            repo_id="_tests/collision_repo",
            control_mode="joint",
            data_features_name_mapping={"camera0": "exterior_1_left"},
        )
        with pytest.warns(UserWarning, match="overwritten with a different mapping"):
            DatasetConfig(
                repo_id="_tests/collision_repo",
                control_mode="joint",
                data_features_name_mapping={"camera0": "exterior_2_left"},
            )
        # Registry keeps the last registration (documented last-wins).
        assert DATA_FEATURES_NAME_MAPPING["_tests/collision_repo::joint"]["camera0"] == "exterior_2_left"

    def test_identical_registration_does_not_warn(self):
        mapping = {"camera0": "exterior_1_left"}
        DatasetConfig(
            repo_id="_tests/collision_repo", control_mode="joint", data_features_name_mapping=mapping
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            DatasetConfig(
                repo_id="_tests/collision_repo",
                control_mode="joint",
                data_features_name_mapping=dict(mapping),
            )

    def test_different_control_modes_do_not_warn(self):
        """The joint/ee pattern: plain repo_id fallback overwrite is by design."""
        DatasetConfig(
            repo_id="_tests/collision_repo",
            control_mode="joint",
            data_features_name_mapping={"actions": "action_joint"},
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            DatasetConfig(
                repo_id="_tests/collision_repo",
                control_mode="ee",
                data_features_name_mapping={"actions": "action_ee"},
            )


def _colliding_configs() -> tuple[DatasetConfig, DatasetConfig]:
    """Two mixture entries, same repo_id + control_mode, different camera0."""
    cfg_a = DatasetConfig(
        repo_id="_tests/collision_repo",
        control_mode="joint",
        data_features_name_mapping={
            "camera0": "exterior_1_left",
            "state": "observation.state",
            "actions": "action",
        },
    )
    with pytest.warns(UserWarning, match="overwritten with a different mapping"):
        cfg_b = DatasetConfig(
            repo_id="_tests/collision_repo",
            control_mode="joint",
            data_features_name_mapping={
                "camera0": "exterior_2_left",
                "state": "observation.state",
                "actions": "action",
            },
        )
    return cfg_a, cfg_b


class TestResolveDeltaTimestampsPerEntryMap:
    def test_colliding_entries_get_own_camera_deltas(self):
        """resolve_delta_timestamps runs in the same make_dataset path — it
        must use the entry's own mapping, not the (clobbered) registry."""
        cfg_a, cfg_b = _colliding_configs()
        train_cfg = SimpleNamespace(
            dataset_mixture=SimpleNamespace(action_freq=15.0, n_obs_history=None), policy=None
        )
        meta = SimpleNamespace(
            features={"exterior_1_left": {}, "exterior_2_left": {}, "observation.state": {}, "action": {}},
            fps=15,
            control_mode="joint",
        )
        dt_a, _, _, _ = resolve_delta_timestamps(train_cfg, cfg_a, meta)
        dt_b, _, _, _ = resolve_delta_timestamps(train_cfg, cfg_b, meta)
        assert "exterior_1_left" in dt_a and "exterior_2_left" not in dt_a
        assert "exterior_2_left" in dt_b and "exterior_1_left" not in dt_b


class TestMixtureMetadataPerEntryMaps:
    @staticmethod
    def _entry_meta(base, cam_col: str, cam_mean: float):
        stats = copy.deepcopy({k: v for k, v in base.stats.items() if k != "camera0"})
        cam_stats = copy.deepcopy(base.stats["camera0"])
        cam_stats["mean"] = np.full_like(cam_stats["mean"], cam_mean)
        stats[cam_col] = cam_stats
        return SimpleNamespace(repo_id="_tests/collision_repo", info=dict(base.info), stats=stats)

    def test_colliding_entries_standardize_own_camera_stats(
        self, train_pipeline_config, lerobot_dataset_metadata
    ):
        m_a = self._entry_meta(lerobot_dataset_metadata, "exterior_1_left", 0.25)
        m_b = self._entry_meta(lerobot_dataset_metadata, "exterior_2_left", 0.75)
        maps = [
            {"camera0": "exterior_1_left", "state": "state", "actions": "actions"},
            {"camera0": "exterior_2_left", "state": "state", "actions": "actions"},
        ]
        mm = DatasetMixtureMetadata(
            train_pipeline_config, [m_a, m_b], [0.5, 0.5], dataset_names=["a", "b"], name_maps=maps
        )
        assert float(np.asarray(mm.per_dataset_stats[0]["camera0"]["mean"]).reshape(-1)[0]) == 0.25
        assert float(np.asarray(mm.per_dataset_stats[1]["camera0"]["mean"]).reshape(-1)[0]) == 0.75

    def test_success_role_column_excluded_from_stats_schema(
        self, train_pipeline_config, lerobot_dataset_metadata
    ):
        m = self._entry_meta(lerobot_dataset_metadata, "exterior_1_left", 0.5)
        m.stats["is_episode_successful"] = {
            "min": np.array([0.0]),
            "max": np.array([1.0]),
            "mean": np.array([0.8]),
            "std": np.array([0.4]),
            "count": np.array([150]),
        }
        maps = [
            {
                "camera0": "exterior_1_left",
                "state": "state",
                "actions": "actions",
                "success": "is_episode_successful",
            }
        ]
        mm = DatasetMixtureMetadata(train_pipeline_config, [m], [1.0], dataset_names=["a"], name_maps=maps)
        assert "success" not in mm.per_dataset_stats[0]
        assert "is_episode_successful" not in mm.per_dataset_stats[0]


class TestMakeDatasetThreadsInstanceMapping:
    def test_each_entry_constructor_receives_own_mapping(self, train_pipeline_config):
        """make_dataset must hand each colliding entry its own mapping."""
        cfg_a, cfg_b = _colliding_configs()
        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds_cls.return_value = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))
            make_dataset(cfg_a, train_pipeline_config)
            make_dataset(cfg_b, train_pipeline_config)
        first, second = mock_ds_cls.call_args_list
        assert first.kwargs["data_features_name_mapping"]["camera0"] == "exterior_1_left"
        assert second.kwargs["data_features_name_mapping"]["camera0"] == "exterior_2_left"
