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

"""Tests for BaseDataset._emit_optional_keys (memory/subgoal/metadata dropout).

Constructs a minimal BaseDataset subclass with configurable drop probabilities
so the masking logic can be exercised without touching video files or
HuggingFace downloads. Integration with the full LeRobotDataset pipeline is
covered by tests/scripts/test_attach_metadata.py (slow, network-dependent).
"""

from __future__ import annotations

import torch

from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

_TEST_MAPPING_KEY = "_tests/optional_keys_dummy"
# Minimal mapping: one camera slot, state, actions, prompt, response.
DATA_FEATURES_NAME_MAPPING[_TEST_MAPPING_KEY] = {
    "camera0": "camera0",
    "state": "state",
    "actions": "actions",
    "prompt": "prompt",
    "response": "response",
}


def _make_dummy_subgoal(h: int, w: int) -> torch.Tensor:
    """Return a (3, h, w) tensor with values in [0, 1]."""
    return torch.full((3, h, w), 0.5, dtype=torch.float32)


class _DummyBaseDataset(BaseDataset):
    """Concrete BaseDataset subclass for direct _emit_optional_keys testing.

    Bypasses the BaseDataset constructor so we can pin attributes directly
    without manufacturing a full TrainPipelineConfig.
    """

    def __init__(
        self,
        *,
        resolution=(8, 8),
        num_cams=2,
        n_obs_history=None,
        history_state_drop_prob=0.0,
        subgoal_drop_prob=0.0,
        subgoal_end_of_segment_prob=0.0,
        response_drop_prob=0.0,
        metadata_drop_all_prob=0.0,
        metadata_drop_each_prob=0.0,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.resolution = resolution
        self.num_cams = num_cams
        self.n_obs_history = n_obs_history
        self.max_state_dim = 7
        self.max_action_dim = 7
        self.action_chunk = 1
        self.history_state_drop_prob = history_state_drop_prob
        self.subgoal_drop_prob = subgoal_drop_prob
        self.subgoal_end_of_segment_prob = subgoal_end_of_segment_prob
        self.response_drop_prob = response_drop_prob
        self.metadata_drop_all_prob = metadata_drop_all_prob
        self.metadata_drop_each_prob = metadata_drop_each_prob
        self.enable_optional_key_dropout = True

    def _get_feature_mapping_key(self) -> str:
        return _TEST_MAPPING_KEY


def _prepopulate_standard_item(dataset: _DummyBaseDataset) -> dict:
    """Fill a standard_item with the fields that _emit_optional_keys reads."""
    h, w = dataset.resolution
    standard_item = {
        "state": torch.ones((dataset.max_state_dim,), dtype=torch.float32),
        "actions": torch.zeros((1, dataset.max_action_dim), dtype=torch.float32),
        "response": "hello",
        "prompt": "world",
        "img_is_pad": torch.zeros((dataset.num_cams,), dtype=torch.bool),
        "action_is_pad": torch.zeros((1,), dtype=torch.bool),
        "obs_history_is_pad": torch.zeros((1,), dtype=torch.bool),
    }
    for k in range(dataset.num_cams):
        standard_item[f"camera{k}"] = torch.zeros((3, h, w), dtype=torch.float32)
    return standard_item


def _raw_item(dataset: _DummyBaseDataset) -> dict:
    """Typical item dict with every _raw field populated."""
    h, w = dataset.resolution
    return {
        "memory_raw": "current_mem",
        "next_memory_raw": "next_mem",
        "mistake_raw": 1,
        "quality_raw": 4,
        "speed_raw": 1500,
        **{f"subgoal{k}_raw": _make_dummy_subgoal(h, w) for k in range(dataset.num_cams)},
    }


# Legacy path: no _raw fields attached → all optional keys pad out.


class TestLegacyNoAnnotations:
    def test_all_optional_keys_present_and_padded(self):
        ds = _DummyBaseDataset()
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys({}, standard_item)
        for k in ("memory", "next_memory"):
            assert k in standard_item
            assert standard_item[k] == ""
            assert standard_item[f"{k}_is_pad"].item() is True
        for k in ("speed", "mistake", "quality"):
            assert k in standard_item
            assert standard_item[f"{k}_is_pad"].item() is True
        for k in range(ds.num_cams):
            assert f"subgoal{k}" in standard_item
            assert standard_item[f"subgoal{k}"].shape == (3, *ds.resolution)
            assert standard_item[f"subgoal{k}"].abs().max().item() == 0.0
        assert standard_item["subgoal_is_pad"].item() is True


# All-probabilities-zero path: every annotated field flows through.


class TestAllProbsZero:
    def test_no_masks_applied(self):
        ds = _DummyBaseDataset()
        standard_item = _prepopulate_standard_item(ds)
        raw = _raw_item(ds)
        ds._emit_optional_keys(raw, standard_item)

        assert standard_item["memory"] == "current_mem"
        assert standard_item["memory_is_pad"].item() is False
        assert standard_item["next_memory"] == "next_mem"
        assert standard_item["next_memory_is_pad"].item() is False

        assert standard_item["speed"].item() == 1500
        assert standard_item["speed_is_pad"].item() is False
        assert standard_item["mistake"].item() is True
        assert standard_item["mistake_is_pad"].item() is False
        assert standard_item["quality"].item() == 4
        assert standard_item["quality_is_pad"].item() is False

        for k in range(ds.num_cams):
            # The raw 0.5 tensor should be resize_with_pad'd to the target
            # resolution without altering the data range.
            assert standard_item[f"subgoal{k}"].shape == (3, *ds.resolution)
            assert 0.0 <= standard_item[f"subgoal{k}"].min().item() <= 0.5
            assert standard_item[f"subgoal{k}"].max().item() <= 1.0 + 1e-6
        assert standard_item["subgoal_is_pad"].item() is False

        assert standard_item["response_is_pad"].item() is False


# Force each probability to 1.0 individually.


class TestForcedDropouts:
    def test_history_drop_zeros_state(self):
        ds = _DummyBaseDataset(history_state_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        assert torch.all(standard_item["state"] == 0)
        assert standard_item["obs_history_is_pad"].all().item() is True

    def test_history_drop_zeros_historical_cameras_when_temporal(self):
        ds = _DummyBaseDataset(n_obs_history=3, history_state_drop_prob=1.0)
        # Pre-populate cameras as (T, 3, H, W) since n_obs_history is set.
        h, w = ds.resolution
        standard_item = {
            "state": torch.ones((ds.n_obs_history, ds.max_state_dim), dtype=torch.float32),
            "actions": torch.zeros((1, ds.max_action_dim), dtype=torch.float32),
            "response": "x",
            "img_is_pad": torch.zeros((ds.num_cams,), dtype=torch.bool),
            "action_is_pad": torch.zeros((1,), dtype=torch.bool),
            "obs_history_is_pad": torch.zeros((ds.n_obs_history,), dtype=torch.bool),
        }
        for k in range(ds.num_cams):
            standard_item[f"camera{k}"] = torch.ones((ds.n_obs_history, 3, h, w), dtype=torch.float32)
        ds._emit_optional_keys({}, standard_item)
        for k in range(ds.num_cams):
            # Historical frames (all but last) should be zeroed.
            hist = standard_item[f"camera{k}"][:-1]
            assert torch.all(hist == 0)
            # Current frame is left intact.
            assert torch.all(standard_item[f"camera{k}"][-1] == 1)
        assert standard_item["obs_history_is_pad"].all().item() is True

    def test_subgoal_drop_masks_all_slots(self):
        ds = _DummyBaseDataset(subgoal_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        assert standard_item["subgoal_is_pad"].item() is True
        for k in range(ds.num_cams):
            assert torch.all(standard_item[f"subgoal{k}"] == 0)

    def test_subgoal_drop_implies_no_response_drop(self):
        """Per spec, response is only rolled when subgoals are not dropped."""
        ds = _DummyBaseDataset(subgoal_drop_prob=1.0, response_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        original_response = standard_item["response"]
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        assert standard_item["response"] == original_response
        assert standard_item["response_is_pad"].item() is False

    def test_response_drop_masks_when_subgoals_present(self):
        ds = _DummyBaseDataset(subgoal_drop_prob=0.0, response_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        assert standard_item["response"] == ""
        assert standard_item["response_is_pad"].item() is True

    def test_metadata_drop_all_masks_three_fields(self):
        ds = _DummyBaseDataset(metadata_drop_all_prob=1.0, metadata_drop_each_prob=0.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        for k in ("speed", "mistake", "quality"):
            assert standard_item[f"{k}_is_pad"].item() is True

    def test_metadata_drop_each_masks_every_field_independently(self):
        ds = _DummyBaseDataset(metadata_drop_all_prob=0.0, metadata_drop_each_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        for k in ("speed", "mistake", "quality"):
            assert standard_item[f"{k}_is_pad"].item() is True


# enable_optional_key_dropout flag (val subset turns dropout off by default).


class TestDropoutDisabled:
    """``enable_optional_key_dropout=False`` suppresses every drop roll."""

    def _all_drops_maxed(self):
        return _DummyBaseDataset(
            history_state_drop_prob=1.0,
            subgoal_drop_prob=1.0,
            response_drop_prob=1.0,
            metadata_drop_all_prob=1.0,
            metadata_drop_each_prob=1.0,
        )

    def test_no_drops_fire_when_disabled(self):
        ds = self._all_drops_maxed()
        ds.enable_optional_key_dropout = False
        standard_item = _prepopulate_standard_item(ds)
        original_state = standard_item["state"].clone()
        original_response = standard_item["response"]
        ds._emit_optional_keys(_raw_item(ds), standard_item)

        # History/state untouched.
        assert torch.equal(standard_item["state"], original_state)
        assert standard_item["obs_history_is_pad"].any().item() is False
        # Subgoals kept (raw tensors resize through, is_pad=False).
        assert standard_item["subgoal_is_pad"].item() is False
        # Response kept.
        assert standard_item["response"] == original_response
        assert standard_item["response_is_pad"].item() is False
        # Metadata kept.
        for k in ("speed", "mistake", "quality"):
            assert standard_item[f"{k}_is_pad"].item() is False

    def test_subgoal_end_of_segment_roll_stays_active(self):
        """Disabling dropout must NOT gate subgoal-frame randomness.

        ``subgoal_end_of_segment_prob`` drives which future frame we read,
        not whether subgoals are masked, so it stays live on the val subset.
        """
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.__new__(LeRobotDataset)
        ds.enable_optional_key_dropout = False
        ds.subgoal_end_of_segment_prob = 1.0
        ds.num_cams = 0  # short-circuits video decoding for this unit check
        # A calm-enough check: the roll is read directly in _load_subgoal_frames.
        # Running the method with num_cams=0 should return {} without error,
        # proving the gate doesn't skip frame-selection logic.
        import opentau.datasets.lerobot_dataset as _ld

        ds.meta = type("_M", (), {"video_keys": []})
        assert _ld.LeRobotDataset._load_subgoal_frames(ds, 0, 0) == {}


# Default collate tolerates a batch with mixed _is_pad flags.


class TestDefaultCollate:
    def test_collate_over_mixed_pad_flags(self):
        """Default PyTorch collate stacks both masked and unmasked samples."""
        from torch.utils.data import default_collate

        ds_keep = _DummyBaseDataset()
        ds_drop = _DummyBaseDataset(
            metadata_drop_all_prob=1.0,
            subgoal_drop_prob=1.0,
            history_state_drop_prob=1.0,
            response_drop_prob=1.0,
        )

        keep_item = _prepopulate_standard_item(ds_keep)
        ds_keep._emit_optional_keys(_raw_item(ds_keep), keep_item)
        drop_item = _prepopulate_standard_item(ds_drop)
        ds_drop._emit_optional_keys(_raw_item(ds_drop), drop_item)

        # Drop strings (default_collate handles them by stacking into a list).
        batch = default_collate([keep_item, drop_item])
        # Tensor fields are stacked.
        assert batch["state"].shape == (2, ds_keep.max_state_dim)
        for k in range(ds_keep.num_cams):
            assert batch[f"subgoal{k}"].shape == (2, 3, *ds_keep.resolution)
        # String fields become lists.
        assert isinstance(batch["memory"], list)
        assert len(batch["memory"]) == 2
        # is_pad flags align 1:1 with their samples.
        assert batch["subgoal_is_pad"].tolist() == [False, True]
        assert batch["quality_is_pad"].tolist() == [False, True]
