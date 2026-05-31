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

from types import SimpleNamespace

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
        emit_fps=False,
        action_freq: float | None = None,
        meta_info: dict | None = None,
        meta_fps: int = 30,
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
        self.emit_fps = emit_fps
        # Mirrors `BaseDataset._action_freq` (mixture-level `action_freq`).
        # When set, `_emit_optional_keys` emits this value as the effective
        # fps instead of `meta.fps` — matching the rate that
        # `resolve_delta_timestamps` resampled the chunk to.
        self._action_freq = action_freq
        self.enable_optional_key_dropout = True
        # Stub meta surface so _to_standard_data_format can read robot_type /
        # control_mode without instantiating a real LeRobotDatasetMetadata.
        # `fps` mirrors `LeRobotDatasetMetadata.fps` so `_emit_optional_keys`
        # can populate the new `fps` / `fps_is_pad` sample keys.
        self.meta = SimpleNamespace(
            info=meta_info if meta_info is not None else {},
            fps=meta_fps,
        )

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
        "speed_raw": 50,
        **{f"subgoal{k}_raw": _make_dummy_subgoal(h, w) for k in range(dataset.num_cams)},
    }


# Legacy path: no _raw fields attached → all optional keys pad out.


class TestLegacyNoAnnotations:
    def test_all_optional_keys_present_and_padded(self):
        ds = _DummyBaseDataset()
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys({}, standard_item)
        # String keys use "" as the pad signal — no separate flag.
        for k in ("memory", "next_memory"):
            assert k in standard_item
            assert standard_item[k] == ""
            assert f"{k}_is_pad" not in standard_item
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

        # String keys: populated value means "not padded".
        assert standard_item["memory"] == "current_mem"
        assert standard_item["next_memory"] == "next_mem"
        # No memory_is_pad / next_memory_is_pad flags — empty-string IS the pad signal.
        assert "memory_is_pad" not in standard_item
        assert "next_memory_is_pad" not in standard_item

        assert standard_item["speed"].item() == 50
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

        # Response not dropped → unchanged (no separate response_is_pad flag).
        assert standard_item["response"] == "hello"
        assert "response_is_pad" not in standard_item


# Force each probability to 1.0 individually.


class TestForcedDropouts:
    def test_history_drop_marks_pad_keeps_state_content(self):
        """history_state_drop marks the history padded but does NOT zero ``state``.

        Zeroing a raw state would normalize to ``-mean/std`` downstream, so the
        dropped history is instead zeroed *after* normalization inside the
        policy. At the dataset level the state content is left intact and only
        ``obs_history_is_pad`` flips to all-True.
        """
        ds = _DummyBaseDataset(history_state_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        original_state = standard_item["state"].clone()
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        assert torch.equal(standard_item["state"], original_state)
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
        # State is NOT zeroed at the dataset level (a raw zero would normalize to
        # -mean/std); the dropped history is zeroed post-normalization in the
        # policy. Every temporal state step stays intact here.
        assert torch.all(standard_item["state"] == 1)

    def test_absent_subgoal_raw_marks_all_slots_padded(self):
        """If ``_load_subgoal_frames`` dropped (returned {}), every subgoal{k}
        tensor is zero-filled and the shared ``subgoal_is_pad`` flag is True.
        """
        ds = _DummyBaseDataset()
        standard_item = _prepopulate_standard_item(ds)
        # Simulate the upstream drop by stripping subgoalK_raw from the item.
        item = {k: v for k, v in _raw_item(ds).items() if not k.startswith("subgoal")}
        ds._emit_optional_keys(item, standard_item)
        assert standard_item["subgoal_is_pad"].item() is True
        for k in range(ds.num_cams):
            assert torch.all(standard_item[f"subgoal{k}"] == 0)

    def test_absent_subgoal_raw_implies_no_response_drop(self):
        """Per spec, response drop is only rolled when subgoals are present."""
        ds = _DummyBaseDataset(response_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        original_response = standard_item["response"]
        # Simulate the upstream subgoal drop.
        item = {k: v for k, v in _raw_item(ds).items() if not k.startswith("subgoal")}
        ds._emit_optional_keys(item, standard_item)
        # Response left intact (still the seed string, not "").
        assert standard_item["response"] == original_response
        assert "response_is_pad" not in standard_item

    def test_response_drop_masks_when_subgoals_present(self):
        ds = _DummyBaseDataset(subgoal_drop_prob=0.0, response_drop_prob=1.0)
        standard_item = _prepopulate_standard_item(ds)
        ds._emit_optional_keys(_raw_item(ds), standard_item)
        # "" is the pad signal; no separate response_is_pad flag.
        assert standard_item["response"] == ""
        assert "response_is_pad" not in standard_item

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
        # Response kept — still the seed string, not masked to "".
        assert standard_item["response"] == original_response
        assert "response_is_pad" not in standard_item
        # Metadata kept.
        for k in ("speed", "mistake", "quality"):
            assert standard_item[f"{k}_is_pad"].item() is False

    def test_subgoal_end_of_segment_roll_stays_active(self, monkeypatch):
        """Disabling dropout must NOT gate subgoal-frame randomness.

        Even with ``subgoal_drop_prob=1.0`` (would drop every sample in
        training), the val path (``enable_optional_key_dropout=False``) still
        decodes a subgoal — and ``subgoal_end_of_segment_prob=1.0`` picks the
        last frame of the current segment.
        """
        from types import SimpleNamespace

        import numpy as _np

        import opentau.datasets.lerobot_dataset as _ld

        mapping_key = "_tests/subgoal_end_of_segment"
        _ld.DATA_FEATURES_NAME_MAPPING[mapping_key] = {"camera0": "camera0"}

        ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
        ds.enable_optional_key_dropout = False
        ds.subgoal_end_of_segment_prob = 1.0
        ds.subgoal_drop_prob = 1.0  # would drop every sample in train mode
        ds.num_cams = 1
        ds.resolution = (8, 8)
        ds.episode_lengths = {0: 100}
        ds.segment_starts_by_episode = {0: _np.array([0])}
        ds.episode_data_index = {
            "from": torch.tensor([0], dtype=torch.long),
            "to": torch.tensor([100], dtype=torch.long),
        }
        ds.epi2idx = {0: 0}
        ds.meta = SimpleNamespace(
            video_keys=["camera0"],
            image_keys=[],
            camera_keys=["camera0"],
            episodes={0: {"segments": [0]}},
            fps=30,
            info={},
        )
        monkeypatch.setattr(type(ds), "_get_feature_mapping_key", lambda self: mapping_key)

        query_calls: list = []

        def _fake_query_videos(self, query_ts, ep_idx):
            query_calls.append((dict(query_ts), ep_idx))
            return {"camera0": torch.zeros((3, *self.resolution))}

        monkeypatch.setattr(type(ds), "_query_videos", _fake_query_videos)

        out = ds._load_subgoal_frames(0, 0)

        assert "subgoal0_raw" in out, "val path should still load subgoals despite subgoal_drop_prob=1.0"
        assert len(query_calls) == 1, "subgoal decode was called exactly once"
        ts_dict, _ep = query_calls[0]
        # End-of-segment pick: last frame of the single segment = ep_length - 1.
        expected_ts = (ds.episode_lengths[0] - 1) / ds.meta.fps
        assert abs(ts_dict["camera0"][0] - expected_ts) < 1e-9

    def test_subgoal_drop_skips_video_decode_in_train_mode(self, monkeypatch):
        """In train mode with ``subgoal_drop_prob=1.0`` we must NOT call
        ``_query_videos`` — the whole point of rolling the drop upstream is
        to avoid wasted video decoding.
        """
        from types import SimpleNamespace

        import numpy as _np

        import opentau.datasets.lerobot_dataset as _ld

        mapping_key = "_tests/subgoal_drop_skip_decode"
        _ld.DATA_FEATURES_NAME_MAPPING[mapping_key] = {"camera0": "camera0"}

        ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
        ds.enable_optional_key_dropout = True  # train mode
        ds.subgoal_end_of_segment_prob = 0.0
        ds.subgoal_drop_prob = 1.0  # always drop
        ds.num_cams = 1
        ds.resolution = (8, 8)
        ds.episode_lengths = {0: 100}
        ds.segment_starts_by_episode = {0: _np.array([0])}
        ds.meta = SimpleNamespace(
            video_keys=["camera0"],
            image_keys=[],
            camera_keys=["camera0"],
            episodes={0: {"segments": [0]}},
            fps=30,
            info={},
        )
        monkeypatch.setattr(type(ds), "_get_feature_mapping_key", lambda self: mapping_key)

        def _fake_query_videos(self, query_ts, ep_idx):
            raise AssertionError("_query_videos should not be called when dropping subgoals")

        monkeypatch.setattr(type(ds), "_query_videos", _fake_query_videos)

        out = ds._load_subgoal_frames(0, 0)
        assert out == {}, "drop should return an empty dict and skip decoding"

    def test_no_cameras_returns_empty(self, monkeypatch):
        """Datasets with no camera keys still short-circuit to ``{}`` so
        :meth:`BaseDataset._emit_optional_keys` can mark every subgoal slot
        padded.
        """
        from types import SimpleNamespace

        import numpy as _np

        import opentau.datasets.lerobot_dataset as _ld

        mapping_key = "_tests/subgoal_no_cameras"
        _ld.DATA_FEATURES_NAME_MAPPING[mapping_key] = {"camera0": "camera0"}

        ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
        ds.enable_optional_key_dropout = False
        ds.subgoal_end_of_segment_prob = 1.0
        ds.subgoal_drop_prob = 0.0
        ds.num_cams = 0
        ds.resolution = (8, 8)
        ds.episode_lengths = {0: 100}
        ds.segment_starts_by_episode = {0: _np.array([0])}
        ds.meta = SimpleNamespace(
            video_keys=[],
            image_keys=[],
            camera_keys=[],
            episodes={0: {"segments": [0]}},
            fps=30,
            info={},
        )
        monkeypatch.setattr(type(ds), "_get_feature_mapping_key", lambda self: mapping_key)

        def _fail_query_videos(self, query_ts, ep_idx):
            raise AssertionError("_query_videos must not be called when no cameras are present")

        monkeypatch.setattr(type(ds), "_query_videos", _fail_query_videos)

        assert ds._load_subgoal_frames(0, 0) == {}

    def test_subgoals_load_without_info_subgoals_key(self, monkeypatch):
        """Always-on behavior: info.json with no ``subgoals`` key still
        triggers subgoal loading from the camera streams.

        Pins the deliberate removal of the prior opt-in gate so a future
        refactor doesn't silently restore it.
        """
        from types import SimpleNamespace

        import numpy as _np

        import opentau.datasets.lerobot_dataset as _ld

        mapping_key = "_tests/subgoal_no_info_key"
        _ld.DATA_FEATURES_NAME_MAPPING[mapping_key] = {"camera0": "camera0"}

        ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
        ds.enable_optional_key_dropout = False
        ds.subgoal_end_of_segment_prob = 0.0
        ds.subgoal_drop_prob = 0.0
        ds.num_cams = 1
        ds.resolution = (8, 8)
        ds.episode_lengths = {0: 100}
        ds.segment_starts_by_episode = {0: _np.array([0])}
        ds.episode_data_index = {
            "from": torch.tensor([0], dtype=torch.long),
            "to": torch.tensor([100], dtype=torch.long),
        }
        ds.epi2idx = {0: 0}
        # info.json deliberately omits "subgoals" — mirrors every legacy
        # LeRobot dataset that pre-dates this PR.
        ds.meta = SimpleNamespace(
            video_keys=["camera0"],
            image_keys=[],
            camera_keys=["camera0"],
            episodes={0: {"segments": [0]}},
            fps=30,
            info={},
        )
        monkeypatch.setattr(type(ds), "_get_feature_mapping_key", lambda self: mapping_key)

        called: list = []

        def _fake_query_videos(self, query_ts, ep_idx):
            called.append(dict(query_ts))
            return {"camera0": torch.zeros((3, *self.resolution))}

        monkeypatch.setattr(type(ds), "_query_videos", _fake_query_videos)

        out = ds._load_subgoal_frames(0, 0)
        assert "subgoal0_raw" in out
        assert len(called) == 1, "video decode fired exactly once for the single camera"

    def test_image_dtype_fallback_uses_absolute_row_index(self, monkeypatch):
        """``image``-dtype cameras read from the parquet rows of
        ``hf_dataset`` instead of decoding a video. The lookup index must be
        ``ep_start + subgoal_frame`` — the absolute row in the table — not
        the within-episode index.
        """
        from types import SimpleNamespace

        import numpy as _np

        import opentau.datasets.lerobot_dataset as _ld

        mapping_key = "_tests/subgoal_image_dtype"
        _ld.DATA_FEATURES_NAME_MAPPING[mapping_key] = {"camera0": "camera0"}

        ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
        ds.enable_optional_key_dropout = False
        ds.subgoal_end_of_segment_prob = 0.0
        ds.subgoal_drop_prob = 0.0
        ds.num_cams = 1
        ds.resolution = (8, 8)
        # Episode 1 starts at absolute row 100, length 50.
        ep_start_abs = 100
        ep_len = 50
        ds.episode_lengths = {1: ep_len}
        ds.segment_starts_by_episode = {1: _np.array([0])}
        ds.episode_data_index = {
            "from": torch.tensor([0, ep_start_abs], dtype=torch.long),
            "to": torch.tensor([ep_start_abs, ep_start_abs + ep_len], dtype=torch.long),
        }
        ds.epi2idx = {0: 0, 1: 1}
        ds.meta = SimpleNamespace(
            video_keys=[],
            image_keys=["camera0"],
            camera_keys=["camera0"],
            episodes={1: {"segments": [0]}},
            fps=10,
            info={},
        )
        monkeypatch.setattr(type(ds), "_get_feature_mapping_key", lambda self: mapping_key)

        # Pin the within-episode subgoal index so the test asserts a known
        # absolute row.
        within_ep_subgoal = 7

        def _fake_sample(self, ep_idx, frame_in_ep, *, at_end_of_segment):
            assert ep_idx == 1 and frame_in_ep == 0
            return within_ep_subgoal

        monkeypatch.setattr(type(ds), "_sample_subgoal_frame", _fake_sample)

        # Stub hf_dataset to capture the row index requested and return a
        # sentinel tensor identifying that row.
        sentinel_payload = torch.full((3, 8, 8), 0.42, dtype=torch.float32)
        hf_calls: list[int] = []

        class _HFStub:
            def __getitem__(self, idx):
                hf_calls.append(idx)
                return {"camera0": sentinel_payload}

        ds.hf_dataset = _HFStub()

        def _fail_query_videos(self, query_ts, ep_idx):
            raise AssertionError("_query_videos must not be called for image-dtype cameras")

        monkeypatch.setattr(type(ds), "_query_videos", _fail_query_videos)

        out = ds._load_subgoal_frames(1, 0)

        assert hf_calls == [ep_start_abs + within_ep_subgoal], (
            f"image-dtype subgoal lookup used the wrong row: got {hf_calls}, "
            f"expected [{ep_start_abs + within_ep_subgoal}]"
        )
        assert "subgoal0_raw" in out
        assert torch.equal(out["subgoal0_raw"], sentinel_payload)


# Default collate tolerates a batch with mixed _is_pad flags.


class TestDefaultCollate:
    def test_collate_over_mixed_pad_flags(self):
        """Default PyTorch collate stacks both masked and unmasked samples."""
        from torch.utils.data import default_collate

        ds_keep = _DummyBaseDataset()
        ds_drop = _DummyBaseDataset(
            metadata_drop_all_prob=1.0,
            history_state_drop_prob=1.0,
            response_drop_prob=1.0,
        )

        keep_item = _prepopulate_standard_item(ds_keep)
        ds_keep._emit_optional_keys(_raw_item(ds_keep), keep_item)
        drop_item = _prepopulate_standard_item(ds_drop)
        # Simulate `_load_subgoal_frames` dropping by omitting subgoal_raw keys.
        item = {k: v for k, v in _raw_item(ds_drop).items() if not k.startswith("subgoal")}
        ds_drop._emit_optional_keys(item, drop_item)

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


# Dataset-level identifiers (robot_type, control_mode) are emitted inside
# _emit_optional_keys and participate in the metadata_drop_all_prob /
# metadata_drop_each_prob dropout rolls, same as speed / mistake / quality.


def _full_raw_item(dataset: _DummyBaseDataset) -> dict:
    """Minimal raw item that ``_to_standard_data_format`` can consume end-to-end."""
    h, w = dataset.resolution
    return {
        "camera0": torch.full((3, h, w), 0.5, dtype=torch.float32),
        "state": torch.zeros(dataset.max_state_dim, dtype=torch.float32),
        "actions": torch.zeros((dataset.action_chunk, dataset.max_action_dim), dtype=torch.float32),
        "actions_is_pad": torch.zeros(dataset.action_chunk, dtype=torch.bool),
        "prompt": "do the thing",
        "response": "ok",
    }


class TestRobotTypeControlMode:
    def test_both_present_pass_through(self):
        ds = _DummyBaseDataset(
            num_cams=1,
            meta_info={"robot_type": "aloha", "control_mode": "joint"},
        )
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["robot_type"] == "aloha"
        assert out["control_mode"] == "joint"

    def test_both_absent_emit_empty_string(self):
        # Default _DummyBaseDataset has empty meta.info — mirrors VQA datasets and
        # legacy LeRobot datasets that pre-date PR #183.
        ds = _DummyBaseDataset(num_cams=1)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["robot_type"] == ""
        assert out["control_mode"] == ""

    def test_robot_type_none_emit_empty_string(self):
        # `info["robot_type"]` is typed `str | None`; both None and missing must
        # collapse to the same "" sentinel so the downstream contract is uniform.
        ds = _DummyBaseDataset(num_cams=1, meta_info={"robot_type": None})
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["robot_type"] == ""

    def test_dropped_when_metadata_drop_all_prob_is_one(self):
        # When metadata_drop_all_prob=1.0, robot_type / control_mode collapse to
        # "" — they participate in the same metadata drop group as speed/mistake.
        ds = _DummyBaseDataset(
            num_cams=1,
            meta_info={"robot_type": "panda", "control_mode": "ee"},
            history_state_drop_prob=1.0,
            subgoal_drop_prob=1.0,
            response_drop_prob=1.0,
            metadata_drop_all_prob=1.0,
            metadata_drop_each_prob=1.0,
        )
        assert ds.enable_optional_key_dropout is True
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["robot_type"] == ""
        assert out["control_mode"] == ""

    def test_pass_through_when_drop_probs_zero(self):
        # With all drop probabilities at zero, robot_type / control_mode pass
        # through unchanged even when dropout is enabled.
        ds = _DummyBaseDataset(
            num_cams=1,
            meta_info={"robot_type": "panda", "control_mode": "ee"},
            metadata_drop_all_prob=0.0,
            metadata_drop_each_prob=0.0,
        )
        assert ds.enable_optional_key_dropout is True
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["robot_type"] == "panda"
        assert out["control_mode"] == "ee"

    def test_emitted_as_python_strings_for_default_collate(self):
        # Default PyTorch collate batches str fields into list[str]; verify the
        # emitted values are plain strings (not bytes / np.str_ / 0-dim tensors).
        ds = _DummyBaseDataset(
            num_cams=1,
            meta_info={"robot_type": "human", "control_mode": "mixed"},
        )
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert type(out["robot_type"]) is str
        assert type(out["control_mode"]) is str


# `fps` is emitted as an intrinsic dataset property — always present (non-pad)
# when `emit_fps=True`, omitted entirely when `emit_fps=False` (the default).
# Unlike speed / quality / mistake / robot_type / control_mode it does NOT
# participate in any of the `metadata_drop_*_prob` dropout rolls.


class TestFpsEmission:
    def test_fps_emitted_with_meta_fps_value(self):
        ds = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=20)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].dtype == torch.long
        assert out["fps"].item() == 20
        assert out["fps_is_pad"].item() is False

    def test_fps_emitted_as_torch_long_dtype(self):
        ds = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=50)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].dtype == torch.long

    def test_fps_omitted_when_emit_fps_false(self):
        # The default — pre-PR behaviour preserved.
        ds = _DummyBaseDataset(num_cams=1, emit_fps=False, meta_fps=20)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert "fps" not in out
        assert "fps_is_pad" not in out

    def test_fps_omitted_by_default(self):
        """The fixture default mirrors production: ``emit_fps=False`` ⇒ no fps keys."""
        ds = _DummyBaseDataset(num_cams=1, meta_fps=20)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert "fps" not in out
        assert "fps_is_pad" not in out

    def test_fps_unaffected_by_metadata_drop_all_prob(self):
        # fps is intrinsic to the dataset, not a noisy label — it must NOT
        # participate in `metadata_drop_*_prob` rolls.
        ds = _DummyBaseDataset(
            num_cams=1,
            emit_fps=True,
            meta_fps=15,
            metadata_drop_all_prob=1.0,
            metadata_drop_each_prob=1.0,
        )
        out = ds._to_standard_data_format(_full_raw_item(ds))
        # Other metadata fields are dropped — but fps stays.
        assert out["speed_is_pad"].item() is True
        assert out["quality_is_pad"].item() is True
        assert out["mistake_is_pad"].item() is True
        assert out["robot_type"] == ""
        assert out["control_mode"] == ""
        assert out["fps"].item() == 15
        assert out["fps_is_pad"].item() is False

    def test_fps_unaffected_by_disabled_dropout(self):
        # `enable_optional_key_dropout=False` (the val-subset path) must keep
        # the same emit behavior — fps is gated by `emit_fps`, not by dropout.
        ds = _DummyBaseDataset(
            num_cams=1,
            emit_fps=True,
            meta_fps=25,
            metadata_drop_all_prob=1.0,
            metadata_drop_each_prob=1.0,
        )
        ds.enable_optional_key_dropout = False
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].item() == 25
        assert out["fps_is_pad"].item() is False

    def test_default_collate_stacks_mixed_fps(self):
        """Heterogeneous-fps batches (different datasets in a mixture) must
        collate cleanly: the per-sample `fps` tensors become a (B,) long
        tensor and `fps_is_pad` becomes a (B,) bool tensor."""
        from torch.utils.data import default_collate

        ds_a = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=30)
        ds_b = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=50)
        item_a = ds_a._to_standard_data_format(_full_raw_item(ds_a))
        item_b = ds_b._to_standard_data_format(_full_raw_item(ds_b))
        batch = default_collate([item_a, item_b])
        assert batch["fps"].dtype == torch.long
        assert batch["fps"].tolist() == [30, 50]
        assert batch["fps_is_pad"].tolist() == [False, False]

    def test_fps_reports_action_freq_when_resampling(self):
        """When the mixture sets `action_freq`, every chunk is nearest-neighbor
        resampled to that rate by `resolve_delta_timestamps`. The emitted
        `fps` must reflect the *effective* rate of the resampled chunk, not
        the dataset's native rate — otherwise the prefix tells the model a
        different timing than the chunk actually carries.
        """
        # Dataset is natively 50 Hz, but the mixture pins everything to 30 Hz.
        ds = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=50, action_freq=30.0)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].item() == 30
        assert out["fps_is_pad"].item() is False

    def test_fps_reports_native_when_action_freq_none(self):
        """When `action_freq is None` (no resampling), the chunk runs at the
        dataset's native rate — that's what gets emitted."""
        ds = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=50, action_freq=None)
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].item() == 50
        assert out["fps_is_pad"].item() is False


# VQA datasets have no temporal axis — `DatasetMetadata.fps` returns ``None``
# on the base class (and on ``VQADatasetMetadata``, which inherits via ``pass``).
# `_emit_optional_keys` must catch that and emit ``fps=0, fps_is_pad=True``
# instead of crashing on ``int(None)`` — otherwise any heterogeneous mixture
# with ``emit_fps=True`` crashes at the first VQA sample fetch. (Under the
# default ``emit_fps=False`` the path is a no-op; these tests opt in to
# exercise the heterogeneous-batch behaviour explicitly.)


class TestFpsEmissionVQA:
    def test_base_metadata_fps_returns_none(self):
        """``DatasetMetadata.fps`` is ``None`` for non-LeRobot datasets."""
        from opentau.datasets.lerobot_dataset import DatasetMetadata, VQADatasetMetadata

        assert DatasetMetadata().fps is None
        assert VQADatasetMetadata().fps is None

    def test_vqa_metadata_emits_pad_row(self):
        """A dataset whose ``meta.fps`` is ``None`` (i.e. a VQA dataset) must
        still produce a complete sample dict so heterogeneous batches stay
        schema-aligned. The fps row is padded out as
        ``fps=0, fps_is_pad=True`` (mirroring the speed/quality/mistake pad
        shape) — the policy's ``prepare_metadata`` then drops the ``FPS:``
        segment for that sample.
        """
        from opentau.datasets.lerobot_dataset import VQADatasetMetadata

        ds = _DummyBaseDataset(num_cams=1, emit_fps=True)
        ds.meta = VQADatasetMetadata(info={"features": {}})
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert "fps" in out and "fps_is_pad" in out
        assert out["fps"].dtype == torch.long
        assert out["fps"].item() == 0
        assert out["fps_is_pad"].item() is True

    def test_vqa_pad_ignores_action_freq(self):
        """Even when the mixture sets ``action_freq``, a VQA sample emits
        pad — VQA has no actions to be at any rate, so reporting
        ``action_freq`` would be misleading."""
        from opentau.datasets.lerobot_dataset import VQADatasetMetadata

        ds = _DummyBaseDataset(num_cams=1, emit_fps=True, action_freq=30.0)
        ds.meta = VQADatasetMetadata(info={"features": {}})
        out = ds._to_standard_data_format(_full_raw_item(ds))
        assert out["fps"].item() == 0
        assert out["fps_is_pad"].item() is True

    def test_heterogeneous_vla_vqa_collate(self):
        """The pad-row trick keeps batches schema-aligned across a
        VLA + VQA mixture: default_collate stacks fps into a ``(B,)`` long
        tensor and fps_is_pad into a ``(B,)`` bool tensor with the
        per-sample on/off correctly set.
        """
        from torch.utils.data import default_collate

        from opentau.datasets.lerobot_dataset import VQADatasetMetadata

        # LeRobot-shaped sample.
        ds_vla = _DummyBaseDataset(num_cams=1, emit_fps=True, meta_fps=30)
        item_vla = ds_vla._to_standard_data_format(_full_raw_item(ds_vla))

        # VQA-shaped sample (same dummy class, real VQA metadata).
        ds_vqa = _DummyBaseDataset(num_cams=1, emit_fps=True)
        ds_vqa.meta = VQADatasetMetadata(info={"features": {}})
        item_vqa = ds_vqa._to_standard_data_format(_full_raw_item(ds_vqa))

        batch = default_collate([item_vla, item_vqa])
        assert batch["fps"].dtype == torch.long
        assert batch["fps"].tolist() == [30, 0]
        assert batch["fps_is_pad"].tolist() == [False, True]
