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

"""Tests for observation history support (n_obs_history and history_interval)."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import resolve_delta_timestamps
from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from opentau.datasets.transforms import ImageTransformsConfig
from tests.fixtures.constants import DUMMY_REPO_ID


@dataclass
class DummyPolicyConfig(PreTrainedConfig):
    chunk_size: int = 50

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestDatasetMixtureConfigValidation:
    def test_n_obs_history_none_is_default(self):
        cfg = DatasetMixtureConfig()
        assert cfg.n_obs_history is None
        assert cfg.history_interval == 1

    def test_n_obs_history_positive_int_accepted(self):
        cfg = DatasetMixtureConfig(n_obs_history=3)
        assert cfg.n_obs_history == 3

    def test_n_obs_history_one_accepted(self):
        cfg = DatasetMixtureConfig(n_obs_history=1)
        assert cfg.n_obs_history == 1

    def test_n_obs_history_zero_rejected(self):
        with pytest.raises(ValueError, match="n_obs_history"):
            DatasetMixtureConfig(n_obs_history=0)

    def test_n_obs_history_negative_rejected(self):
        with pytest.raises(ValueError, match="n_obs_history"):
            DatasetMixtureConfig(n_obs_history=-1)

    def test_n_obs_history_float_rejected(self):
        with pytest.raises(ValueError, match="n_obs_history"):
            DatasetMixtureConfig(n_obs_history=2.5)

    def test_history_interval_positive_int_accepted(self):
        cfg = DatasetMixtureConfig(n_obs_history=3, history_interval=2)
        assert cfg.history_interval == 2

    def test_history_interval_zero_rejected(self):
        with pytest.raises(ValueError, match="history_interval"):
            DatasetMixtureConfig(history_interval=0)

    def test_history_interval_negative_rejected(self):
        with pytest.raises(ValueError, match="history_interval"):
            DatasetMixtureConfig(history_interval=-1)


# ---------------------------------------------------------------------------
# resolve_delta_timestamps tests
# ---------------------------------------------------------------------------


def _make_train_cfg(n_obs_history=None, history_interval=1, action_freq=30.0):
    transforms_cfg = ImageTransformsConfig(enable=False)
    dataset_cfg = DatasetConfig(
        repo_id="mock_dataset",
        root="/tmp/mock",
        image_transforms=transforms_cfg,
        episodes=[0],
        video_backend=None,
    )
    mixture_cfg = DatasetMixtureConfig(
        datasets=[dataset_cfg],
        weights=[1.0],
        action_freq=action_freq,
        n_obs_history=n_obs_history,
        history_interval=history_interval,
    )
    policy_cfg = DummyPolicyConfig()
    return TrainPipelineConfig(
        dataset_mixture=mixture_cfg,
        policy=policy_cfg,
        batch_size=8,
    ), dataset_cfg


def _make_metadata(features):
    meta = MagicMock()
    meta.features = features
    return meta


class TestResolveDeltaTimestampsHistory:
    def test_no_history_single_delta(self):
        """Without n_obs_history, camera/state get [0.0]."""
        cfg, ds_cfg = _make_train_cfg(n_obs_history=None)
        meta = _make_metadata({"camera0": {}, "state": {}})
        dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
        np.testing.assert_array_equal(dt_mean["camera0"], [0.0])
        np.testing.assert_array_equal(dt_mean["state"], [0.0])

    def test_history_t3_k1(self):
        """n_obs_history=3, history_interval=1 at 30Hz."""
        cfg, ds_cfg = _make_train_cfg(n_obs_history=3, history_interval=1, action_freq=30.0)
        meta = _make_metadata({"camera0": {}, "state": {}})
        dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
        expected = np.array([-2 / 30, -1 / 30, 0.0])
        np.testing.assert_allclose(dt_mean["camera0"], expected)
        np.testing.assert_allclose(dt_mean["state"], expected)

    def test_history_t3_k2(self):
        """n_obs_history=3, history_interval=2 at 30Hz."""
        cfg, ds_cfg = _make_train_cfg(n_obs_history=3, history_interval=2, action_freq=30.0)
        meta = _make_metadata({"camera0": {}, "state": {}})
        dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
        expected = np.array([-4 / 30, -2 / 30, 0.0])
        np.testing.assert_allclose(dt_mean["camera0"], expected)
        np.testing.assert_allclose(dt_mean["state"], expected)

    def test_history_t1_single_delta(self):
        """n_obs_history=1 should produce [0.0] (same as no history)."""
        cfg, ds_cfg = _make_train_cfg(n_obs_history=1, history_interval=1)
        meta = _make_metadata({"state": {}})
        dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
        np.testing.assert_array_equal(dt_mean["state"], [0.0])

    def test_history_does_not_affect_actions(self):
        """Action deltas come from policy config, not n_obs_history."""
        cfg, ds_cfg = _make_train_cfg(n_obs_history=3, history_interval=1)
        meta = _make_metadata({"camera0": {}, "state": {}, "actions": {}})
        dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
        assert len(dt_mean["actions"]) == cfg.policy.chunk_size
        assert len(dt_mean["camera0"]) == 3
        assert len(dt_mean["state"]) == 3

    def test_history_last_element_is_zero(self):
        """The last delta timestamp should always be 0.0 (current time)."""
        for t in [2, 5, 10]:
            cfg, ds_cfg = _make_train_cfg(n_obs_history=t, history_interval=3)
            meta = _make_metadata({"state": {}})
            dt_mean, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, meta)
            assert dt_mean["state"][-1] == 0.0
            assert len(dt_mean["state"]) == t


# ---------------------------------------------------------------------------
# Integration tests: __getitem__ with observation history
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fix_dummy_mapping():
    """Fix DUMMY_REPO_ID mapping so standardized __getitem__ works."""
    original = DATA_FEATURES_NAME_MAPPING.get(DUMMY_REPO_ID, {}).copy()
    DATA_FEATURES_NAME_MAPPING[DUMMY_REPO_ID] = {
        "state": "state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    }
    yield
    DATA_FEATURES_NAME_MAPPING[DUMMY_REPO_ID] = original


def _make_dataset_with_history(
    lerobot_dataset_factory, info_factory, tmp_path, n_obs_history, history_interval=1, suffix=""
):
    """Create a LeRobotDataset with observation history configured.

    Uses camera_features={} to avoid video decoding in tests.
    """
    from opentau.datasets.lerobot_dataset import LeRobotDataset

    info = info_factory(
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
        camera_features={},
    )

    dataset = lerobot_dataset_factory(
        root=tmp_path / f"obs_hist_{n_obs_history}_{history_interval}{suffix}",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
        info=info,
    )

    # Patch config with desired history settings and a policy with action_delta_indices
    dataset.cfg.dataset_mixture.n_obs_history = n_obs_history
    dataset.cfg.dataset_mixture.history_interval = history_interval
    dataset.n_obs_history = n_obs_history
    dataset.cfg.policy = DummyPolicyConfig(chunk_size=dataset.cfg.action_chunk)

    dt_info = resolve_delta_timestamps(dataset.cfg, dataset.cfg.dataset_mixture.datasets[0], dataset.meta)
    dataset.delta_timestamps_params = LeRobotDataset.compute_delta_params(*dt_info)

    return dataset


def test_default_no_history_shapes(lerobot_dataset_factory, info_factory, tmp_path):
    """n_obs_history=None preserves single-step shapes and adds obs_history_is_pad."""
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=None)
    item = dataset[25]
    assert item["state"].shape == (dataset.cfg.max_state_dim,)
    assert "obs_history_is_pad" in item
    assert item["obs_history_is_pad"].shape == (1,)
    assert item["obs_history_is_pad"].dtype == torch.bool
    assert not item["obs_history_is_pad"].any()


def test_history_t3_state_shape(lerobot_dataset_factory, info_factory, tmp_path):
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3)
    item = dataset[25]
    assert item["state"].shape == (3, dataset.cfg.max_state_dim)


def test_history_t3_obs_history_is_pad_shape(lerobot_dataset_factory, info_factory, tmp_path):
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3)
    item = dataset[25]
    assert "obs_history_is_pad" in item
    assert item["obs_history_is_pad"].shape == (3,)
    assert item["obs_history_is_pad"].dtype == torch.bool


def test_history_t3_obs_history_is_pad_mid_episode(lerobot_dataset_factory, info_factory, tmp_path):
    """In the middle of an episode, no observation should be padded."""
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3)
    ep_from = int(dataset.episode_data_index["from"][0].item())
    mid_idx = ep_from + 10
    item = dataset[mid_idx]
    assert not item["obs_history_is_pad"].any(), "Mid-episode observations should not be padded"


def test_history_padding_at_episode_start(lerobot_dataset_factory, info_factory, tmp_path):
    """At the start of an episode, earlier history frames should be padded."""
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=5)
    ep_from = int(dataset.episode_data_index["from"][0].item())
    item = dataset[ep_from]
    assert item["obs_history_is_pad"][0].item() is True, "Earliest history frame at ep start should be padded"
    assert item["obs_history_is_pad"][-1].item() is False, "Current frame should never be padded"


def test_history_interval_k2_padding(lerobot_dataset_factory, info_factory, tmp_path):
    """With history_interval=2, padding extends further at episode start."""
    dataset = _make_dataset_with_history(
        lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3, history_interval=2
    )
    ep_from = int(dataset.episode_data_index["from"][0].item())
    item = dataset[ep_from + 1]
    assert item["obs_history_is_pad"][0].item() is True
    assert item["obs_history_is_pad"][1].item() is True
    assert item["obs_history_is_pad"][2].item() is False


def test_history_actions_unchanged(lerobot_dataset_factory, info_factory, tmp_path):
    """Action shape should be unchanged regardless of n_obs_history."""
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3)
    item = dataset[25]
    assert item["actions"].shape == (dataset.cfg.action_chunk, dataset.cfg.max_action_dim)
    assert item["action_is_pad"].shape == (dataset.cfg.action_chunk,)


def test_history_img_is_pad_unchanged(lerobot_dataset_factory, info_factory, tmp_path):
    """img_is_pad should remain (num_cams,) regardless of n_obs_history."""
    dataset = _make_dataset_with_history(lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=3)
    item = dataset[25]
    assert item["img_is_pad"].shape == (dataset.cfg.num_cams,)


def test_obs_history_is_pad_fallback_shape(lerobot_dataset_factory, info_factory, tmp_path):
    """When state_is_pad is absent from item, obs_history_is_pad fallback must match (T,) shape."""
    dataset = _make_dataset_with_history(
        lerobot_dataset_factory, info_factory, tmp_path, n_obs_history=4, suffix="_no_state"
    )

    dt_mean, dt_std, dt_lower, dt_upper = dataset.delta_timestamps_params
    dt_mean = {k: v for k, v in dt_mean.items() if k != "state"}
    dt_std = {k: v for k, v in dt_std.items() if k != "state"}
    dt_lower = {k: v for k, v in dt_lower.items() if k != "state"}
    dt_upper = {k: v for k, v in dt_upper.items() if k != "state"}
    dataset.delta_timestamps_params = (dt_mean, dt_std, dt_lower, dt_upper)

    item = dataset[25]
    assert "obs_history_is_pad" in item
    assert item["obs_history_is_pad"].shape == (4,), (
        f"Expected fallback shape (4,), got {item['obs_history_is_pad'].shape}"
    )
    assert item["obs_history_is_pad"].dtype == torch.bool
    assert not item["obs_history_is_pad"].any()


# ---------------------------------------------------------------------------
# Batched resize_with_pad tests
# ---------------------------------------------------------------------------


class TestResizeWithPadBatch:
    """Verify that batched (T,C,H,W) resize_with_pad matches per-frame results."""

    @staticmethod
    def _resize(img, width, height, pad_value=0):
        return BaseDataset.resize_with_pad(None, img, width, height, pad_value)

    @pytest.mark.parametrize(
        "src_hw, target_hw",
        [
            ((64, 64), (48, 48)),  # square -> square
            ((120, 80), (48, 64)),  # portrait -> landscape
            ((80, 120), (64, 48)),  # landscape -> portrait
            ((100, 200), (48, 48)),  # wide -> square
            ((200, 100), (48, 48)),  # tall -> square
        ],
        ids=["square", "portrait-to-landscape", "landscape-to-portrait", "wide-to-square", "tall-to-square"],
    )
    @pytest.mark.parametrize("T", [1, 3, 7])
    def test_batch_matches_per_frame(self, src_hw, target_hw, T):  # noqa: N803
        C = 3  # noqa: N806
        h, w = src_hw
        target_h, target_w = target_hw
        imgs = torch.rand(T, C, h, w)

        batched_result = self._resize(imgs, target_w, target_h)

        per_frame = torch.stack(
            [self._resize(imgs[t], target_w, target_h) for t in range(T)],
            dim=0,
        )

        assert batched_result.shape == (T, C, target_h, target_w)
        assert per_frame.shape == (T, C, target_h, target_w)
        assert torch.allclose(batched_result, per_frame), (
            f"Max diff: {(batched_result - per_frame).abs().max().item()}"
        )

    def test_single_frame_output_is_3d(self):
        img = torch.rand(3, 100, 80)
        result = self._resize(img, 48, 64)
        assert result.shape == (3, 64, 48)

    def test_batched_output_is_4d(self):
        imgs = torch.rand(5, 3, 100, 80)
        result = self._resize(imgs, 48, 64)
        assert result.shape == (5, 3, 64, 48)

    def test_t1_batched_stays_4d(self):
        imgs = torch.rand(1, 3, 100, 80)
        result = self._resize(imgs, 48, 64)
        assert result.ndim == 4
        assert result.shape == (1, 3, 64, 48)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            self._resize(torch.rand(100, 80), 48, 64)

    def test_pad_value_propagated(self):
        img = torch.zeros(3, 100, 200)
        result = self._resize(img, 48, 48, pad_value=1.0)
        assert result.max() == 1.0, "Padding region should contain pad_value"
