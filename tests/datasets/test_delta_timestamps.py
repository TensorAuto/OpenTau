# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections import defaultdict
from itertools import accumulate
from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import pyarrow.compute as pc
import pytest
import torch

from opentau.datasets.factory import resolve_delta_timestamps
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.utils import check_timestamps_sync
from tests.fixtures.constants import DUMMY_MOTOR_FEATURES


def calculate_total_episode(
    hf_dataset: datasets.Dataset, raise_if_not_contiguous: bool = True
) -> dict[str, torch.Tensor]:
    episode_indices = sorted(hf_dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    if raise_if_not_contiguous and episode_indices != list(range(total_episodes)):
        raise ValueError("episode_index values are not sorted and contiguous.")
    return total_episodes


def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> dict[str, np.ndarray]:
    episode_lengths = []
    table = hf_dataset.data.table
    total_episodes = calculate_total_episode(hf_dataset)
    for ep_idx in range(total_episodes):
        ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
        episode_lengths.insert(ep_idx, len(ep_table))

    cumulative_lengths = list(accumulate(episode_lengths))
    return {
        "from": np.array([0] + cumulative_lengths[:-1], dtype=np.int64),
        "to": np.array(cumulative_lengths, dtype=np.int64),
    }


@pytest.fixture(scope="module")
def synced_timestamps_factory(hf_dataset_factory):
    def _create_synced_timestamps(fps: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hf_dataset = hf_dataset_factory(fps=fps)
        timestamps = torch.stack(list(hf_dataset["timestamp"])).numpy()
        episode_indices = torch.stack(list(hf_dataset["episode_index"])).numpy()
        episode_data_index = calculate_episode_data_index(hf_dataset)
        return timestamps, episode_indices, episode_data_index

    return _create_synced_timestamps


@pytest.fixture(scope="module")
def unsynced_timestamps_factory(synced_timestamps_factory):
    def _create_unsynced_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps, episode_indices, episode_data_index = synced_timestamps_factory(fps=fps)
        timestamps[30] += tolerance_s * 1.1  # Modify a single timestamp just outside tolerance
        return timestamps, episode_indices, episode_data_index

    return _create_unsynced_timestamps


@pytest.fixture(scope="module")
def slightly_off_timestamps_factory(synced_timestamps_factory):
    def _create_slightly_off_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps, episode_indices, episode_data_index = synced_timestamps_factory(fps=fps)
        timestamps[30] += tolerance_s * 0.9  # Modify a single timestamp just inside tolerance
        return timestamps, episode_indices, episode_data_index

    return _create_slightly_off_timestamps


@pytest.fixture(scope="module")
def valid_delta_timestamps_factory():
    def _create_valid_delta_timestamps(
        fps: int = 30, keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)
    ) -> dict:
        delta_timestamps = {key: [i * (1 / fps) for i in range(*min_max_range)] for key in keys}
        return delta_timestamps

    return _create_valid_delta_timestamps


@pytest.fixture(scope="module")
def invalid_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_invalid_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just outside tolerance
        for key in keys:
            delta_timestamps[key][3] += tolerance_s * 1.1
        return delta_timestamps

    return _create_invalid_delta_timestamps


@pytest.fixture(scope="module")
def slightly_off_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_slightly_off_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just inside tolerance
        for key in delta_timestamps:
            delta_timestamps[key][3] += tolerance_s * 0.9
            delta_timestamps[key][-3] += tolerance_s * 0.9
        return delta_timestamps

    return _create_slightly_off_delta_timestamps


@pytest.fixture(scope="module")
def delta_indices_factory():
    def _delta_indices(keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)) -> dict:
        return {key: list(range(*min_max_range)) for key in keys}

    return _delta_indices


def test_check_timestamps_sync_synced(synced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = synced_timestamps_factory(fps)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_unsynced(unsynced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = unsynced_timestamps_factory(fps, tolerance_s)
    with pytest.raises(ValueError):
        check_timestamps_sync(
            timestamps=timestamps,
            episode_indices=ep_idx,
            episode_data_index=ep_data_index,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_timestamps_sync_unsynced_no_exception(unsynced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = unsynced_timestamps_factory(fps, tolerance_s)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_timestamps_sync_slightly_off(slightly_off_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = slightly_off_timestamps_factory(fps, tolerance_s)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_single_timestamp():
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx = np.array([0.0]), np.array([0])
    episode_data_index = {"to": np.array([1]), "from": np.array([0])}
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_resolve_delta_timestamps_basic_structure(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that resolve_delta_timestamps returns the correct basic structure."""
    result = resolve_delta_timestamps(train_pipeline_config, dataset_config, lerobot_dataset_metadata)

    # Check that result is a tuple with 2 elements
    assert isinstance(result, tuple)
    assert len(result) == 2

    # First element should be a 4-tuple
    delta_params, feature2group = result
    assert isinstance(delta_params, tuple)
    assert len(delta_params) == 4

    # Second element should be a dict
    assert isinstance(feature2group, dict)


def test_resolve_delta_timestamps_input_group_calculation(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that input_group delta timestamps are calculated correctly."""
    delta_params, _ = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )

    delta_timestamps, delta_timestamps_std, delta_timestamps_lower, delta_timestamps_upper = delta_params

    # Check input_group exists
    assert "input_group" in delta_timestamps
    assert "input_group" in delta_timestamps_std
    assert "input_group" in delta_timestamps_lower
    assert "input_group" in delta_timestamps_upper

    # Check values are correct (negative because they represent past timestamps)
    # Using default PI0Config values
    expected_mean = [-0.032, -0.16]  # action_decoder_latency_mean, cloud_vlm_latency_mean
    expected_std = [0.010, 0.05]  # action_decoder_latency_std, cloud_vlm_latency_std
    expected_lower = [-0.050, -0.25]  # action_decoder_latency_upper, cloud_vlm_latency_upper (note: -upper)
    expected_upper = [-0.020, -0.10]  # action_decoder_latency_lower, cloud_vlm_latency_lower (note: -lower)

    assert delta_timestamps["input_group"] == expected_mean
    assert delta_timestamps_std["input_group"] == expected_std
    assert delta_timestamps_lower["input_group"] == expected_lower
    assert delta_timestamps_upper["input_group"] == expected_upper


def test_resolve_delta_timestamps_no_reward_action_features(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that reward and action features are not included when not present in dataset."""
    delta_params, feature2group = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )
    delta_timestamps, _, _, _ = delta_params

    # The lerobot_dataset_metadata fixture doesn't have "next.reward" or "action" features
    # So they should not be in delta_timestamps or feature2group
    assert "next.reward" not in delta_timestamps
    assert "action" not in delta_timestamps
    assert "next.reward" not in feature2group
    assert "action" not in feature2group


def test_resolve_delta_timestamps_with_reward_action_features(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that reward and action features are handled correctly when present."""
    # Add reward and action features to the metadata
    lerobot_dataset_metadata.features.update(
        {
            "next.reward": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [7]},
        }
    )

    delta_params, feature2group = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )
    delta_timestamps, _, _, _ = delta_params

    # Check reward feature (PI0Config has reward_delta_indices = None by default)
    assert "next.reward" not in delta_timestamps
    assert "next.reward" not in feature2group

    # Check action feature (PI0Config has action_delta_indices = None by default)
    assert "action" not in delta_timestamps
    assert "action" not in feature2group


@patch("opentau.policies.tau0.configuration_tau0.TAU0Config.action_delta_indices", [0, 1, 2, 3])
@patch("opentau.policies.tau0.configuration_tau0.TAU0Config.reward_delta_indices", [0, 1, 2])
def test_resolve_delta_timestamps_with_delta_indices(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that reward and action features are handled correctly when delta indices are set."""
    # Add reward and action features to the metadata
    lerobot_dataset_metadata.features = {
        "next.reward": {"dtype": "float32", "shape": [1]},
        "actions": {"dtype": "float32", "shape": [7]},
    }

    delta_params, feature2group = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )
    delta_timestamps, _, _, _ = delta_params

    # Check actions feature
    assert "actions" in delta_timestamps
    assert "actions" in feature2group

    # Check values are calculated correctly (indices / action_freq)
    expected_action_timestamps = [0 / 30.0, 1 / 30.0, 2 / 30.0, 3 / 30.0]  # [0, 1, 2, 3] / 30.0
    assert delta_timestamps["actions"] == expected_action_timestamps
    assert feature2group["actions"] == ("actions", None)


@patch("opentau.policies.tau0.configuration_tau0.TAU0Config.action_delta_indices", [0, 1, 2, 3])
@patch("opentau.policies.tau0.configuration_tau0.TAU0Config.reward_delta_indices", [0, 1, 2])
def test_resolve_delta_timestamps_different_action_freq(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that different action frequencies are handled correctly."""
    # Add reward and action features to the metadata
    lerobot_dataset_metadata.features = {
        "next.reward": {"dtype": "float32", "shape": [1]},
        "actions": {"dtype": "float32", "shape": [7]},
    }

    # Change action_freq to 60.0
    train_pipeline_config.dataset_mixture.action_freq = 60.0

    delta_params, _ = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )
    delta_timestamps, _, _, _ = delta_params

    # Check that timestamps are calculated with new action_freq
    expected_action_timestamps = [0 / 60.0, 1 / 60.0, 2 / 60.0, 3 / 60.0]  # [0, 1, 2, 3] / 60.0

    assert delta_timestamps["actions"] == expected_action_timestamps


def test_resolve_delta_timestamps_empty_features(train_pipeline_config, dataset_config):
    """Test behavior with empty features."""
    metadata = MagicMock()
    metadata.features = {}

    delta_params, feature2group = resolve_delta_timestamps(train_pipeline_config, dataset_config, metadata)

    # Should still have input_group in delta_timestamps
    delta_timestamps, _, _, _ = delta_params
    assert "input_group" in delta_timestamps

    # But feature2group should be empty
    assert len(feature2group) == 0


def test_resolve_delta_timestamps_only_image_features(train_pipeline_config, dataset_config):
    """Test behavior with only image features."""
    metadata = MagicMock()
    metadata.features = {
        "camera0": {"dtype": "video", "shape": [3, 224, 224]},
        "camera1": {"dtype": "video", "shape": [3, 224, 224]},
    }

    delta_params, feature2group = resolve_delta_timestamps(train_pipeline_config, dataset_config, metadata)

    # Should have input_group in delta_timestamps
    delta_timestamps, _, _, _ = delta_params
    assert "input_group" in delta_timestamps

    # Should have image features in feature2group
    assert len(feature2group) == 2
    assert "camera0" in feature2group
    assert "camera1" in feature2group
    assert feature2group["camera0"] == ("input_group", [0, 1])
    assert feature2group["camera1"] == ("input_group", [0, 1])


def test_resolve_delta_timestamps_only_state_feature(train_pipeline_config, dataset_config):
    """Test behavior with only state feature."""
    metadata = MagicMock()
    metadata.features = {
        "state": {"dtype": "float32", "shape": [7]},
    }

    delta_params, feature2group = resolve_delta_timestamps(train_pipeline_config, dataset_config, metadata)

    # Should have input_group in delta_timestamps
    delta_timestamps, _, _, _ = delta_params
    assert "input_group" in delta_timestamps

    # Should have state feature in feature2group
    assert len(feature2group) == 1
    assert "state" in feature2group
    assert feature2group["state"] == ("input_group", 0)


def test_resolve_delta_timestamps_zero_latency_values(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test behavior with zero latency values."""
    # Set all latency values to 0
    train_pipeline_config.policy.action_decoder_latency_mean = 0.0
    train_pipeline_config.policy.action_decoder_latency_std = 0.0
    train_pipeline_config.policy.action_decoder_latency_lower = 0.0
    train_pipeline_config.policy.action_decoder_latency_upper = 0.0
    train_pipeline_config.policy.cloud_vlm_latency_mean = 0.0
    train_pipeline_config.policy.cloud_vlm_latency_std = 0.0
    train_pipeline_config.policy.cloud_vlm_latency_lower = 0.0
    train_pipeline_config.policy.cloud_vlm_latency_upper = 0.0

    delta_params, _ = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )
    delta_timestamps, delta_timestamps_std, delta_timestamps_lower, delta_timestamps_upper = delta_params

    # Check that all values are 0
    expected_zero = [0.0, 0.0]
    assert delta_timestamps["input_group"] == expected_zero
    assert delta_timestamps_std["input_group"] == expected_zero
    assert delta_timestamps_lower["input_group"] == expected_zero
    assert delta_timestamps_upper["input_group"] == expected_zero


def test_resolve_delta_timestamps_other_features_ignored(
    train_pipeline_config, dataset_config, lerobot_dataset_metadata
):
    """Test that other features are not included in feature2group."""
    # Add some other features to the metadata
    lerobot_dataset_metadata.features.update(
        {
            "timestamp": {"dtype": "float32", "shape": []},
            "episode_index": {"dtype": "int32", "shape": []},
            "some_other_feature": {"dtype": "float32", "shape": [5]},
        }
    )

    _, feature2group = resolve_delta_timestamps(
        train_pipeline_config, dataset_config, lerobot_dataset_metadata
    )

    # These features should not be in feature2group
    assert "timestamp" not in feature2group
    assert "episode_index" not in feature2group
    assert "some_other_feature" not in feature2group


def test_check_feature_group_mapping_valid():
    """Test that valid feature-group mappings pass validation."""
    # Create a minimal dataset instance with the required attributes
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up valid delta_timestamps_params (4-tuple: mean, std, lower, upper)
    dataset.delta_timestamps_params = (
        {"input_group": [-0.1, 0.0, 0.1]},  # mean
        {"input_group": [0.01, 0.0, 0.01]},  # std
        {"input_group": [-0.2, -0.05, 0.05]},  # lower
        {"input_group": [0.0, 0.05, 0.2]},  # upper
    )

    # Set up valid feature2group mapping
    dataset.feature2group = {
        "observation.state": ("input_group", 0),
        "observation.images.camera0": ("input_group", [0, 1]),
        "action": ("input_group", None),
    }

    # Should not raise any exception
    dataset._check_feature_group_mapping()


def test_check_feature_group_mapping_invalid_group():
    """Test that missing group in delta_timestamps_params raises ValueError."""
    # Create a minimal dataset instance with the required attributes
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up delta_timestamps_params with only one group
    dataset.delta_timestamps_params = (
        {"input_group": [-0.1, 0.0, 0.1]},  # mean
        {"input_group": [0.01, 0.0, 0.01]},  # std
        {"input_group": [-0.2, -0.05, 0.05]},  # lower
        {"input_group": [0.0, 0.05, 0.2]},  # upper
    )

    # Set up feature2group with a group that doesn't exist in delta_timestamps_params
    dataset.feature2group = {
        "observation.images.camera0": ("missing_group", [0, 1]),  # invalid
    }

    # Should raise ValueError
    with pytest.raises(
        ValueError,
        match="Feature 'observation.images.camera0' is mapped to group 'missing_group', which is not present in delta_timestamps_params",
    ):
        dataset._check_feature_group_mapping()


def test_compute_delta_params_basic():
    mean = {"group1": [-0.1, 0.0, 0.1]}
    std = {"group1": [0.01, 0.02, 0.03]}
    lower = {"group1": [-0.2, -0.1, 0.0]}
    upper = {"group1": [0.0, 0.1, 0.2]}
    mean, std, lower, upper = LeRobotDataset.compute_delta_params(mean, std, lower, upper)
    assert isinstance(mean["group1"], np.ndarray)
    assert np.allclose(mean["group1"], [-0.1, 0.0, 0.1])
    assert np.allclose(std["group1"], [0.01, 0.02, 0.03])
    assert np.allclose(lower["group1"], [-0.2, -0.1, 0.0])
    assert np.allclose(upper["group1"], [0.0, 0.1, 0.2])


def test_compute_delta_params_defaults():
    mean = {"group1": [-0.1, 0.0]}
    # std, lower, upper missing
    mean, std, lower, upper = LeRobotDataset.compute_delta_params(mean, None, None, None)
    assert np.allclose(std["group1"], [0.0, 0.0])
    assert np.all(lower["group1"] == -np.inf)
    assert np.all(upper["group1"] == np.inf)


def test_compute_delta_params_partial_defaults():
    mean = {"group1": [-0.1, 0.0]}
    std = {"group1": [0.1, 0.2]}
    # lower missing, upper present
    upper = {"group1": [0.5, 0.6]}
    mean, std, lower, upper = LeRobotDataset.compute_delta_params(mean, std, None, upper)
    assert np.allclose(std["group1"], [0.1, 0.2])
    assert np.all(lower["group1"] == -np.inf)
    assert np.allclose(upper["group1"], [0.5, 0.6])


def test_compute_delta_params_shape_mismatch():
    mean = {"group1": [-0.1, 0.0]}
    std = {"group1": [0.1]}  # wrong shape
    with pytest.raises(ValueError, match="inconsistent shapes"):
        LeRobotDataset.compute_delta_params(mean, std, None, None)


def test_compute_delta_params_empty():
    mean, std, lower, upper = LeRobotDataset.compute_delta_params({}, {}, {}, {})
    assert mean == {}
    assert std == {}
    assert lower == {}
    assert upper == {}


# Tests for get_delta_indices_soft method
def test_get_delta_indices_soft_basic():
    """Test basic functionality of get_delta_indices_soft."""
    from opentau.datasets.utils import get_delta_indices_soft

    # Set up delta timestamps info (mean, std, lower, upper)
    delta_timestamps_info = (
        {"group1": np.array([-0.1, 0.0, 0.1])},  # mean
        {"group1": np.array([0.01, 0.02, 0.03])},  # std
        {"group1": np.array([-0.2, -0.1, 0.0])},  # lower
        {"group1": np.array([0.0, 0.1, 0.2])},  # upper
    )

    fps = 30

    # Set random seed for reproducible test
    np.random.seed(42)
    result = get_delta_indices_soft(delta_timestamps_info, fps)

    # Check that result has the expected structure
    assert "group1" in result
    assert isinstance(result["group1"], np.ndarray)
    assert result["group1"].shape == (3,)

    # Check that values are within expected bounds (converted to frame indices)
    expected_lower = np.array([-0.2, -0.1, 0.0]) * fps
    expected_upper = np.array([0.0, 0.1, 0.2]) * fps
    assert np.all(result["group1"] >= expected_lower)
    assert np.all(result["group1"] <= expected_upper)


def test_get_delta_indices_soft_multiple_groups():
    """Test get_delta_indices_soft with multiple groups."""
    from opentau.datasets.utils import get_delta_indices_soft

    # Set up delta timestamps info with multiple groups
    delta_timestamps_info = (
        {
            "group1": np.array([-0.1, 0.0]),
            "group2": np.array([0.0, 0.1]),
        },  # mean
        {
            "group1": np.array([0.01, 0.02]),
            "group2": np.array([0.02, 0.03]),
        },  # std
        {
            "group1": np.array([-0.2, -0.1]),
            "group2": np.array([-0.1, 0.0]),
        },  # lower
        {
            "group1": np.array([0.0, 0.1]),
            "group2": np.array([0.1, 0.2]),
        },  # upper
    )

    fps = 30

    # Set random seed for reproducible test
    np.random.seed(42)
    result = get_delta_indices_soft(delta_timestamps_info, fps)

    # Check that both groups are present
    assert "group1" in result
    assert "group2" in result

    # Check shapes
    assert result["group1"].shape == (2,)
    assert result["group2"].shape == (2,)

    # Check bounds for group1
    expected_lower_1 = np.array([-0.2, -0.1]) * fps
    expected_upper_1 = np.array([0.0, 0.1]) * fps
    assert np.all(result["group1"] >= expected_lower_1)
    assert np.all(result["group1"] <= expected_upper_1)

    # Check bounds for group2
    expected_lower_2 = np.array([-0.1, 0.0]) * fps
    expected_upper_2 = np.array([0.1, 0.2]) * fps
    assert np.all(result["group2"] >= expected_lower_2)
    assert np.all(result["group2"] <= expected_upper_2)


def test_get_delta_indices_soft_zero_std():
    """Test get_delta_indices_soft with zero standard deviation (deterministic)."""
    from opentau.datasets.utils import get_delta_indices_soft

    # Set up delta timestamps info with zero std
    delta_timestamps_info = (
        {"group1": np.array([-0.1, 0.0, 0.1])},  # mean
        {"group1": np.array([0.0, 0.0, 0.0])},  # std (zero)
        {"group1": np.array([-0.2, -0.1, 0.0])},  # lower
        {"group1": np.array([0.0, 0.1, 0.2])},  # upper
    )

    fps = 30

    # Set random seed for reproducible test
    np.random.seed(42)
    result1 = get_delta_indices_soft(delta_timestamps_info, fps)
    result2 = get_delta_indices_soft(delta_timestamps_info, fps)

    # With zero std, results should be deterministic
    assert np.allclose(result1["group1"], result2["group1"])

    # Values should be exactly the mean * fps
    expected = np.array([-0.1, 0.0, 0.1]) * fps
    assert np.allclose(result1["group1"], expected)


def test_get_delta_indices_soft_clipping():
    """Test that get_delta_indices_soft properly clips values to bounds."""
    from opentau.datasets.utils import get_delta_indices_soft

    # Set up delta timestamps info with tight bounds
    delta_timestamps_info = (
        {"group1": np.array([0.0])},  # mean
        {"group1": np.array([1.0])},  # large std
        {"group1": np.array([-0.1])},  # lower bound
        {"group1": np.array([0.1])},  # upper bound
    )

    fps = 30

    # Set random seed for reproducible test
    np.random.seed(42)
    result = get_delta_indices_soft(delta_timestamps_info, fps)

    # Check that values are clipped to bounds
    expected_lower = -0.1 * fps
    expected_upper = 0.1 * fps
    assert np.all(result["group1"] >= expected_lower)
    assert np.all(result["group1"] <= expected_upper)


def test_get_delta_indices_soft_different_fps():
    """Test get_delta_indices_soft with different FPS values."""
    from opentau.datasets.utils import get_delta_indices_soft

    # Set up delta timestamps info
    delta_timestamps_info = (
        {"group1": np.array([-0.1, 0.0, 0.1])},  # mean
        {"group1": np.array([0.01, 0.02, 0.03])},  # std
        {"group1": np.array([-0.2, -0.1, 0.0])},  # lower
        {"group1": np.array([0.0, 0.1, 0.2])},  # upper
    )

    fps1 = 30
    fps2 = 60

    # Set random seed for reproducible test
    np.random.seed(42)
    result1 = get_delta_indices_soft(delta_timestamps_info, fps1)
    np.random.seed(42)
    result2 = get_delta_indices_soft(delta_timestamps_info, fps2)

    # Results should be scaled by the FPS ratio
    assert np.allclose(result2["group1"], result1["group1"] * (fps2 / fps1))


# Tests for _get_query_indices_soft method
def test_get_query_indices_soft_basic():
    """Test basic functionality of _get_query_indices_soft."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up episode_data_index
    dataset.episode_data_index = {
        "from": np.array([0, 100, 200], dtype=np.int64),
        "to": np.array([100, 200, 300], dtype=np.int64),
    }

    # Set up delta_timestamps_params
    dataset.delta_timestamps_params = (
        {"input_group": np.array([-0.1, 0.0, 0.1])},  # mean
        {"input_group": np.array([0.01, 0.02, 0.03])},  # std
        {"input_group": np.array([-0.2, -0.1, 0.0])},  # lower
        {"input_group": np.array([0.0, 0.1, 0.2])},  # upper
    )

    # Set up feature2group mapping
    dataset.feature2group = {
        "observation.state": ("input_group", 0),
        "observation.images.camera0": ("input_group", [0, 1]),
        "action": ("input_group", None),
    }

    # Mock the fps property to return 30
    type(dataset).fps = property(lambda self: 30)

    # Mock get_delta_indices_soft to return predictable values
    with patch("opentau.datasets.lerobot_dataset.get_delta_indices_soft") as mock_get_delta:
        mock_get_delta.return_value = {
            "input_group": np.array([-3.0, 0.0, 3.0])  # -0.1s, 0s, 0.1s * 30fps
        }

        idx = 50  # Current index
        ep_idx = 1  # Episode 1 (starts at 100, ends at 200)
        dataset.epi2idx = {1: 1}

        query_indices, padding = dataset._get_query_indices_soft(idx, ep_idx)

        # Check query_indices
        assert "observation.state" in query_indices
        assert "observation.images.camera0" in query_indices
        assert "action" in query_indices

        # observation.state should use index 0 from input_group
        expected_state_idx = np.clip(50 + (-3.0), 100, 199)  # 47 clipped to 100
        assert query_indices["observation.state"] == expected_state_idx

        # observation.images.camera0 should use indices [0, 1] from input_group
        expected_camera_indices = np.clip(50 + np.array([-3.0, 0.0]), 100, 199)
        assert np.allclose(query_indices["observation.images.camera0"], expected_camera_indices)

        # action should use all indices from input_group
        expected_action_indices = np.clip(50 + np.array([-3.0, 0.0, 3.0]), 100, 199)
        assert np.allclose(query_indices["action"], expected_action_indices)

        # Check padding masks
        assert "observation.state_is_pad" in padding
        assert "observation.images.camera0_is_pad" in padding
        assert "action_is_pad" in padding

        # observation.state should be padded (47 < 100)
        assert padding["observation.state_is_pad"].item() is True

        # observation.images.camera0 should have mixed padding
        expected_camera_padding = torch.BoolTensor([True, True])  # 47 < 100
        assert torch.all(padding["observation.images.camera0_is_pad"] == expected_camera_padding)

        # action should have mixed padding
        expected_action_padding = torch.BoolTensor([True, True, True])  # 47 < 100
        assert torch.all(padding["action_is_pad"] == expected_action_padding)


def test_get_query_indices_soft_no_padding():
    """Test _get_query_indices_soft when no indices are outside episode bounds."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up episode_data_index
    dataset.episode_data_index = {
        "from": np.array([0, 100, 200], dtype=np.int64),
        "to": np.array([100, 200, 300], dtype=np.int64),
    }

    # Set up delta_timestamps_params
    dataset.delta_timestamps_params = (
        {"input_group": np.array([-0.1, 0.0, 0.1])},  # mean
        {"input_group": np.array([0.01, 0.02, 0.03])},  # std
        {"input_group": np.array([-0.2, -0.1, 0.0])},  # lower
        {"input_group": np.array([0.0, 0.1, 0.2])},  # upper
    )

    # Set up feature2group mapping
    dataset.feature2group = {
        "observation.state": ("input_group", 0),
    }

    # Mock the fps property to return 30
    type(dataset).fps = property(lambda self: 30)

    # Mock get_delta_indices_soft to return small values
    with patch("opentau.datasets.lerobot_dataset.get_delta_indices_soft") as mock_get_delta:
        mock_get_delta.return_value = {
            "input_group": np.array([-1.0])  # Small offset
        }

        idx = 150  # Current index (well within episode 1 bounds)
        ep_idx = 1  # Episode 1 (starts at 100, ends at 200)
        dataset.epi2idx = {1: 1}

        query_indices, padding = dataset._get_query_indices_soft(idx, ep_idx)

        # Check query_indices
        expected_state_idx = np.clip(150 + (-1.0), 100, 199)  # 149
        assert query_indices["observation.state"] == expected_state_idx

        # Check padding mask (should be False since 149 >= 100)
        assert padding["observation.state_is_pad"].item() is False


def test_get_query_indices_soft_empty_feature2group():
    """Test _get_query_indices_soft with empty feature2group."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up episode_data_index
    dataset.episode_data_index = {
        "from": np.array([0, 100, 200], dtype=np.int64),
        "to": np.array([100, 200, 300], dtype=np.int64),
    }

    # Set up delta_timestamps_params
    dataset.delta_timestamps_params = (
        {"input_group": np.array([-0.1, 0.0, 0.1])},  # mean
        {"input_group": np.array([0.01, 0.02, 0.03])},  # std
        {"input_group": np.array([-0.2, -0.1, 0.0])},  # lower
        {"input_group": np.array([0.0, 0.1, 0.2])},  # upper
    )

    # Set up empty feature2group
    dataset.feature2group = {}

    # Mock the fps property to return 30
    type(dataset).fps = property(lambda self: 30)

    # Mock get_delta_indices_soft
    with patch("opentau.datasets.lerobot_dataset.get_delta_indices_soft") as mock_get_delta:
        mock_get_delta.return_value = {"input_group": np.array([-3.0, 0.0, 3.0])}

        idx = 50
        ep_idx = 1
        dataset.epi2idx = {1: 1}

        query_indices, padding = dataset._get_query_indices_soft(idx, ep_idx)

        # Should return empty dictionaries
        assert query_indices == {}
        assert padding == {}


def test_get_query_indices_soft_edge_cases():
    """Test _get_query_indices_soft with edge cases."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)

    # Set up episode_data_index
    dataset.episode_data_index = {
        "from": np.array([0, 100, 200], dtype=np.int64),
        "to": np.array([100, 200, 300], dtype=np.int64),
    }

    # Set up delta_timestamps_params
    dataset.delta_timestamps_params = (
        {"input_group": np.array([-0.1, 0.0, 0.1])},  # mean
        {"input_group": np.array([0.01, 0.02, 0.03])},  # std
        {"input_group": np.array([-0.2, -0.1, 0.0])},  # lower
        {"input_group": np.array([0.0, 0.1, 0.2])},  # upper
    )

    # Set up feature2group mapping
    dataset.feature2group = {
        "feature1": ("input_group", 0),  # single index
        "feature2": ("input_group", [1, 2]),  # list of indices
        "feature3": ("input_group", None),  # all indices
    }

    # Mock the fps property to return 30
    type(dataset).fps = property(lambda self: 30)

    # Mock get_delta_indices_soft
    with patch("opentau.datasets.lerobot_dataset.get_delta_indices_soft") as mock_get_delta:
        mock_get_delta.return_value = {"input_group": np.array([-5.0, 0.0, 5.0])}

        # Test at episode start
        idx = 100  # Episode 1 start
        ep_idx = 1
        dataset.epi2idx = {1: 1}

        query_indices, padding = dataset._get_query_indices_soft(idx, ep_idx)

        # feature1 should be clipped to episode start
        assert query_indices["feature1"] == 100  # 100 + (-5) clipped to 100

        # feature2 should use indices [1, 2]
        expected_feature2 = np.clip(100 + np.array([0.0, 5.0]), 100, 199)
        assert np.allclose(query_indices["feature2"], expected_feature2)

        # feature3 should use all indices
        expected_feature3 = np.clip(100 + np.array([-5.0, 0.0, 5.0]), 100, 199)
        assert np.allclose(query_indices["feature3"], expected_feature3)

        # Check padding masks
        assert padding["feature1_is_pad"].item() is True
        assert torch.all(padding["feature2_is_pad"] == torch.BoolTensor([False, False]))
        assert torch.all(padding["feature3_is_pad"] == torch.BoolTensor([True, False, False]))


# Tests for _query_hf_dataset_soft method
def test_query_hf_dataset_soft_linear_interpolation():
    """Test _query_hf_dataset_soft with linear interpolation strategy."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.vector_resample_strategy = "linear"

    # Mock the _query_hf_dataset method
    with patch.object(dataset, "_query_hf_dataset") as mock_query:
        # Set up mock return values for floor and ceil queries
        mock_query.side_effect = [
            {
                "action": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # floor values
                "state": torch.tensor([10.0, 20.0]),  # floor values
            },
            {
                "action": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),  # ceil values
                "state": torch.tensor([30.0, 40.0]),  # ceil values
            },
        ]

        # Soft indices with fractional parts
        soft_indices = {
            "action": np.array([10.3, 11.7]),  # 0.3 and 0.7 distance to floor
            "state": np.array([15.2]),  # 0.2 distance to floor
        }

        result = dataset._query_hf_dataset_soft(soft_indices)

        # Check that _query_hf_dataset was called with correct floor and ceil indices
        assert mock_query.call_count == 2

        # First call should be with floor indices
        floor_call_args = mock_query.call_args_list[0][0][0]
        assert np.allclose(floor_call_args["action"], [10, 11])
        assert np.allclose(floor_call_args["state"], [15])

        # Second call should be with ceil indices
        ceil_call_args = mock_query.call_args_list[1][0][0]
        assert np.allclose(ceil_call_args["action"], [11, 12])
        assert np.allclose(ceil_call_args["state"], [16])

        # Check interpolation results
        # action[0]: 0.7 * [1,2] + 0.3 * [5,6] = [0.7, 1.4] + [1.5, 1.8] = [2.2, 3.2]
        # action[1]: 0.3 * [3,4] + 0.7 * [7,8] = [0.9, 1.2] + [4.9, 5.6] = [5.8, 6.8]
        expected_action = torch.tensor([[2.2, 3.2], [5.8, 6.8]], dtype=torch.float64)
        assert torch.allclose(result["action"], expected_action, atol=1e-6)

        # state: 0.8 * [10,20] + 0.2 * [30,40] = [8,16] + [6,8] = [14,24]
        expected_state = torch.tensor([14.0, 24.0], dtype=torch.float64)
        assert torch.allclose(result["state"], expected_state, atol=1e-6)


def test_query_hf_dataset_soft_linear_exact_integers():
    """Test _query_hf_dataset_soft with linear interpolation when indices are exact integers."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.vector_resample_strategy = "linear"

    # Mock the _query_hf_dataset method
    with patch.object(dataset, "_query_hf_dataset") as mock_query:
        # Set up mock return values (floor and ceil should be the same for exact integers)
        mock_query.side_effect = [
            {
                "action": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            },
            {
                "action": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # same as floor
            },
        ]

        # Soft indices that are exact integers
        soft_indices = {
            "action": np.array([10.0, 11.0]),  # exact integers
        }

        result = dataset._query_hf_dataset_soft(soft_indices)

        # Check that _query_hf_dataset was called with correct indices
        assert mock_query.call_count == 2

        # Both calls should be with the same integer indices
        floor_call_args = mock_query.call_args_list[0][0][0]
        ceil_call_args = mock_query.call_args_list[1][0][0]
        assert np.allclose(floor_call_args["action"], [10, 11])
        assert np.allclose(ceil_call_args["action"], [10, 11])

        # Result should be the same as input (no interpolation needed)
        expected_action = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        assert torch.allclose(result["action"], expected_action)


def test_query_hf_dataset_soft_nearest_interpolation():
    """Test _query_hf_dataset_soft with nearest neighbor interpolation strategy."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.vector_resample_strategy = "nearest"

    # Mock the _query_hf_dataset method
    with patch.object(dataset, "_query_hf_dataset") as mock_query:
        mock_query.return_value = {
            "action": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "state": torch.tensor([10.0, 20.0]),
        }

        # Soft indices with fractional parts
        soft_indices = {
            "action": np.array([10.3, 11.7]),  # should round to [10, 12]
            "state": np.array([15.2]),  # should round to [15]
        }

        result = dataset._query_hf_dataset_soft(soft_indices)

        # Check that _query_hf_dataset was called with rounded indices
        mock_query.assert_called_once()
        call_args = mock_query.call_args[0][0]
        assert np.allclose(call_args["action"], [10, 12])
        assert np.allclose(call_args["state"], [15])

        # Result should be the same as the mock return value
        expected_action = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_state = torch.tensor([10.0, 20.0])
        assert torch.allclose(result["action"], expected_action)
        assert torch.allclose(result["state"], expected_state)


def test_query_hf_dataset_soft_nearest_rounding():
    """Test _query_hf_dataset_soft with nearest neighbor rounding behavior."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.vector_resample_strategy = "nearest"

    # Mock the _query_hf_dataset method
    with patch.object(dataset, "_query_hf_dataset") as mock_query:
        mock_query.return_value = {
            "action": torch.tensor([[1.0, 2.0]]),
        }

        # Test various rounding scenarios
        # NOTE: numpy round, rounds to nearest even integer
        soft_indices = {
            "action": np.array([10.4, 10.5, 10.6]),  # should round to [10, 10, 11]
        }

        dataset._query_hf_dataset_soft(soft_indices)

        # Check that _query_hf_dataset was called with correctly rounded indices
        call_args = mock_query.call_args[0][0]
        assert np.allclose(call_args["action"], np.array([10, 10, 11]))


def test_query_hf_dataset_soft_invalid_strategy():
    """Test _query_hf_dataset_soft raises error for invalid strategy."""
    # Create a minimal dataset instance
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.vector_resample_strategy = "invalid_strategy"

    soft_indices = {"action": np.array([10.5])}

    with pytest.raises(ValueError, match="Unsupported vector_resample_strategy"):
        dataset._query_hf_dataset_soft(soft_indices)


@pytest.mark.parametrize(
    ("feature2group", "delta_indices"),
    [
        (
            {
                "observation.images.exterior_image_1_left": ("input_group", [0, 1]),
                "observation.images.exterior_image_2_left": ("input_group", [0, 1]),
                "observation.images.wrist_image_left": ("input_group", [0, 1]),
                "observation.state": ("input_group", 0),
                "action": ("action", None),
            },
            {
                "input_group": np.array([-0.5902292, -3.7006334]),
                "action": np.array([0.5 * i for i in range(50)]),
            },
        ),
        (
            {
                "observation.images.exterior_image_1_left": ("input_group", [0, 1]),
                "observation.images.exterior_image_2_left": ("input_group", [0, 1]),
                "observation.images.wrist_image_left": ("input_group", [1, 1]),
                "observation.state": ("input_group", 1),
                "action": ("action", None),
            },
            {
                "input_group": np.array([-0.5902292, -3.7006334]),
                "action": np.array([0.5 * i for i in range(50)]),
            },
        ),
        (
            {
                "observation.images.exterior_image_1_left": ("input_group", [1, 0]),
                "observation.images.exterior_image_2_left": ("input_group", [1, 0]),
                "observation.images.wrist_image_left": ("input_group", [0, 0]),
                "observation.state": ("input_group", 0),
                "action": ("action", None),
            },
            {
                "input_group": np.array([-0.5902292, -3.7006334]),
                "action": np.array([0.5 * i for i in range(50)]),
            },
        ),
        (
            {
                "observation.images.exterior_image_1_left": ("input_group", [1, 0]),
                "observation.images.exterior_image_2_left": ("input_group", [1, 0]),
                "observation.images.wrist_image_left": ("control_group", [0, 0]),
                "observation.state": ("input_group", 0),
                "action": ("action", None),
            },
            {
                "input_group": np.array([-0.5902292, -3.7006334]),
                "control_group": np.array([-0.7802292, -2.7465334]),
                "action": np.array([0.5 * i for i in range(50)]),
            },
        ),
    ],
)
def test_feature2group(lerobot_dataset, feature2group, delta_indices):
    """
    Test if the function get_query_indices_soft returns same timesteps for same group features. The group inside features group are changed
    """

    # setting the feature2group attribute in lerobot dataset
    lerobot_dataset.feature2group = feature2group

    # mocking start id  and end id of episode and setting delta_indices to predefined value
    with (
        patch.object(
            lerobot_dataset, "episode_data_index", new_callable=MagicMock
        ) as mock_episode_data_index,
        patch("opentau.datasets.lerobot_dataset.get_delta_indices_soft", return_value=delta_indices),
    ):
        # Configure the mocks just like before
        mock_from_item = MagicMock()
        mock_from_item.item.return_value = 700

        mock_to_item = MagicMock()
        mock_to_item.item.return_value = 799

        mock_episode_data_index.__getitem__.side_effect = lambda key: {
            "from": [mock_from_item],
            "to": [mock_to_item],
        }[key]

        query_indices, _ = lerobot_dataset._get_query_indices_soft(idx=710, ep_idx=0)

        # grouping all the query indices with same group into dictionary
        dict1 = defaultdict(list)
        for key, (group, indices) in feature2group.items():
            if type(indices) is int:
                dict1[(str(indices) + "_" + str(group))].append(float(query_indices[key]))
            elif indices:
                for i, idx in enumerate(indices):
                    dict1[(str(idx) + "_" + str(group))].append(float(query_indices[key][i]))

        # checking if all the values are same in the list of same group

        for _, values in dict1.items():
            assert len(list(set(values))) == 1
