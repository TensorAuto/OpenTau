#!/usr/bin/env python

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

import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.opentau.datasets.dataset_mixture import (
    DatasetMixtureMetadata,
    WeightedDatasetMixture,
    pad_vector,
)
from src.opentau.datasets.factory import make_dataset
from src.opentau.configs.default import DatasetConfig


class TestPadVector:
    """Test the pad_vector utility function."""

    def test_pad_vector_no_padding_needed(self):
        """Test pad_vector when no padding is needed."""
        vector = np.array([1, 2, 3], dtype=np.float32)
        result = pad_vector(vector, 3)
        np.testing.assert_array_equal(result, vector)
        assert result.dtype == vector.dtype

    def test_pad_vector_with_padding(self):
        """Test pad_vector when padding is needed."""
        vector = np.array([1, 2], dtype=np.float32)
        result = pad_vector(vector, 4)
        expected = np.array([1, 2, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == vector.dtype

    def test_pad_vector_multidimensional(self):
        """Test pad_vector with multidimensional arrays."""
        vector = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = pad_vector(vector, 3)
        expected = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_different_dtypes(self):
        """Test pad_vector with different data types."""
        vector = np.array([1, 2], dtype=np.int32)
        result = pad_vector(vector, 4)
        expected = np.array([1, 2, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == vector.dtype


class TestDatasetMixtureMetadata:
    """Test the DatasetMixtureMetadata class."""

    @patch("src.opentau.datasets.dataset_mixture.DATA_FEATURES_NAME_MAPPING")
    def test_init_success(self, mock_name_mapping, train_pipeline_config, lerobot_dataset_metadata):
        """Test successful initialization of DatasetMixtureMetadata."""
        # Mock the name mapping
        mock_name_mapping.__getitem__.return_value = {
            "state": "state",
            "actions": "actions",
            "camera0": "camera0",
        }

        metadatas = [lerobot_dataset_metadata]
        dataset_weights = [1.0]

        metadata_mixture = DatasetMixtureMetadata(train_pipeline_config, metadatas, dataset_weights)

        assert metadata_mixture.cfg == train_pipeline_config
        assert "state" in metadata_mixture.stats
        assert "actions" in metadata_mixture.stats
        assert "camera0" in metadata_mixture.stats

    @patch("src.opentau.datasets.dataset_mixture.DATA_FEATURES_NAME_MAPPING")
    def test_to_standard_data_format(self, mock_name_mapping, train_pipeline_config):
        """Test _to_standard_data_format method."""
        # Mock the name mapping
        mock_name_mapping.__getitem__.return_value = {
            "state": "state",
            "actions": "actions",
        }

        train_pipeline_config.num_cams = 2
        train_pipeline_config.action_expert_num_cams = 2

        metadata_mixture = DatasetMixtureMetadata.__new__(DatasetMixtureMetadata)
        metadata_mixture.cfg = train_pipeline_config

        stats = {
            "state": {
                "mean": np.array([0.5, 0.3], dtype=np.float32),
                "std": np.array([0.1, 0.05], dtype=np.float32),
                "min": np.array([0.0, 0.0], dtype=np.float32),
                "max": np.array([1.0, 1.0], dtype=np.float32),
                "count": np.array([100], dtype=np.float32),
            },
            "actions": {
                "mean": np.array([0.2], dtype=np.float32),
                "std": np.array([0.05], dtype=np.float32),
                "min": np.array([0.0], dtype=np.float32),
                "max": np.array([1.0], dtype=np.float32),
                "count": np.array([100], dtype=np.float32),
            },
        }

        result = metadata_mixture._to_standard_data_format("test_dataset", stats)

        # Check that vectors were padded
        assert result["state"]["mean"].shape[-1] == train_pipeline_config.max_state_dim
        assert result["actions"]["mean"].shape[-1] == train_pipeline_config.max_action_dim

        # Check that missing cameras were added
        # there should be 2 cameras and 2 local cameras
        assert "camera0" in result
        assert "camera1" in result
        assert "camera2" not in result
        assert "local_camera0" in result
        assert "local_camera1" in result
        assert "local_camera2" not in result

    @patch("src.opentau.datasets.dataset_mixture.DATA_FEATURES_NAME_MAPPING")
    def test_to_standard_data_format_missing_key(self, mock_name_mapping, train_pipeline_config):
        """Test _to_standard_data_format with missing key."""
        # Mock the name mapping to include a key not in stats
        mock_name_mapping.__getitem__.return_value = {
            "state": "missing_key",
        }

        metadata_mixture = DatasetMixtureMetadata.__new__(DatasetMixtureMetadata)
        metadata_mixture.cfg = train_pipeline_config

        stats = {"existing_key": {"mean": np.array([0.5])}}

        with pytest.raises(KeyError, match="Key 'missing_key' not found in stats"):
            metadata_mixture._to_standard_data_format("test_dataset", stats)

    @patch("src.opentau.datasets.dataset_mixture.DATA_FEATURES_NAME_MAPPING")
    def test_to_standard_data_format_missing_stats(self, mock_name_mapping, train_pipeline_config):
        """Test _to_standard_data_format with missing required statistics."""
        # Mock the name mapping
        mock_name_mapping.__getitem__.return_value = {
            "state": "state",
            "actions": "actions",
        }

        metadata_mixture = DatasetMixtureMetadata.__new__(DatasetMixtureMetadata)
        metadata_mixture.cfg = train_pipeline_config

        # Missing required stats
        stats = {
            "state": {
                "mean": np.array([0.5]),
                "count": np.array([100]),
                # Missing std, min, max
            },
            "actions": {
                "mean": np.array([0.2]),
                "count": np.array([100]),
                # Missing std, min, max
            },
        }

        with pytest.raises(KeyError, match="missing required statistics"):
            metadata_mixture._to_standard_data_format("test_dataset", stats)

    def test_features_property(self, train_pipeline_config):
        """Test the features property."""
        metadata_mixture = DatasetMixtureMetadata.__new__(DatasetMixtureMetadata)
        train_pipeline_config.num_cams = 2
        train_pipeline_config.action_expert_num_cams = 2
        metadata_mixture.cfg = train_pipeline_config

        features = metadata_mixture.features

        assert "state" in features
        assert "actions" in features
        assert "camera0" in features
        assert "camera1" in features
        assert "camera2" not in features
        assert "local_camera0" in features
        assert "local_camera1" in features
        assert "local_camera2" not in features

        assert features["state"]["shape"] == (train_pipeline_config.max_state_dim,)
        assert features["actions"]["shape"] == (train_pipeline_config.max_action_dim,)
        assert features["camera0"]["shape"] == (
            3,
            train_pipeline_config.resolution[0],
            train_pipeline_config.resolution[1],
        )
        assert features["camera1"]["shape"] == (
            3,
            train_pipeline_config.resolution[0],
            train_pipeline_config.resolution[1],
        )
        assert features["local_camera0"]["shape"] == (
            3,
            train_pipeline_config.resolution[0],
            train_pipeline_config.resolution[1],
        )
        assert features["local_camera1"]["shape"] == (
            3,
            train_pipeline_config.resolution[0],
            train_pipeline_config.resolution[1],
        )


class TestWeightedDatasetMixture:
    """Test the WeightedDatasetMixture class."""

    @pytest.mark.slow  # 3 sec
    def test_init_success(self, train_pipeline_config, datasets_factory):
        """Test successful initialization of WeightedDatasetMixture."""
        action_freq = 30.0
        datasets = datasets_factory(5)
        dataset_weights = [1.0 / len(datasets)] * len(datasets)
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, action_freq)

        assert mixture.cfg == train_pipeline_config
        assert mixture.datasets == datasets
        assert mixture.dataset_weights == dataset_weights
        assert mixture.action_freq == action_freq
        assert len(mixture.dataset_names) == 5
        assert mixture.concatenated_dataset is not None
        assert mixture.sample_weights is not None
        assert mixture.meta is not None

    def test_init_empty_datasets(self, train_pipeline_config):
        """Test initialization with empty dataset list."""
        with pytest.raises(ValueError, match="The list of datasets cannot be empty"):
            WeightedDatasetMixture(train_pipeline_config, [], [1.0], 30.0)

    @pytest.mark.slow  # 2 sec
    def test_init_mismatched_lengths(self, train_pipeline_config, datasets_factory):
        """Test initialization with mismatched dataset and weight lengths."""
        with pytest.raises(
            ValueError, match="The number of datasets must match the number of dataset_weights"
        ):
            datasets = datasets_factory(5)
            dataset_weights = [1.0 / len(datasets)] * (len(datasets) - 1)
            WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

    def test_init_negative_weights(self, train_pipeline_config, datasets_factory):
        """Test initialization with negative weights."""
        with pytest.raises(ValueError, match="Dataset weights must be non-negative"):
            WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [1.0, -0.5], 30.0)

    @pytest.mark.slow  # 2 sec
    def test_init_zero_weights_warning(self, train_pipeline_config, datasets_factory, caplog):
        """Test initialization with all zero weights triggers warning."""
        with caplog.at_level(logging.WARNING):
            WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [0.0, 0.0], 30.0)

        assert "All dataset weights are zero" in caplog.text

    def test_init_dataset_without_meta(self, train_pipeline_config, datasets_factory):
        """Test initialization with dataset that has no meta attribute."""
        dataset_without_meta = datasets_factory(1)[0]
        dataset_without_meta.meta = None

        with pytest.raises(ValueError, match="All datasets must have a 'meta' attribute"):
            WeightedDatasetMixture(train_pipeline_config, [dataset_without_meta], [1.0], 30.0)

    @pytest.mark.slow  # 1 sec
    def test_calculate_sample_weights_normal(self, train_pipeline_config, datasets_factory):
        """Test _calculate_sample_weights with normal datasets."""
        datasets = datasets_factory(3)
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, [0.7, 0.3, 0.2], 30.0)

        weights = mixture._calculate_sample_weights()

        assert weights is not None
        assert len(weights) == 450
        assert torch.sum(weights) > 0

    def test_calculate_sample_weights_zero_weights(self, train_pipeline_config, datasets_factory):
        """Test _calculate_sample_weights with zero weights."""
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [0.0, 0.0], 30.0)

        weights = mixture._calculate_sample_weights()

        assert weights is not None
        assert torch.sum(weights) == 0

    def test_get_dataloader_success(self, train_pipeline_config, datasets_factory):
        """Test successful dataloader creation."""
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [0.7, 0.3], 30.0)

        dataloader = mixture.get_dataloader()

        assert dataloader is not None
        assert dataloader.batch_size == train_pipeline_config.batch_size
        assert dataloader.num_workers == train_pipeline_config.num_workers

    @pytest.mark.slow  # 1 sec
    def test_get_dataloader_zero_weights_error(self, train_pipeline_config, datasets_factory):
        """Test dataloader creation with zero weights raises error."""
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [0.0, 0.0], 30.0)

        with pytest.raises(ValueError, match="No non-empty dataset has a positive sampling weight."):
            mixture.get_dataloader()

    def test_log_dataset_info(self, train_pipeline_config, datasets_factory, caplog):
        """Test _log_dataset_info method."""
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets_factory(2), [0.7, 0.3], 30.0)

        with caplog.at_level(logging.INFO):
            mixture._log_dataset_info()

        assert "Dataset information:" in caplog.text
        assert "Length=150" in caplog.text
        assert "Weight=0.7" in caplog.text
        assert "Weight=0.3" in caplog.text


class TestWeightedDatasetMixtureIntegration:
    """Integration tests for WeightedDatasetMixture class."""

    def test_integration_basic_functionality(self, train_pipeline_config, datasets_factory):
        """Test basic integration functionality with multiple datasets."""
        # Create datasets with different sizes
        datasets = datasets_factory(3)
        dataset_weights = [0.5, 0.3, 0.2]
        action_freq = 30.0

        # Create mixture
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, action_freq)

        # Verify basic properties
        assert len(mixture.datasets) == 3
        assert len(mixture.dataset_weights) == 3
        assert mixture.action_freq == action_freq
        assert len(mixture.dataset_names) == 3
        assert mixture.concatenated_dataset is not None
        assert mixture.sample_weights is not None
        assert mixture.meta is not None

        # Verify concatenated dataset length
        expected_total_length = sum(len(ds) for ds in datasets)
        assert len(mixture.concatenated_dataset) == expected_total_length

        # Verify sample weights length
        assert len(mixture.sample_weights) == expected_total_length

        # Create dataloader and verify it works
        dataloader = mixture.get_dataloader()
        assert dataloader is not None
        assert dataloader.batch_size == train_pipeline_config.batch_size
        assert dataloader.num_workers == train_pipeline_config.num_workers

    @pytest.mark.slow  # ~3.5 min
    def test_integration_basic_functionality_with_no_latency_and_same_fps_as_dataset(
        self, train_pipeline_config
    ):
        """Test that the mixture with no latency and same fps as datasets actually uses the samples from the datasets."""
        # Create dataset
        dataset_config = DatasetConfig(
            repo_id="lerobot/droid_100",
            episodes=[0, 1],
        )

        # set fps to same as droid_100 dataset and resample strategy to nearest
        train_pipeline_config.dataset_mixture.action_freq = 15.0  # same fps as droid_100 dataset
        train_pipeline_config.dataset_mixture.image_resample_strategy = "nearest"
        train_pipeline_config.dataset_mixture.vector_resample_strategy = "nearest"

        # set latency to 0
        train_pipeline_config.policy.cloud_vlm_latency_mean = 0.0
        train_pipeline_config.policy.cloud_vlm_latency_std = 0.0
        train_pipeline_config.policy.cloud_vlm_latency_lower = 0.0
        train_pipeline_config.policy.cloud_vlm_latency_upper = 0.0
        train_pipeline_config.policy.action_decoder_latency_mean = 0.0
        train_pipeline_config.policy.action_decoder_latency_std = 0.0
        train_pipeline_config.policy.action_decoder_latency_lower = 0.0
        train_pipeline_config.policy.action_decoder_latency_upper = 0.0

        train_pipeline_config.batch_size = 1
        train_pipeline_config.dataloader_batch_size = 1
        train_pipeline_config.num_workers = 1

        datasets = [make_dataset(dataset_config, train_pipeline_config)]
        dataset_weights = [1]
        action_freq = 30.0

        # Create mixture
        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, action_freq)

        # Create dataloader and verify it works
        dataloader = mixture.get_dataloader()

        def are_tensor_dicts_equal(d1: dict, d2: dict) -> bool:
            """
            Compares two dictionaries with PyTorch tensors as values.
            """
            if d1.keys() != d2.keys():
                return False
            return all(
                torch.equal(d1[key], d2[key])
                if isinstance(d1[key], torch.Tensor) and isinstance(d2[key], torch.Tensor)
                else d1[key] == d2[key]
                for key in d1
            )

        # somehow this is faster than list(dataset[0])
        dataset_samples = [datasets[0][idx] for idx in range(len(datasets[0]))]
        dataloader_iter = iter(dataloader)
        # sample from dataloader 20 times and check that the samples exist in the dataset
        for _ in range(20):
            sample = next(dataloader_iter)
            assert sample is not None
            # Squeeze all tensors in the sample; for non-tensor values, extract the first element if it's a list
            sample = {
                key: value.squeeze(dim=0)
                if isinstance(value, torch.Tensor)
                else (value[0] if isinstance(value, list) and len(value) > 0 else value)
                for key, value in sample.items()
            }
            assert any(are_tensor_dicts_equal(sample, item) for item in dataset_samples)

    def test_integration_zero_weight_handling(self, train_pipeline_config, datasets_factory):
        """Test handling of datasets with zero weights."""
        datasets = datasets_factory(3)
        dataset_weights = [0.5, 0.0, 0.5]  # Middle dataset has zero weight

        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Verify sample weights for zero-weight dataset are zero
        dataset_lengths = [len(ds) for ds in datasets]
        start_idx = 0

        for i, length in enumerate(dataset_lengths):
            end_idx = start_idx + length
            if dataset_weights[i] == 0.0:
                # All samples from this dataset should have zero weight
                assert torch.all(mixture.sample_weights[start_idx:end_idx] == 0.0)
            else:
                # Samples from this dataset should have non-zero weight
                assert torch.all(mixture.sample_weights[start_idx:end_idx] > 0.0)
            start_idx = end_idx

    @pytest.mark.slow  # 1 sec
    def test_integration_metadata_aggregation(self, train_pipeline_config, datasets_factory):
        """Test that metadata is properly aggregated from multiple datasets."""
        datasets = datasets_factory(2)
        dataset_weights = [0.6, 0.4]

        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Verify aggregated metadata
        assert mixture.meta is not None
        assert hasattr(mixture.meta, "stats")
        assert hasattr(mixture.meta, "features")

        # Verify features are properly configured
        features = mixture.meta.features
        assert "state" in features
        assert "actions" in features
        assert f"camera{train_pipeline_config.num_cams - 1}" in features
        assert f"camera{train_pipeline_config.num_cams}" not in features

    @pytest.mark.slow  # 3 sec
    def test_integration_large_dataset_mixture(self, train_pipeline_config, datasets_factory):
        """Test mixture with many datasets to ensure scalability."""
        num_datasets = 10
        datasets = datasets_factory(num_datasets)
        dataset_weights = [1.0 / num_datasets] * num_datasets

        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Verify all datasets are included
        assert len(mixture.datasets) == num_datasets
        assert len(mixture.dataset_weights) == num_datasets
        assert len(mixture.dataset_names) == num_datasets

        # Verify concatenated dataset
        expected_total_length = sum(len(ds) for ds in datasets)
        assert len(mixture.concatenated_dataset) == expected_total_length

        # Verify sample weights
        assert len(mixture.sample_weights) == expected_total_length
        assert torch.sum(mixture.sample_weights) > 0

        # Should be able to create dataloader
        dataloader = mixture.get_dataloader()
        assert dataloader is not None

    def test_integration_weight_distribution(self, train_pipeline_config, datasets_factory):
        """Test that the weight distribution is correctly applied."""
        # Create datasets with very different sizes
        datasets = datasets_factory(2)

        # Make the second dataset much smaller by mocking its length
        original_length = len(datasets[1])
        datasets[1].__len__ = lambda: original_length // 4  # Make it 4x smaller

        dataset_weights = [0.5, 0.5]  # Equal weights

        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Calculate expected sample weights
        dataset_lengths = [len(ds) for ds in datasets]
        expected_weights = []

        for i, length in enumerate(dataset_lengths):
            if length > 0:
                weight_per_sample = dataset_weights[i] / length
                expected_weights.extend([weight_per_sample] * length)

        # Verify sample weights match expected
        assert len(mixture.sample_weights) == len(expected_weights)
        for i, expected_weight in enumerate(expected_weights):
            assert abs(mixture.sample_weights[i].item() - expected_weight) < 1e-6

    @pytest.mark.slow  # 2 sec
    def test_integration_memory_efficiency(self, train_pipeline_config, datasets_factory):
        """Test that the mixture doesn't cause memory issues with large datasets."""
        # Create a larger number of datasets
        datasets = datasets_factory(5)
        dataset_weights = [0.2] * 5

        mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Verify memory-efficient concatenation
        total_length = sum(len(ds) for ds in datasets)
        assert len(mixture.concatenated_dataset) == total_length

        # Verify sample weights are properly sized
        assert len(mixture.sample_weights) == total_length
        assert mixture.sample_weights.dtype == torch.float64  # Should be double precision

        # Should be able to create dataloader without memory issues
        dataloader = mixture.get_dataloader()
        assert dataloader is not None

    @pytest.mark.slow  # 2 sec
    def test_integration_logging_behavior(self, train_pipeline_config, datasets_factory, caplog):
        """Test that appropriate logging messages are generated."""
        datasets = datasets_factory(3)
        dataset_weights = [0.4, 0.3, 0.3]

        with caplog.at_level(logging.INFO):
            mixture = WeightedDatasetMixture(train_pipeline_config, datasets, dataset_weights, 30.0)

        # Verify logging messages
        assert "Initializing WeightedDatasetMixture" in caplog.text
        assert "Dataset information:" in caplog.text
        assert "Total length of concatenated dataset:" in caplog.text
        assert "Calculating per-sample weights" in caplog.text

        # Test dataloader creation logging
        with caplog.at_level(logging.INFO):
            dataloader = mixture.get_dataloader()  # noqa: F841

        assert "Creating DataLoader" in caplog.text
        assert "DataLoader created successfully" in caplog.text
