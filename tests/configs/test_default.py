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

import pytest

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING


@pytest.mark.parametrize(
    "repo_id, grounding, ground_truth", [("", None, True), (None, "", True), (None, None, False)]
)
def test_datasetconfig(repo_id, grounding, ground_truth):
    """
    Tests if datasetConfig object is successfully created
    """
    if ground_truth:
        DatasetConfig(repo_id=repo_id, grounding=grounding)
    else:
        with pytest.raises(ValueError):
            DatasetConfig(repo_id=repo_id, grounding=grounding)


def test_valid_instantiation_with_data():
    """Tests a valid configuration with datasets and matching weights."""
    try:
        DatasetMixtureConfig(
            datasets=[DatasetConfig("repo1"), DatasetConfig("repo2")],
            weights=[0.5, 0.5],
            action_freq=50.0,
            image_resample_strategy="linear",
            vector_resample_strategy="linear",
        )
    except ValueError:
        pytest.fail("DatasetMixtureConfig raised ValueError unexpectedly for a valid configuration.")


def test_mismatched_datasets_and_weights_raises_error():
    """
    Tests that a ValueError is raised if the lengths of datasets and weights are different.
    """
    with pytest.raises(ValueError, match="The length of `weights` must match the length of `datasets`."):
        DatasetMixtureConfig(datasets=[DatasetConfig("repo1")], weights=[0.5, 0.5])


@pytest.mark.parametrize("invalid_freq", [0, -10.5])
def test_invalid_action_freq_raises_error(invalid_freq):
    """
    Tests that a ValueError is raised if action_freq is zero or negative.
    """
    with pytest.raises(ValueError, match=f"`action_freq` must be a positive number, got {invalid_freq}."):
        DatasetMixtureConfig(action_freq=invalid_freq)


def test_invalid_image_resample_strategy_raises_error():
    """
    Tests that a ValueError is raised for an unsupported image_resample_strategy.
    """
    strategy = "invalid_strategy"
    with pytest.raises(
        ValueError,
        match=rf"`image_resample_strategy` must be one of \['linear', 'nearest'\], got {strategy}.",
    ):
        DatasetMixtureConfig(image_resample_strategy=strategy)


def test_invalid_vector_resample_strategy_raises_error():
    """
    Tests that a ValueError is raised for an unsupported vector_resample_strategy.
    """
    strategy = "cubic"
    with pytest.raises(
        ValueError,
        match=rf"`vector_resample_strategy` must be one of \['linear', 'nearest'\], got {strategy}.",
    ):
        DatasetMixtureConfig(vector_resample_strategy=strategy)


class TestDatasetConfigDataMapping:
    """Test class for DatasetConfig data mapping functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Store original state of global mappings
        self.original_data_mapping = DATA_FEATURES_NAME_MAPPING.copy()

    def teardown_method(self):
        """Clean up after each test method."""
        # Restore original state of global mappings
        DATA_FEATURES_NAME_MAPPING.clear()
        DATA_FEATURES_NAME_MAPPING.update(self.original_data_mapping)
