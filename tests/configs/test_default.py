import pytest

from src.opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, LOSS_TYPE_MAPPING
from src.opentau.configs.default import DatasetConfig, DatasetMixtureConfig


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
        self.original_loss_mapping = LOSS_TYPE_MAPPING.copy()

    def teardown_method(self):
        """Clean up after each test method."""
        # Restore original state of global mappings
        DATA_FEATURES_NAME_MAPPING.clear()
        DATA_FEATURES_NAME_MAPPING.update(self.original_data_mapping)
        LOSS_TYPE_MAPPING.clear()
        LOSS_TYPE_MAPPING.update(self.original_loss_mapping)

    @pytest.mark.parametrize(
        "data_mapping, loss_mapping, should_raise",
        [
            (None, None, False),  # Both None - valid
            ({"camera0": "image"}, "MSE", False),  # Both provided - valid
            (None, "MSE", True),  # Only loss_mapping provided - invalid
            ({"camera0": "image"}, None, True),  # Only data_mapping provided - invalid
        ],
    )
    def test_data_mapping_validation(self, data_mapping, loss_mapping, should_raise):
        """Test that data_features_name_mapping and loss_type_mapping must be provided together."""
        if should_raise:
            with pytest.raises(
                ValueError,
                match="`data_features_name_mapping` and `loss_type_mapping` have to be provided together.",
            ):
                DatasetConfig(
                    repo_id="test_repo",
                    data_features_name_mapping=data_mapping,
                    loss_type_mapping=loss_mapping,
                )
        else:
            # Should not raise an error
            DatasetConfig(
                repo_id="test_repo", data_features_name_mapping=data_mapping, loss_type_mapping=loss_mapping
            )

    def test_mapping_addition_to_global_dicts(self):
        """Test that mappings are added to global dictionaries when both are provided."""
        test_repo_id = "test_custom_repo"
        test_data_mapping = {"camera0": "observation.image", "state": "observation.state"}
        test_loss_mapping = "MSE"

        # Ensure the repo_id is not already in the mappings
        assert test_repo_id not in DATA_FEATURES_NAME_MAPPING
        assert test_repo_id not in LOSS_TYPE_MAPPING

        # Create DatasetConfig with both mappings
        config = DatasetConfig(  # noqa: F841
            repo_id=test_repo_id,
            data_features_name_mapping=test_data_mapping,
            loss_type_mapping=test_loss_mapping,
        )

        # Check that mappings were added to global dictionaries
        assert test_repo_id in DATA_FEATURES_NAME_MAPPING
        assert test_repo_id in LOSS_TYPE_MAPPING
        assert DATA_FEATURES_NAME_MAPPING[test_repo_id] == test_data_mapping
        assert LOSS_TYPE_MAPPING[test_repo_id] == test_loss_mapping

    def test_mapping_not_added_when_both_none(self):
        """Test that mappings are not added to global dictionaries when both are None."""
        test_repo_id = "test_none_repo"

        # Ensure the repo_id is not already in the mappings
        assert test_repo_id not in DATA_FEATURES_NAME_MAPPING
        assert test_repo_id not in LOSS_TYPE_MAPPING

        # Create DatasetConfig with both mappings as None
        config = DatasetConfig(repo_id=test_repo_id, data_features_name_mapping=None, loss_type_mapping=None)  # noqa: F841

        # Check that mappings were not added to global dictionaries
        assert test_repo_id not in DATA_FEATURES_NAME_MAPPING
        assert test_repo_id not in LOSS_TYPE_MAPPING

    def test_mapping_overwrites_existing(self):
        """Test that providing mappings overwrites existing entries for the same repo_id."""
        test_repo_id = "test_overwrite_repo"
        original_data_mapping = {"old": "mapping"}
        original_loss_mapping = "CE"
        new_data_mapping = {"camera0": "observation.image", "state": "observation.state"}
        new_loss_mapping = "MSE"

        # Add original mappings
        DATA_FEATURES_NAME_MAPPING[test_repo_id] = original_data_mapping
        LOSS_TYPE_MAPPING[test_repo_id] = original_loss_mapping

        # Create DatasetConfig with new mappings
        config = DatasetConfig(  # noqa: F841
            repo_id=test_repo_id,
            data_features_name_mapping=new_data_mapping,
            loss_type_mapping=new_loss_mapping,
        )

        # Check that mappings were overwritten
        assert DATA_FEATURES_NAME_MAPPING[test_repo_id] == new_data_mapping
        assert LOSS_TYPE_MAPPING[test_repo_id] == new_loss_mapping
        assert DATA_FEATURES_NAME_MAPPING[test_repo_id] != original_data_mapping
        assert LOSS_TYPE_MAPPING[test_repo_id] != original_loss_mapping

    def test_empty_mappings(self):
        """Test behavior with empty mappings."""
        test_repo_id = "test_empty_repo"
        empty_data_mapping = {}
        test_loss_mapping = "MSE"

        # Create DatasetConfig with empty data mapping
        config = DatasetConfig(  # noqa: F841
            repo_id=test_repo_id,
            data_features_name_mapping=empty_data_mapping,
            loss_type_mapping=test_loss_mapping,
        )

        # Check that empty mapping was added
        assert test_repo_id in DATA_FEATURES_NAME_MAPPING
        assert test_repo_id in LOSS_TYPE_MAPPING
        assert DATA_FEATURES_NAME_MAPPING[test_repo_id] == empty_data_mapping
        assert LOSS_TYPE_MAPPING[test_repo_id] == test_loss_mapping
