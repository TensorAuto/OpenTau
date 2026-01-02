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

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from draccus.utils import ParsingError
from huggingface_hub.constants import CONFIG_NAME

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, PolicyFeature

ARTIFACT_DIR = Path("tests/artifacts/configs")
with open(ARTIFACT_DIR / "train_config.json") as f:
    train_config = json.load(f)


def test_type_property(get_inherited_pretrainedconfig):
    """Tests that the `type` property correctly calls get_choice_name."""
    config = get_inherited_pretrainedconfig()
    with patch.object(config, "get_choice_name", return_value="my_concrete_type") as mock_get_choice:
        assert config.type == "my_concrete_type"
        mock_get_choice.assert_called_once_with(get_inherited_pretrainedconfig)


@pytest.fixture
def sample_features():
    """Provides a sample set of features for testing properties."""
    return {
        "robot_state": PolicyFeature(type=FeatureType.STATE, shape=(10, 10)),
        "wrist_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(10, 10)),
        "base_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(10, 10)),
        "env_state": PolicyFeature(type=FeatureType.ENV, shape=(10, 10)),
    }


def test_robot_state_feature_property(sample_features, get_inherited_pretrainedconfig):
    """Tests that the robot_state_feature property finds the correct feature."""
    config = get_inherited_pretrainedconfig(input_features=sample_features)
    assert config.robot_state_feature == sample_features["robot_state"]


def test_env_state_feature_property(sample_features, get_inherited_pretrainedconfig):
    """Tests that the env_state_feature property finds the correct feature."""
    config = get_inherited_pretrainedconfig(input_features=sample_features)
    assert config.env_state_feature == sample_features["env_state"]


def test_image_features_property(sample_features, get_inherited_pretrainedconfig):
    """Tests that the image_features property finds all visual features."""
    config = get_inherited_pretrainedconfig(input_features=sample_features)
    expected = {
        "wrist_cam": sample_features["wrist_cam"],
        "base_cam": sample_features["base_cam"],
    }
    assert config.image_features == expected


def test_action_feature_property(get_inherited_pretrainedconfig):
    """Tests that the action_feature property finds the correct feature."""
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(5, 4))}
    config = get_inherited_pretrainedconfig(output_features=output_features)
    assert config.action_feature == output_features["action"]


def test_feature_properties_return_none_when_not_found(get_inherited_pretrainedconfig):
    """Tests that properties correctly return None when no matching feature exists."""
    config = get_inherited_pretrainedconfig(input_features={}, output_features={})
    assert config.robot_state_feature is None
    assert config.env_state_feature is None
    assert config.action_feature is None
    assert config.image_features == {}


def test_save_pretrained(tmp_path, get_inherited_pretrainedconfig):
    """
    Tests that _save_pretrained writes a file using draccus.dump.
    """
    # Setup: Mock the file system and draccus dependencies
    with patch("draccus.dump") as mock_dump, patch("builtins.open", mock_open()):
        config = get_inherited_pretrainedconfig()

        # Act
        config._save_pretrained(tmp_path)

        # Assert
        # Check that open was called with the correct file path
        open.assert_called_once_with(tmp_path / CONFIG_NAME, "w")

        # Check that draccus.dump was called with the config instance
        # We check the first argument of the first call to draccus.dump
        assert mock_dump.call_args[0][0] is config


def test_from_pretrained(tmp_path):
    with pytest.raises(ParsingError):
        PreTrainedConfig.from_pretrained(pretrained_name_or_path=tmp_path)


def test_from_pretrained_model_exits(tmp_path):
    """
    Tests if from_pretrained  downloads config from hugging face
    """

    repo_id = "ML_GOD/test"

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config["policy"], f, indent=4)

    with patch("opentau.configs.policies.hf_hub_download", return_value=tmp_path / "train_config.json"):
        try:
            PreTrainedConfig.from_pretrained(pretrained_name_or_path=repo_id)
        except Exception as e:
            pytest.fail(f"The pytests failed due to {e}")


def test_from_pretrained_path_does_not_exits():
    """
    Tests if from_pretrained raises FIleNotFpund Error when invalid repo id is passed
    """
    with pytest.raises(FileNotFoundError):
        PreTrainedConfig.from_pretrained(pretrained_name_or_path="bert123")
