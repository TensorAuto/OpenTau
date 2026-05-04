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
from unittest.mock import patch

import pytest
from draccus.utils import ParsingError
from huggingface_hub.constants import CONFIG_NAME

from opentau.configs.policies import (
    PreTrainedConfig,
    _strip_keys,
    load_stripped_config_to_tempfile,
    strip_deprecated_fields_from_json,
    warn_removed_policy_fields,
)
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
    """`_save_pretrained` writes a JSON file at `<dir>/CONFIG_NAME` whose top
    contains the choice-type discriminator (`type`) so the file round-trips
    through `PreTrainedConfig.from_pretrained` (which dispatches off the
    parent class and needs `type` to pick a subclass)."""
    config = get_inherited_pretrainedconfig()

    config._save_pretrained(tmp_path)

    config_path = tmp_path / CONFIG_NAME
    assert config_path.is_file(), f"{config_path} was not written"
    with open(config_path) as f:
        data = json.load(f)
    # The discriminator must be present (the key invariant from_pretrained
    # depends on); for this fixture it's "concrete_test" per its
    # @PreTrainedConfig.register_subclass decorator.
    assert data.get("type") == config.type, (
        f"expected top-level type={config.type!r}, got {data.get('type')!r}"
    )
    # Round-trip: load via the parent class and confirm we get the original
    # subclass back. This is the failure mode that motivated the fix.
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, type(config)), (
        f"round-trip lost subclass: expected {type(config).__name__}, got {type(loaded).__name__}"
    )


def test_from_pretrained(tmp_path):
    with pytest.raises(ParsingError):
        PreTrainedConfig.from_pretrained(pretrained_name_or_path=tmp_path)


def test_from_pretrained_model_exits(tmp_path):
    """
    Tests if from_pretrained downloads config from hugging face
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


def test_strip_keys_top_level_and_nested():
    data = {
        "init_strategy": "expert_only_he_init",
        "chunk_size": 50,
        "policy": {"init_strategy": "no_init", "n_action_steps": 10},
    }
    changed = _strip_keys(data, ("init_strategy",))
    assert changed is True
    assert "init_strategy" not in data
    assert "init_strategy" not in data["policy"]
    assert data["chunk_size"] == 50
    assert data["policy"]["n_action_steps"] == 10


def test_strip_keys_no_change():
    data = {"chunk_size": 50, "policy": {"n_action_steps": 10}}
    changed = _strip_keys(data, ("init_strategy",))
    assert changed is False
    assert data == {"chunk_size": 50, "policy": {"n_action_steps": 10}}


def test_load_stripped_config_to_tempfile_does_not_mutate_source(tmp_path):
    src = tmp_path / "config.json"
    original = {"chunk_size": 50, "init_strategy": "expert_only_he_init"}
    src.write_text(json.dumps(original))

    tmp = load_stripped_config_to_tempfile(src)
    try:
        assert tmp != src
        # Source is byte-for-byte unchanged.
        assert json.loads(src.read_text()) == original
        # Temp copy has the removed key stripped.
        assert json.loads(tmp.read_text()) == {"chunk_size": 50}
    finally:
        tmp.unlink(missing_ok=True)


def test_load_stripped_config_does_not_follow_symlink(tmp_path):
    """HF cache paths are symlinks to content-addressed blobs — make sure load
    never rewrites the linked-to file (which would corrupt the cache)."""
    blob = tmp_path / "blob.json"
    original = {"chunk_size": 50, "init_strategy": "no_init"}
    blob.write_text(json.dumps(original))

    link = tmp_path / "snapshot_link.json"
    link.symlink_to(blob)

    tmp = load_stripped_config_to_tempfile(link)
    try:
        # Both the symlink and the target blob are untouched.
        assert json.loads(link.read_text()) == original
        assert json.loads(blob.read_text()) == original
    finally:
        tmp.unlink(missing_ok=True)


def test_strip_deprecated_fields_from_json_in_place_for_owned_files(tmp_path):
    """The in-place variant is still valid for files we just wrote ourselves."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"chunk_size": 50, "init_strategy": "no_init"}))
    strip_deprecated_fields_from_json(path)
    assert json.loads(path.read_text()) == {"chunk_size": 50}


def test_warn_removed_policy_fields_emits_deprecation_warning(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"init_strategy": "expert_only_he_init", "chunk_size": 50}))

    with pytest.warns(DeprecationWarning, match="init_strategy"):
        warn_removed_policy_fields(path)


def test_warn_removed_policy_fields_silent_when_clean(tmp_path):
    import warnings as _warnings

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"chunk_size": 50}))

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        warn_removed_policy_fields(path)  # must not raise
