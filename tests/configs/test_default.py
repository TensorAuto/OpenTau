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
import warnings
from pathlib import Path

import draccus
import pytest

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING


@pytest.mark.parametrize(
    "repo_id, vqa, ground_truth", [("", None, True), (None, "", True), (None, None, False)]
)
def test_datasetconfig(repo_id, vqa, ground_truth):
    """
    Tests if datasetConfig object is successfully created
    """
    if ground_truth:
        DatasetConfig(repo_id=repo_id, vqa=vqa)
    else:
        with pytest.raises(ValueError):
            DatasetConfig(repo_id=repo_id, vqa=vqa)


def test_valid_instantiation_with_data():
    """Tests a valid configuration with datasets and matching weights."""
    cfg = DatasetMixtureConfig(
        datasets=[DatasetConfig("repo1"), DatasetConfig("repo2")],
        weights=[0.5, 0.5],
        action_freq=50.0,
        image_resample_strategy="linear",
        vector_resample_strategy="linear",
    )
    assert cfg.weights == [0.5, 0.5]


def test_mismatched_datasets_and_weights_raises_error():
    """
    Tests that a ValueError is raised if the lengths of datasets and weights are different.
    """
    with pytest.raises(ValueError, match="The length of `weights` must match the length of `datasets`."):
        DatasetMixtureConfig(datasets=[DatasetConfig("repo1")], weights=[0.5, 0.5])


def test_none_weights_is_valid():
    """Tests that None weights are allowed and defer to runtime inference."""
    cfg = DatasetMixtureConfig(datasets=[DatasetConfig("repo1"), DatasetConfig("repo2")], weights=None)
    assert cfg.weights is None


def test_empty_weights_with_empty_datasets():
    """Tests that an explicit empty weights list is accepted when datasets is also empty."""
    cfg = DatasetMixtureConfig(datasets=[], weights=[])
    assert cfg.weights == []


@pytest.mark.parametrize("invalid_freq", [0, -10.5])
def test_invalid_action_freq_raises_error(invalid_freq):
    """
    Tests that a ValueError is raised if action_freq is zero or negative.
    """
    with pytest.raises(
        ValueError,
        match=rf"`action_freq` must be a positive number or None, got {invalid_freq}\.",
    ):
        DatasetMixtureConfig(action_freq=invalid_freq)


def test_action_freq_defaults_to_none():
    """`action_freq=None` is the new default — each dataset trains at its native fps."""
    cfg = DatasetMixtureConfig()
    assert cfg.action_freq is None


def test_action_freq_none_is_valid():
    """Explicit `None` is accepted and disables resampling."""
    cfg = DatasetMixtureConfig(action_freq=None)
    assert cfg.action_freq is None


def test_action_freq_positive_still_valid():
    """A positive float still pins every dataset to that common rate."""
    cfg = DatasetMixtureConfig(action_freq=30.0)
    assert cfg.action_freq == 30.0


def test_emit_fps_defaults_to_false():
    """The new `emit_fps` toggle is opt-in — pre-PR checkpoints resume cleanly
    because the policy's metadata prefix doesn't gain an unfamiliar ``FPS:``
    segment by default. New training runs that want per-sample fps
    conditioning set `emit_fps=True` explicitly.
    """
    cfg = DatasetMixtureConfig()
    assert cfg.emit_fps is False


def test_emit_fps_can_be_enabled():
    """Setting `emit_fps=True` opts in to per-sample fps tokenization."""
    cfg = DatasetMixtureConfig(emit_fps=True)
    assert cfg.emit_fps is True


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


def test_val_split_ratio_no_warning_when_only_mixture_customized():
    """Setting only the mixture-level `val_split_ratio` must not warn.

    This is the common path users follow after the deprecation; previously
    a per-dataset default of 0.05 made every customized mixture trip a
    false-positive `DeprecationWarning` because every child still had its
    default value.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DatasetMixtureConfig(
            datasets=[DatasetConfig(repo_id="foo/bar"), DatasetConfig(repo_id="baz/qux")],
            val_split_ratio=0.1,
        )
    val_split_warnings = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "val_split_ratio" in str(w.message)
    ]
    assert not val_split_warnings, (
        f"Unexpected val_split_ratio DeprecationWarning(s): {[str(w.message) for w in val_split_warnings]}"
    )


def test_val_split_ratio_no_warning_when_child_overrides():
    """Setting `val_split_ratio` on a child `DatasetConfig` is a supported
    per-dataset override (inherit-on-None, like `tolerance_s`), not a deprecated
    field, so it must NOT emit a DeprecationWarning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DatasetMixtureConfig(
            datasets=[DatasetConfig(repo_id="foo/bar", val_split_ratio=0.2)],
            val_split_ratio=0.1,
        )
    val_split_warnings = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "val_split_ratio" in str(w.message)
    ]
    assert not val_split_warnings, (
        f"Unexpected val_split_ratio DeprecationWarning(s): {[str(w.message) for w in val_split_warnings]}"
    )


def test_dataset_config_val_split_ratio_out_of_range_raises():
    """A per-dataset `val_split_ratio` outside [0, 1] must be rejected at config time."""
    with pytest.raises(ValueError, match=r"`DatasetConfig.val_split_ratio` must be in \[0, 1\]"):
        DatasetConfig(repo_id="foo/bar", val_split_ratio=1.5)


def test_dataset_mixture_config_tolerance_defaults():
    """Mixture-level timestamp-sync defaults match the historical `LeRobotDataset` behavior."""
    cfg = DatasetMixtureConfig()
    assert cfg.tolerance_s == 1e-4
    assert cfg.skip_timestamp_check is False


def test_dataset_config_tolerance_defaults():
    """Per-dataset timestamp-sync overrides default to None (inherit from mixture)."""
    cfg = DatasetConfig(repo_id="foo/bar")
    assert cfg.tolerance_s is None
    assert cfg.skip_timestamp_check is None


def test_invalid_negative_mixture_tolerance_raises():
    """Mixture-level `tolerance_s` must be non-negative."""
    with pytest.raises(ValueError, match=r"`tolerance_s` must be >= 0, got -1\.0"):
        DatasetMixtureConfig(tolerance_s=-1.0)


def test_invalid_negative_per_dataset_tolerance_raises():
    """Per-dataset `tolerance_s` must be non-negative when set."""
    with pytest.raises(
        ValueError,
        match=r"`DatasetConfig\.tolerance_s` must be >= 0 \(or None to inherit\), got -0\.5",
    ):
        DatasetMixtureConfig(datasets=[DatasetConfig(repo_id="foo/bar", tolerance_s=-0.5)])


def test_invalid_negative_bare_dataset_tolerance_raises():
    """Bare `DatasetConfig` must validate `tolerance_s` without going through a mixture."""
    with pytest.raises(
        ValueError,
        match=r"`DatasetConfig\.tolerance_s` must be >= 0 \(or None to inherit\), got -1\.0 for foo/bar",
    ):
        DatasetConfig(repo_id="foo/bar", tolerance_s=-1.0)


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


# transformers.PretrainedConfig codec round-trip — pinned because pi07's
# policy configs declare `vlm_config: Gemma3WithExpertConfig` (a PretrainedConfig
# subclass) as a draccus field, and save_pretrained/from_pretrained walk it.


class TestPretrainedConfigCodec:
    """Verify the encode/decode handlers registered in
    ``opentau.configs.policies`` round-trip ``transformers.PretrainedConfig``
    subclasses through draccus.

    Without these handlers, ``PI07HighLevelPlannerConfig._save_pretrained``
    raises ``Exception("No parser for object ...")`` when draccus tries to
    serialise the nested ``vlm_config: Gemma3WithExpertConfig`` field —
    which is the bug originally surfaced when bootstrapping a π0.7
    high-level / low-level planner checkpoint from public Gemma 3 weights.
    """

    def test_encode_dispatches_via_to_dict_for_subclass(self):
        """``draccus.encode`` resolves to ``obj.to_dict()`` even for a subclass
        of ``PretrainedConfig`` (registered with ``include_subclasses=True``)."""
        from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig

        cfg = Gemma3WithExpertConfig(dropout=0.42)
        encoded = draccus.encode(cfg)

        assert isinstance(encoded, dict)
        assert encoded["dropout"] == 0.42
        # Nested transformers configs serialize as dicts too.
        assert isinstance(encoded["gemma3_config"], dict)
        assert encoded["gemma3_config"]["text_config"]["hidden_size"] == 2560

    def test_decode_reconstructs_correct_subclass(self):
        """``draccus.decode(SubCls, data)`` returns an instance of ``SubCls``,
        not the parent ``PretrainedConfig``."""
        from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig

        original = Gemma3WithExpertConfig(dropout=0.3)
        encoded = draccus.encode(original)
        decoded = draccus.decode(Gemma3WithExpertConfig, encoded)

        assert type(decoded) is Gemma3WithExpertConfig
        assert decoded.dropout == 0.3
        # Nested config also reconstructs to a real PretrainedConfig (not a dict).
        assert decoded.gemma3_config.text_config.hidden_size == 2560

    def test_pi07_high_level_config_save_load_round_trip(self, tmp_path: Path):
        """End-to-end: ``PI07HighLevelPlannerConfig`` round-trips through
        ``_save_pretrained`` -> ``PreTrainedConfig.from_pretrained``, with the
        ``vlm_config`` subtree preserved.

        This is the actual failure mode that broke a π0.7 high-level
        planner checkpoint bootstrap before the codec was registered.
        """
        from opentau.configs.policies import PreTrainedConfig
        from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
            PI07HighLevelPlannerConfig,
        )

        cfg = PI07HighLevelPlannerConfig(dropout=0.25)
        cfg._save_pretrained(tmp_path)

        # The serialised vlm_config must round-trip as a dict (not crash) and
        # must contain the nested Gemma 3 text + vision config blobs.
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert isinstance(data["vlm_config"], dict)
        assert "gemma3_config" in data["vlm_config"]
        assert "gemma_expert_config" in data["vlm_config"]

        loaded = PreTrainedConfig.from_pretrained(tmp_path)
        assert isinstance(loaded, PI07HighLevelPlannerConfig)
        assert loaded.dropout == 0.25
        # vlm_config came back as a real Gemma3WithExpertConfig, not a dict.
        from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig

        assert isinstance(loaded.vlm_config, Gemma3WithExpertConfig)
        assert loaded.vlm_config.gemma3_config.text_config.hidden_size == 2560
