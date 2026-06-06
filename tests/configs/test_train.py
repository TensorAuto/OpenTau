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
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from draccus.utils import ParsingError

from opentau.configs import parser
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.train import TrainPipelineConfig

ARTIFACT_DIR = Path("tests/artifacts/configs")
with open(ARTIFACT_DIR / "train_config.json") as f:
    train_config = json.load(f)


def test_train_config_obj_success(policy_config, dataset_mixture_config, tmp_path):
    """
    Tests if TrainPipelineConfig instance is successfully initialized
    """
    try:
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=policy_config,
            output_dir=tmp_path,
            job_name="test_run",
            seed=42,
            batch_size=8,
        )

        assert cfg.checkpoint_path is None
    except Exception as e:
        pytest.fail(f"Test failed due to {e}")


def test_validate_with_policy_param(policy_config, dataset_mixture_config, tmp_path):
    """
    Tests if validate works with output_dir
    """

    cfg = TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        output_dir=str(tmp_path),
        job_name="test_run",
        seed=42,
        batch_size=8,
        use_policy_training_preset=True,
    )

    cfg.validate()


def test_validate_file_exits(dataset_mixture_config, policy_config, tmp_path):
    """
    Tests if validate raises FileExistsError when empty policy is passed
    """
    output_dir = tmp_path / "configs"

    os.makedirs(output_dir, exist_ok=True)

    cfg = TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        output_dir=Path(output_dir),
        seed=42,
        batch_size=8,
    )

    with pytest.raises(FileExistsError):
        cfg.validate()


def test_validate_without_optimizer(dataset_mixture_config, policy_config):
    """
    Tests if validate raises ValurError when training preset is False and optimizers are also false
    """
    cfg = TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        seed=42,
        batch_size=8,
        use_policy_training_preset=False,
    )

    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_with_policy_path(dataset_mixture_config, tmp_path):
    """
    Tests if validate works when policy path is passed
    """

    with patch.object(parser, "get_path_arg", return_value=tmp_path):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=None,
            output_dir=str(tmp_path),
            job_name="test_run",
            seed=42,
            batch_size=8,
            use_policy_training_preset=True,
        )

        with open(tmp_path / "config.json", "w") as f:
            json.dump(train_config["policy"], f, indent=4)

        cfg.validate()

        assert isinstance(cfg.policy, PreTrainedConfig)


def test_validate_without_policy_config(dataset_mixture_config, tmp_path):
    """
    Tests if validate raises FileExistsError when empty policy path is passed
    """

    with (
        patch.object(parser, "get_path_arg", return_value=None),
        patch.object(parser, "parse_arg", return_value=None),
    ):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=None,
            output_dir=tmp_path,
            job_name="test_run",
            seed=42,
            batch_size=8,
            resume=True,
        )

        with open(tmp_path / "config.json", "w") as f:
            json.dump(train_config["policy"], f, indent=4)

        with pytest.raises(ValueError):
            cfg.validate()


def test_validate_without_policy_dir(dataset_mixture_config, tmp_path):
    """
    Tests if validate raises NotADirectoryError when invalid policy path is passed
    """

    with (
        patch.object(parser, "get_path_arg", return_value=None),
        patch.object(parser, "parse_arg", return_value=tmp_path / "train_config.json"),
    ):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=None,
            output_dir=tmp_path,
            job_name="test_run",
            seed=42,
            batch_size=8,
            resume=True,
        )

        with open(tmp_path / "config.json", "w") as f:
            json.dump(train_config["policy"], f, indent=4)

        with pytest.raises(NotADirectoryError):
            cfg.validate()


def test_validate_without_policy_with_dir(dataset_mixture_config, policy_config, tmp_path):
    """
    Tests if validate parse arg method is used
    """

    with (
        patch.object(parser, "get_path_arg", return_value=None),
        patch.object(parser, "parse_arg", return_value=tmp_path / "config.json"),
    ):
        cfg = TrainPipelineConfig(
            dataset_mixture=dataset_mixture_config,
            policy=policy_config,
            output_dir=tmp_path,
            job_name="test_run",
            seed=42,
            batch_size=8,
            resume=True,
            use_policy_training_preset=True,
        )

        with open(tmp_path / "config.json", "w") as f:
            json.dump(train_config["policy"], f, indent=4)

        cfg.validate()

        assert isinstance(cfg.policy, PreTrainedConfig)
        assert str(cfg.policy.pretrained_path) == str(tmp_path)
        assert str(cfg.checkpoint_path) == str(tmp_path)


def test_save_pretrained(dataset_mixture_config, tmp_path):
    """
    Tests if save_pretrained works properly
    """

    cfg = TrainPipelineConfig(dataset_mixture=dataset_mixture_config, batch_size=8)

    os.makedirs(tmp_path / "test", exist_ok=True)
    cfg._save_pretrained(tmp_path / "test")

    assert os.path.exists(tmp_path / "test")


def test_get_path():
    fields = TrainPipelineConfig.__get_path_fields__()

    assert fields == ["policy"]


def test_to_dict(dataset_mixture_config, dataset_config):
    cfg = TrainPipelineConfig(dataset_mixture=dataset_mixture_config, batch_size=8)

    dict1 = cfg.to_dict()
    assert dict1["dataset_mixture"]["datasets"][0]["repo_id"] == dataset_config.repo_id
    assert dict1["dataset_mixture"]["datasets"][0]["root"] == dataset_config.root
    assert dict1["dataset_mixture"]["datasets"][0]["episodes"] == dataset_config.episodes
    assert isinstance(dict1, dict)


def test_from_pretrained_path_exists(tmp_path):
    """
    Tests if from_pretrained works properly
    """

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    try:
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path)
    except Exception as e:
        pytest.fail(f"The test fail due to {e}")


def test_from_pretrained_file_does_not_exists(tmp_path):
    """
    Tests if from_pretrained raises Parsing Error when empty path is given
    """

    with pytest.raises(ParsingError):
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path)


def test_from_pretrained_path_does_not_exits():
    """
    Tests if from_pretrained raises FileNotFound Error when invalid higging face repo id is passed
    """
    with pytest.raises(FileNotFoundError):
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path="bert123")


def test_from_pretrained_file_exits(tmp_path):
    """
    Tests if from_pretrained downloads config from local directory
    """

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    try:
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path / "train_config.json")
    except Exception as e:
        pytest.fail(f"The test fail due to {e}")


def test_from_pretrained_model_exits(tmp_path):
    """
    Tests if from_pretrained downloads config from hugging face
    """

    repo_id = "ML_GOD/test"

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    with patch("opentau.configs.train.hf_hub_download", return_value=tmp_path / "train_config.json"):
        try:
            TrainPipelineConfig.from_pretrained(pretrained_name_or_path=repo_id)
        except Exception as e:
            pytest.fail(f"The pytests failed due to {e}")


# ---------------------------------------------------------------------------
# Running-best checkpoint config
# ---------------------------------------------------------------------------


def _running_best_cfg(dataset_mixture_config, policy_config, tmp_path, **kwargs):
    """Build a minimal TrainPipelineConfig for exercising running-best validation."""
    return TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        output_dir=tmp_path,
        seed=42,
        batch_size=8,
        **kwargs,
    )


def test_running_best_defaults(dataset_mixture_config, policy_config, tmp_path):
    """The feature is off by default (count 0) with the auto metric."""
    cfg = _running_best_cfg(dataset_mixture_config, policy_config, tmp_path)
    assert cfg.running_best_count == 0
    assert cfg.running_best_metric == "auto"
    assert cfg.running_best_metric_resolved is None


def test_running_best_count_zero_is_disabled(dataset_mixture_config, policy_config, tmp_path):
    """running_best_count == 0 disables the feature (no metric source required)."""
    cfg = _running_best_cfg(
        dataset_mixture_config, policy_config, tmp_path, running_best_count=0, eval_freq=0, val_freq=0
    )
    cfg.env = None
    cfg._validate_running_best()  # must not raise
    assert cfg.running_best_metric_resolved is None


def test_running_best_count_negative_raises(dataset_mixture_config, policy_config, tmp_path):
    """running_best_count must be >= 0."""
    cfg = _running_best_cfg(dataset_mixture_config, policy_config, tmp_path, running_best_count=-1)
    with pytest.raises(ValueError, match="running_best_count"):
        cfg._validate_running_best()


def test_running_best_invalid_metric_raises(dataset_mixture_config, policy_config, tmp_path):
    """running_best_metric must be one of the allowed literals (checked even when disabled)."""
    cfg = _running_best_cfg(dataset_mixture_config, policy_config, tmp_path, running_best_metric="foo")
    with pytest.raises(ValueError, match="running_best_metric"):
        cfg._validate_running_best()


def test_running_best_no_metric_source_raises(dataset_mixture_config, policy_config, tmp_path):
    """Enabling (count >= 1) with no eval and no validation source is an error."""
    cfg = _running_best_cfg(
        dataset_mixture_config, policy_config, tmp_path, running_best_count=1, eval_freq=0, val_freq=0
    )
    cfg.env = None
    with pytest.raises(ValueError, match="no metric source"):
        cfg._validate_running_best()


def test_running_best_eval_metric_requires_env(dataset_mixture_config, policy_config, tmp_path):
    """metric='eval_success' requires a sim eval source (env + eval_freq)."""
    cfg = _running_best_cfg(
        dataset_mixture_config,
        policy_config,
        tmp_path,
        running_best_count=1,
        running_best_metric="eval_success",
        eval_freq=0,
    )
    with pytest.raises(ValueError, match="eval_success"):
        cfg._validate_running_best()


def test_running_best_val_metric_requires_val_freq(dataset_mixture_config, policy_config, tmp_path):
    """metric='val_loss' requires val_freq > 0."""
    cfg = _running_best_cfg(
        dataset_mixture_config,
        policy_config,
        tmp_path,
        running_best_count=1,
        running_best_metric="val_loss",
        val_freq=0,
    )
    with pytest.raises(ValueError, match="val_loss"):
        cfg._validate_running_best()


def test_running_best_auto_resolves_to_eval(dataset_mixture_config, policy_config, tmp_path):
    """'auto' resolves to eval_success when an env + eval_freq are configured."""
    cfg = _running_best_cfg(
        dataset_mixture_config, policy_config, tmp_path, running_best_count=1, eval_freq=100
    )
    cfg.env = object()  # any non-None eval source
    cfg._validate_running_best()
    assert cfg.running_best_metric == "auto"  # serialized field is untouched
    assert cfg.running_best_metric_resolved == "eval_success"


def test_running_best_auto_resolves_to_val(dataset_mixture_config, policy_config, tmp_path):
    """'auto' falls back to val_loss when only validation is configured."""
    cfg = _running_best_cfg(
        dataset_mixture_config, policy_config, tmp_path, running_best_count=1, val_freq=100
    )
    cfg._validate_running_best()
    assert cfg.running_best_metric_resolved == "val_loss"


def test_running_best_with_save_checkpoint_false_is_valid(dataset_mixture_config, policy_config, tmp_path):
    """Enabled running best with save_checkpoint=False is a valid 'keep only the best' config."""
    cfg = _running_best_cfg(
        dataset_mixture_config,
        policy_config,
        tmp_path,
        running_best_count=1,
        save_checkpoint=False,
        val_freq=100,
    )
    cfg._validate_running_best()  # must not raise
    assert cfg.running_best_metric_resolved == "val_loss"


def test_running_best_fields_serialize_roundtrip(dataset_mixture_config, policy_config, tmp_path):
    """The config fields serialize; the runtime-resolved metric does not."""
    cfg = _running_best_cfg(
        dataset_mixture_config,
        policy_config,
        tmp_path,
        running_best_metric="val_loss",
        running_best_count=3,
    )
    d = cfg.to_dict()
    assert d["running_best_metric"] == "val_loss"
    assert d["running_best_count"] == 3
    assert "running_best_metric_resolved" not in d


def test_running_best_backward_compat_old_config(tmp_path):
    """A train_config.json predating the feature loads with the defaults."""
    assert "running_best_count" not in train_config  # the fixture predates the feature
    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path)
    assert cfg.running_best_count == 0
    assert cfg.running_best_metric == "auto"


def test_running_best_validate_integration(dataset_mixture_config, policy_config, tmp_path):
    """The full validate() surfaces a bad running_best_count."""
    cfg = TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        output_dir=str(tmp_path),
        job_name="test_run",
        seed=42,
        batch_size=8,
        use_policy_training_preset=True,
        running_best_count=-1,
    )
    with pytest.raises(ValueError, match="running_best_count"):
        cfg.validate()
