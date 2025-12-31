import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from draccus.utils import ParsingError

from src.opentau.configs import parser
from src.opentau.configs.policies import PreTrainedConfig
from src.opentau.configs.train import TrainPipelineConfig

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


def test_validate_without_job_output(policy_config, dataset_mixture_config):
    """
    Tests if validate works without job name and output dir
    """

    cfg = TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        seed=42,
        batch_size=8,
        use_policy_training_preset=True,
    )

    cfg.validate()

    assert cfg.job_name == "tau0"


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
    Tests if from_pretrained  works properly
    """

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    try:
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path)
    except Exception as e:
        pytest.fail(f"The test fail due to {e}")


def test_from_pretrained_file_does_not_exists(tmp_path):
    """
    Tests if from_pretrained  raises Parsing Error when empty path is given
    """

    with pytest.raises(ParsingError):
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path)


def test_from_pretrained_path_does_not_exits():
    """
    Tests if from_pretrained  raises FileNotFound Error when invalid higging face repo id is passed
    """
    with pytest.raises(FileNotFoundError):
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path="bert123")


def test_from_pretrained_file_exits(tmp_path):
    """
    Tests if from_pretrained  downloads config from local directory
    """

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    try:
        TrainPipelineConfig.from_pretrained(pretrained_name_or_path=tmp_path / "train_config.json")
    except Exception as e:
        pytest.fail(f"The test fail due to {e}")


def test_from_pretrained_model_exits(tmp_path):
    """
    Tests if from_pretrained  downloads config from hugging face
    """

    repo_id = "ML_GOD/test"

    with open(tmp_path / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)

    with patch("src.opentau.configs.train.hf_hub_download", return_value=tmp_path / "train_config.json"):
        try:
            TrainPipelineConfig.from_pretrained(pretrained_name_or_path=repo_id)
        except Exception as e:
            pytest.fail(f"The pytests failed due to {e}")
