import pytest
import torch

from opentau.datasets.dataset_mixture import WeightedDatasetMixture
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import dataset_to_policy_features
from opentau.policies.factory import make_policy, make_policy_config
from opentau.policies.tau0.modeling_tau0 import TAU0Policy
from opentau.utils.fake_tensor import FakeTensorContext
from opentau.configs.default import DatasetConfig
from opentau.configs.reward import RewardConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.configs.types import FeatureType, NormalizationMode


@pytest.fixture
def dummy_dataset_metadata(lerobot_dataset_metadata_factory, info_factory, tmp_path):
    # Create only one camera input which is squared to fit all current policy constraints
    # e.g. vqbet and tdmpc works with one camera only, and tdmpc requires it to be squared
    camera_features = {
        "observation.images.laptop": {
            "shape": (84, 84, 3),
            "names": ["height", "width", "channels"],
            "info": None,
        },
    }
    motor_features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
    }
    info = info_factory(
        total_episodes=1, total_frames=1, camera_features=camera_features, motor_features=motor_features
    )
    ds_meta = lerobot_dataset_metadata_factory(root=tmp_path / "init", info=info)
    return ds_meta


@pytest.fixture
def tau0_training_config(train_pipeline_config: TrainPipelineConfig) -> TrainPipelineConfig:
    cfg = train_pipeline_config
    cfg.policy = make_policy_config(
        "tau0",
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
    )
    cfg.dataset_mixture.datasets = [
        DatasetConfig(repo_id="lerobot/droid_100", episodes=[0]),
        DatasetConfig(grounding="clevr"),
    ]
    cfg.dataset_mixture.weights = [1.0, 1.0]
    cfg.resolution = (224, 224)
    ds_meta_features = {
        "state": {
            "shape": (cfg.max_state_dim,),
            "dtype": "float32",
        },
        "actions": {
            "shape": (cfg.action_chunk, cfg.max_action_dim),
            "dtype": "float32",
        },
        "camera0": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
        "camera1": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
        "local_camera0": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
    }
    features = dataset_to_policy_features(ds_meta_features)
    cfg.policy.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.policy.input_features = {
        key: ft for key, ft in features.items() if key not in cfg.policy.output_features
    }
    return cfg


@pytest.fixture
def value_function_training_config(train_pipeline_config: TrainPipelineConfig) -> TrainPipelineConfig:
    cfg = train_pipeline_config
    cfg.policy = make_policy_config(
        "value",
        n_obs_steps=1,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "VALUE": NormalizationMode.MEAN_STD,
        },
        max_state_dim=32,
        tokenizer_max_length=52,
        reward_config=RewardConfig(
            number_of_bins=201,
            C_neg=-1000.0,
            reward_normalizer=400,
            N_steps_look_ahead=50,
        ),
    )

    # Hack: ValueConfig doesn't have max_action_dim but test requires it.
    # We use cfg.max_action_dim from pipeline config (default 32).
    if not hasattr(cfg.policy, "max_action_dim"):
        cfg.policy.max_action_dim = cfg.max_action_dim

    cfg.dataset_mixture.datasets = [
        DatasetConfig(repo_id="lerobot/droid_100", episodes=[0]),
        DatasetConfig(grounding="clevr"),
    ]
    cfg.dataset_mixture.weights = [1.0, 1.0]
    cfg.resolution = (224, 224)

    ds_meta_features = {
        "state": {
            "shape": (cfg.policy.max_state_dim,),
            "dtype": "float32",
        },
        "actions": {
            "shape": (cfg.policy.chunk_size, cfg.policy.max_action_dim),
            "dtype": "float32",
        },
        "camera0": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
        "camera1": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
    }
    features = dataset_to_policy_features(ds_meta_features)
    cfg.policy.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.policy.input_features = {
        key: ft for key, ft in features.items() if key not in cfg.policy.output_features
    }
    return cfg


@pytest.fixture
def pi05_training_config(train_pipeline_config: TrainPipelineConfig) -> TrainPipelineConfig:
    cfg = train_pipeline_config
    cfg.policy = make_policy_config(
        "pi05",
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        discrete_action_max_length=32,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    cfg.dataset_mixture.datasets = [
        DatasetConfig(repo_id="lerobot/droid_100", episodes=[0]),
        DatasetConfig(grounding="clevr"),
    ]
    cfg.dataset_mixture.weights = [1.0, 1.0]
    cfg.resolution = (224, 224)
    ds_meta_features = {
        "state": {
            "shape": (cfg.max_state_dim,),
            "dtype": "float32",
        },
        "actions": {
            "shape": (cfg.action_chunk, cfg.max_action_dim),
            "dtype": "float32",
        },
        "camera0": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
        "camera1": {
            "shape": (3, 224, 224),
            "dtype": "image",
        },
    }
    features = dataset_to_policy_features(ds_meta_features)
    cfg.policy.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.policy.input_features = {
        key: ft for key, ft in features.items() if key not in cfg.policy.output_features
    }
    return cfg


@pytest.fixture
def mixture(tau0_training_config: TrainPipelineConfig) -> WeightedDatasetMixture:
    return make_dataset_mixture(tau0_training_config)


# noinspection PyTypeChecker
@pytest.fixture
def tau0(
    tau0_training_config: TrainPipelineConfig, mixture: WeightedDatasetMixture, device: str
) -> TAU0Policy:
    policy = make_policy(tau0_training_config.policy, ds_meta=mixture.meta)
    policy = policy.to(device, dtype=torch.bfloat16)
    return policy


# noinspection PyTypeChecker
@pytest.fixture
def fake_tensor_tau0(
    tau0_training_config: TrainPipelineConfig, mixture: WeightedDatasetMixture
) -> TAU0Policy:
    with FakeTensorContext():
        policy = make_policy(tau0_training_config.policy, ds_meta=mixture.meta)
        policy = policy.to("cpu", dtype=torch.bfloat16)
        return policy
