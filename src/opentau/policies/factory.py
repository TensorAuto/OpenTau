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

from typing import Optional

import numpy as np
from torch import nn

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import dataset_to_policy_features
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.tau0.configuration_tau0 import TAU0Config
from opentau.policies.value.configuration_value import ValueConfig
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tau0":
        from opentau.policies.tau0.modeling_tau0 import TAU0Policy

        return TAU0Policy
    elif name == "pi0":
        from opentau.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi05":
        from opentau.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "value":
        from opentau.policies.value.modeling_value import ValueFunction

        return ValueFunction
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type in ["tau0"]:
        return TAU0Config(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "value":
        return ValueConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    features: dict[str, FeatureType] | None = None,
    stats: dict[str, dict[str, np.ndarray]] | None = None,
    execution_target: Optional[
        str
    ] = None,  # None for unified training, "robot" for robot action decoder inference, "cloud" for VLM on cloud inference
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        features (dict[str, FeatureType] | None, optional): Input and output features. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    features_already_set = (
        isinstance(cfg.input_features, dict)
        and cfg.input_features
        and isinstance(cfg.output_features, dict)
        and cfg.output_features
    )
    if (bool(ds_meta) + (features is not None) != 1) and not features_already_set:
        raise ValueError("Exactly one of ds_meta or features must be provided.")

    if execution_target not in ["robot", "cloud", None]:
        raise ValueError(
            f"execution_target must be one of ['robot', 'cloud', None], but got {execution_target}."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}

    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats

    if not features_already_set:
        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    if stats is not None:
        kwargs["dataset_stats"] = stats

    if execution_target is not None:
        kwargs["execution_target"] = execution_target

    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
