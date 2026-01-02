#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Factory functions for creating policy instances and configurations.

This module provides utility functions to instantiate policy classes and their
corresponding configurations based on policy names and types. It handles the
logic for creating fresh policies or loading pretrained ones, as well as
parsing features from datasets or environments to properly configure the policies.
"""

from typing import Optional

import numpy as np
from torch import nn

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType
from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import dataset_to_policy_features
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.value.configuration_value import ValueConfig


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Get the policy's class given a name.

    Args:
        name: The name of the policy (e.g., "pi0", "pi05", "value").
            Must match the policy class's `name` attribute.

    Returns:
        type[PreTrainedPolicy]: The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy with the given name is not implemented.
    """
    if name == "pi0":
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
    """Creates a policy configuration object based on the policy type.

    Args:
        policy_type: The type of the policy (e.g., "pi0", "pi05", "value").
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        PreTrainedConfig: An instance of the corresponding policy configuration class.

    Raises:
        ValueError: If the policy type is not available.
    """
    if policy_type == "pi0":
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
        cfg: The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta: Dataset metadata to take input/output shapes and statistics to use for
            (un)normalization of inputs/outputs in the policy. Defaults to None.
        features: Input and output features. Defaults to None.
        stats: Dictionary of statistics for normalization. Defaults to None.
        execution_target: Target for execution. Can be "robot", "cloud", or None.
            None implies unified training. "robot" implies robot action decoder inference.
            "cloud" implies VLM on cloud inference. Defaults to None.

    Returns:
        PreTrainedPolicy: An instance of the created policy.

    Raises:
        ValueError: If neither or both `ds_meta` and `features` are provided when features are not already set in config.
        ValueError: If `execution_target` is invalid.
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
