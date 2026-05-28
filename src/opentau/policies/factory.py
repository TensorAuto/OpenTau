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

import warnings
from typing import Optional

import numpy as np
from torch import nn

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType
from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import dataset_to_policy_features
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig,
)
from opentau.policies.pi07.low_level.configuration_pi07_low_level import (
    PI07LowLevelConfig,
)
from opentau.policies.pi07_paligemma.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07PaligemmaHighLevelPlannerConfig,
)
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
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
    elif name == "pi05_continuous_state":
        warnings.warn(
            "pi05_continuous_state is deprecated. Use pi05 with state_type='continuous' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from opentau.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "pi05_mem":
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        return PI05MemPolicy
    elif name == "pi06":
        from opentau.policies.pi06.modeling_pi06 import PI06Policy

        return PI06Policy
    elif name == "pi07_paligemma_high_level_planner":
        from opentau.policies.pi07_paligemma.high_level_planner.modeling_pi07_high_level import (
            PI07HighLevelPlannerPolicy as PI07PaligemmaHighLevelPlannerPolicy,
        )

        return PI07PaligemmaHighLevelPlannerPolicy
    elif name == "pi07_paligemma_low_level":
        from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
            PI07PaligemmaLowLevelPolicy,
        )

        return PI07PaligemmaLowLevelPolicy
    elif name == "pi07_high_level":
        from opentau.policies.pi07.high_level_planner.modeling_pi07_high_level import (
            PI07HighLevelPlannerPolicy,
        )

        return PI07HighLevelPlannerPolicy
    elif name == "pi07_low_level":
        from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
            PI07LowLevelPolicy,
        )

        return PI07LowLevelPolicy
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
    elif policy_type == "pi05_continuous_state":
        warnings.warn(
            "pi05_continuous_state is deprecated. Use pi05 with state_type='continuous' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("state_type", "continuous")
        return PI05Config(**kwargs)
    elif policy_type == "pi05_mem":
        return PI05MemConfig(**kwargs)
    elif policy_type == "pi06":
        return PI06Config(**kwargs)
    elif policy_type == "pi07_paligemma_high_level_planner":
        return PI07PaligemmaHighLevelPlannerConfig(**kwargs)
    elif policy_type == "pi07_paligemma_low_level":
        return PI07PaligemmaLowLevelConfig(**kwargs)
    elif policy_type == "pi07_high_level":
        return PI07HighLevelPlannerConfig(**kwargs)
    elif policy_type == "pi07_low_level":
        return PI07LowLevelConfig(**kwargs)
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
    # Per-NORM-HEAD stats; one entry per row of the policy's stacked
    # Normalize/Unnormalize buffer. Built from the mixture's pooled
    # `(robot_type, control_mode)` aggregation when available, else from a
    # single dataset, else from caller-supplied opaque stats.
    per_norm_key_stats: list[dict[str, dict[str, np.ndarray]]] | None = None
    norm_keys: list[str] | None = None
    dataset_to_norm_index: dict[str, int] | None = None

    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        # `DatasetMixtureMetadata` (new attrs) provides per-norm-head stats
        # already pooled across datasets sharing a (robot_type, control_mode).
        # A bare `LeRobotDatasetMetadata` exposes only `.stats`; treat it as
        # a singleton head, deriving the key from its info.
        if hasattr(ds_meta, "per_norm_key_stats") and hasattr(ds_meta, "norm_keys"):
            per_norm_key_stats = list(ds_meta.per_norm_key_stats)
            norm_keys = list(ds_meta.norm_keys)
            dataset_to_norm_index = dict(ds_meta.dataset_to_norm_index)
        else:
            # Inline import keeps `make_policy` from pulling
            # `dataset_mixture` (and transitively LeRobot) at module load.
            from opentau.datasets.dataset_mixture import compute_norm_key

            repo_id = getattr(ds_meta, "repo_id", "default")
            info = getattr(ds_meta, "info", {}) or {}
            norm_key, _ = compute_norm_key(info.get("robot_type"), info.get("control_mode"), repo_id)
            per_norm_key_stats = [ds_meta.stats]
            norm_keys = [norm_key]
            dataset_to_norm_index = {repo_id: 0}

    if not features_already_set:
        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    if stats is not None:
        # External opaque stats override. Accept either a single
        # dict-of-features (wrapped into a singleton list) or an
        # already-listed per-norm-head sequence. We leave `dataset_to_norm_index`
        # as None here — the caller passed stats without identity context,
        # so we can't synthesize a dataset_repo_id -> row mapping; inference
        # must use `dataset_index` or `dataset_names` directly.
        if isinstance(stats, dict):
            per_norm_key_stats = [stats]
            norm_keys = norm_keys or ["external"]
        else:
            per_norm_key_stats = list(stats)
            norm_keys = norm_keys or [f"external_{i}" for i in range(len(per_norm_key_stats))]

    if per_norm_key_stats is not None:
        # The Normalize / Unnormalize kwargs are named `per_dataset_stats` /
        # `dataset_names` for back-compat — the values are now per-norm-head.
        kwargs["per_dataset_stats"] = per_norm_key_stats
        kwargs["dataset_names"] = norm_keys
        # Persist into the policy config so the checkpoint round-trip
        # preserves identity:
        #   - `dataset_names` = the norm-head identifiers (one per buffer row)
        #   - `dataset_to_norm_index` = training dataset name -> row, so
        #     inference can resolve a `dataset_repo_id` even when multiple
        #     datasets share a head.
        cfg.dataset_names = list(norm_keys)
        cfg.dataset_to_norm_index = dict(dataset_to_norm_index) if dataset_to_norm_index is not None else None

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

    # If the checkpoint was saved with save_normalization_stats=False the
    # buffers loaded back as +inf — repopulate from the caller's stats if we
    # have them, otherwise raise a clear error rather than letting the first
    # forward fail mid-step.
    if cfg.pretrained_path and per_norm_key_stats is not None:
        policy._inject_stats(per_norm_key_stats, dataset_names=norm_keys)
    elif cfg.pretrained_path:
        policy._check_norm_stats_loaded()

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
