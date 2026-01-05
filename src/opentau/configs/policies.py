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
"""Policy configuration module.

This module provides the base PreTrainedConfig class for policy models, which
defines the interface and common functionality for all policy configurations.
It includes support for feature definitions, normalization modes, and loading
configurations from pretrained models or local paths.
"""

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import OptimizerConfig
from opentau.optim.schedulers import LRSchedulerConfig
from opentau.utils.hub import HubMixin

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False
    pretrained_path: str | None = None

    # Mean latency of cloud VLM in seconds.
    cloud_vlm_latency_mean: float = 0.0
    # Standard deviation of latency of cloud VLM in seconds.
    cloud_vlm_latency_std: float = 0.0
    # Lower bound of latency of cloud VLM in seconds.
    cloud_vlm_latency_lower: float = 0.0
    # Upper bound of latency of cloud VLM in seconds.
    cloud_vlm_latency_upper: float = 0.0

    # Mean latency of action decoder in seconds.
    action_decoder_latency_mean: float = 0.0
    # Standard deviation of latency of action decoder in seconds.
    action_decoder_latency_std: float = 0.0
    # Lower bound of latency of action decoder in seconds.
    action_decoder_latency_lower: float = 0.0
    # Upper bound of latency of action decoder in seconds.
    action_decoder_latency_upper: float = 0.0

    def __post_init__(self):
        """Initialize post-creation attributes.

        This method can be overridden by subclasses to perform additional
        initialization after the dataclass is created.
        """
        pass

    @property
    def type(self) -> str:
        """Get the type name of this configuration.

        Returns:
            The choice name of this configuration class.
        """
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        """Get indices for observation delta features.

        Returns:
            List of indices indicating which observation features should be
            treated as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        """Get indices for action delta features.

        Returns:
            List of indices indicating which action features should be treated
            as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        """Get indices for reward delta features.

        Returns:
            List of indices indicating which reward features should be treated
            as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        """Get the default optimizer configuration for this policy.

        Returns:
            An OptimizerConfig instance with default settings for this policy type.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Get the default learning rate scheduler configuration for this policy.

        Returns:
            An LRSchedulerConfig instance with default settings for this policy type,
            or None if no scheduler should be used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        """Validate that the feature configuration is correct.

        This method should check that all required features are present and
        have valid configurations.

        Raises:
            ValueError: If the feature configuration is invalid.
        """
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Get the robot state feature from input features.

        Returns:
            The PolicyFeature with type STATE if found, or None otherwise.
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """Get the environment state feature from input features.

        Returns:
            The PolicyFeature with type ENV if found, or None otherwise.
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Get all visual/image features from input features.

        Returns:
            Dictionary mapping feature names to PolicyFeature instances with
            type VISUAL.
        """
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Get the action feature from output features.

        Returns:
            The PolicyFeature with type ACTION if found, or None otherwise.
        """
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the configuration to a directory.

        Args:
            save_directory: Directory path where the configuration will be saved.
        """
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """Load a policy configuration from a pretrained model or local path.

        Args:
            cls: The class to instantiate.
            pretrained_name_or_path: Can be either:

                - A string, the model id of a pretrained config hosted inside a model
                  repo on huggingface.co.
                - A path to a directory containing a configuration file saved using
                  the `_save_pretrained` method.
            force_download: Whether to force (re-)downloading the config files and
                configuration from the HuggingFace Hub. Defaults to False.
            resume_download: Whether to resume downloading the config files.
                Defaults to None.
            proxies: Dictionary of proxies to use for requests. Defaults to None.
            token: The token to use as HTTP bearer authorization. If True, will use
                the token generated when running `huggingface-cli login`. Defaults to None.
            cache_dir: Path to a directory in which a downloaded pretrained model
                configuration should be cached. Defaults to None.
            local_files_only: Whether to only look at local files (i.e., do not try
                to download the config). Defaults to False.
            revision: The specific model version to use. It can be a branch name, a
                tag name, or a commit id. Defaults to None.
            **policy_kwargs: Additional keyword arguments. May include 'cli_overrides'
                for command-line argument overrides.

        Returns:
            An instance of the configuration class loaded from the specified path.

        Raises:
            FileNotFoundError: If the configuration file is not found on the
                HuggingFace Hub or in the local path.
        """
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        return draccus.parse(cls, config_file, args=cli_overrides)
