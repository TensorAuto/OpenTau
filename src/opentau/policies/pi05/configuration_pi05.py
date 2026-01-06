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

"""Configuration module for the PI05 Policy.

This module defines the `PI05Config` class, which handles the configuration parameters
for the PI05 Vision-Language-Action Flow Model. It includes settings for the model architecture,
optimization, scheduling, and data processing.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    """Configuration class for the PI05 Policy.

    This class defines the configuration parameters for the PI05 model, including
    input/output structure, model architecture, training settings, and preprocessing.

    Args:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Size of the action chunk. The upper bound for n_action_steps. Defaults to 50.
        n_action_steps: Number of action steps to predict. Defaults to 50.
        normalization_mapping: Mapping of feature names to normalization modes.
            Defaults to identity for visual features and mean-std for state and action.
        max_state_dim: Maximum dimension for state vectors. Shorter vectors are padded. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors. Shorter vectors are padded. Defaults to 32.
        resize_imgs_with_padding: Target size (height, width) for image resizing with padding.
            Defaults to (224, 224).
        empty_cameras: Number of empty camera inputs to add. Used for specific adaptations like
            Aloha simulation. Defaults to 0.
        tokenizer_max_length: Maximum length for tokenizer. Defaults to 256.
        discrete_action_max_length: Maximum length for discrete action tokens. Defaults to 32.
        proj_width: Width of the projection layer. Defaults to 1024.
        dropout: Dropout rate. Defaults to 0.1.
        num_steps: Number of flow matching steps for decoding. Defaults to 10.
        init_strategy: Initialization strategy. One of "no_init", "full_he_init", "expert_only_he_init".
            Defaults to "full_he_init".
        use_cache: Whether to use KV cache during inference. Defaults to True.
        attention_implementation: Attention implementation to use ("eager" or "fa2"). Defaults to "eager".
        freeze_vision_encoder: Whether to freeze the vision encoder during fine-tuning. Defaults to True.
        train_expert_only: Whether to train only the expert module. Defaults to False.
        optimizer_lr: Learning rate for the optimizer. Defaults to 2.5e-5.
        optimizer_betas: Beta parameters for AdamW optimizer. Defaults to (0.9, 0.95).
        optimizer_eps: Epsilon parameter for AdamW optimizer. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay for AdamW optimizer. Defaults to 1e-10.
        scheduler_warmup_steps: Number of warmup steps for the scheduler. Defaults to 1_000.
        scheduler_decay_steps: Number of decay steps for the scheduler. Defaults to 30_000.
        scheduler_decay_lr: Target learning rate after decay. Defaults to 2.5e-6.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi05_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 256

    # Maximum length of the action tokens
    discrete_action_max_length: int = 32

    # Projector
    proj_width: int = 1024

    # Dropout
    dropout: float = 0.1

    # Decoding
    num_steps: int = 10

    # Initialization strategy
    init_strategy: Literal["no_init", "full_he_init", "expert_only_he_init"] = "full_he_init"

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()

        # TODO(Steven): Validate device and amp? in all policy configs?
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        assert self.init_strategy in ["no_init", "full_he_init", "expert_only_he_init"], (
            f"Invalid init strategy: {self.init_strategy} must be one of ['no_init', 'full_he_init', 'expert_only_he_init']"
        )

        if self.init_strategy == "expert_only_he_init" and self.pretrained_path == "lerobot/pi05":
            raise ValueError(
                "You cannot load pretrained PI0 model when init_strategy is 'expert_only_he_init' due to differences in PaliGemma tokenizer vocab sizes."
            )

        if self.pretrained_path is not None and self.pretrained_path != "lerobot/pi05":
            logging.info("Setting init_strategy to 'no_init' because we are resuming from a checkpoint.")
            self.init_strategy = "no_init"

    def validate_features(self) -> None:
        """Validates the features and adds empty cameras if configured.

        This method checks feature configurations and dynamically adds empty camera inputs
        to `self.input_features` based on the `empty_cameras` parameter.
        """

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        """Returns the default optimizer configuration.

        Returns:
            AdamWConfig: The optimizer configuration with default parameters.
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        """Returns the default scheduler configuration.

        Returns:
            CosineDecayWithWarmupSchedulerConfig: The scheduler configuration with default parameters.
        """
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Indices for observation deltas.

        Returns:
            None: As observation deltas are not used.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Indices for action deltas.

        Returns:
            list[int]: A list of indices corresponding to the chunk size.
        """
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Indices for reward deltas.

        Returns:
            None: As reward deltas are not used.
        """
        return None
