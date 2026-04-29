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

"""Configuration module for the PI07 high level planner Policy.

This module defines the `PI07HighLevelPlannerConfig` class, which handles the configuration parameters
for the PI07 high level planner. It includes settings for the model architecture,
optimization, scheduling, and data processing.
"""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


@PreTrainedConfig.register_subclass("pi07_paligemma_high_level_planner")
@dataclass
class PI07HighLevelPlannerConfig(PreTrainedConfig):
    """Configuration for the π07 high-level planner policy.

    The high-level planner takes images, a language instruction, robot state,
    and past memory, then autoregressively predicts updated memory and the
    next subtask string. This config controls model architecture, tokenizer
    limits, initialization, and optimizer/scheduler presets.

    Args:
        n_obs_steps: Number of observation steps to use. Only ``1`` is
            currently supported. Defaults to 1.
        normalization_mapping: Mapping from feature type names to
            normalization modes. Defaults to identity for visual features
            and mean-std for state.
        max_state_dim: Maximum dimension for state vectors. Shorter vectors
            are zero-padded. Defaults to 32.
        resize_imgs_with_padding: Target ``(height, width)`` for image
            resizing with aspect-ratio-preserving padding. Defaults to
            ``(224, 224)``.
        empty_cameras: Number of empty (zero-filled) camera inputs to add.
            Defaults to 0.
        prompt_max_length: Maximum token length for the composite language
            prompt (task + past memory + state). Defaults to 256.
        memory_max_length: Maximum token length for the updated memory
            sequence. Defaults to 52.
        response_max_length: Maximum token length for the subtask response
            sequence. Defaults to 52.
        metadata_max_length: Maximum token length for episode metadata
            strings. Defaults to 52.
        subtask_indicator_max_length: Number of tokenizer pieces for the fixed
            ``"Subtask: "`` span (``encode(..., add_special_tokens=False)``). Used to
            align CE slices with the prefix layout; for
            ``google/paligemma-3b-pt-224`` this is 4. Defaults to 4.
        memory_indicator_max_length: Number of tokenizer pieces for the fixed
            ``"Updated Memory: "`` span. Used for documentation and layout checks;
            for ``google/paligemma-3b-pt-224`` this is 4. Defaults to 4.
        dropout: Dropout rate applied in the transformer expert.
            Defaults to 0.1.
        attention_implementation: Attention backend — ``"eager"`` or
            ``"fa2"`` (Flash Attention 2). Defaults to ``"eager"``.
        freeze_vision_encoder: Whether to freeze the SigLIP vision encoder
            during fine-tuning. Defaults to True.
        optimizer_lr: Peak learning rate for AdamW. Defaults to 2.5e-5.
        optimizer_betas: Beta parameters for AdamW. Defaults to (0.9, 0.95).
        optimizer_eps: Epsilon for AdamW. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay for AdamW. Defaults to 1e-10.
        scheduler_warmup_steps: Linear warmup steps. Defaults to 1_000.
        scheduler_decay_steps: Cosine decay steps. Defaults to 30_000.
        scheduler_decay_lr: Final learning rate after decay.
            Defaults to 2.5e-6.
    """

    # Input / output structure.
    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi05_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Language Tokenizer
    prompt_max_length: int = 256

    # Memory Tokenizer
    memory_max_length: int = 52

    # Response Tokenizer
    response_max_length: int = 52

    # Metadata Tokenizer
    metadata_max_length: int = 52

    subtask_indicator_max_length: int = 4

    memory_indicator_max_length: int = 4

    # Dropout
    dropout: float = 0.1

    # Attention utils
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        """Validates configuration values after dataclass initialization.

        Raises:
            ValueError: If ``n_obs_steps`` is not 1.
        """
        super().__post_init__()

        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def validate_features(self) -> None:
        """Adds placeholder camera features for empty camera slots.

        Dynamically inserts zero-filled camera entries into
        ``self.input_features`` for each configured empty camera, so the
        model receives a fixed number of image inputs regardless of which
        cameras are physically present.
        """

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        """Returns the default AdamW optimizer configuration.

        Returns:
            An ``AdamWConfig`` populated from this config's ``optimizer_*``
            fields.
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        """Returns the default cosine-decay-with-warmup scheduler configuration.

        Returns:
            A ``CosineDecayWithWarmupSchedulerConfig`` populated from this
            config's ``scheduler_*`` and ``optimizer_lr`` fields.
        """
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Returns ``None``; observation deltas are not used by this planner."""
        return None

    @property
    def action_delta_indices(self) -> None:
        """Returns ``None``; action deltas are not used by this planner."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """Returns ``None``; reward deltas are not used by this planner."""
        return None
