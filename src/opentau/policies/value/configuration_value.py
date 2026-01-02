# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.reward import RewardConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("value")
@dataclass
class ValueConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "VALUE": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state vectors will be padded
    max_state_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images.
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 48

    # Reward config
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        max_episode_length = self.reward_config.reward_normalizer + self.reward_config.C_neg
        assert max_episode_length < abs(self.reward_config.C_neg), (
            "Max episode length should be less than the absolute value of C_neg for proper separation of failed and successful episodes"
        )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
