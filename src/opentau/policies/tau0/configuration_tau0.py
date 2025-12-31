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
import logging
from dataclasses import dataclass, field
from typing import Literal

from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("tau0")
@dataclass
class TAU0Config(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    frozen_actions: int = 0  # number of actions from the previous chunk to condition on
    safety_buffer: int = (
        0  # if the action chunk size is smaller than the safety buffer, sample a new action chunk
    )

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

    # num of cameras to pass into the local action expert
    action_expert_num_cams: int = 1

    # Number of image tokens to pass into the local action expert from a single image
    # NOTE: Right now 100 is hardcoded into the SmallCNN image encoder.
    # Ideally, this number is a perfect square.
    num_local_image_tokens: int = 100

    # Tokenizer
    tokenizer_max_length: int = 48  # max number of tokens in the prompt
    response_max_tokens: int = 48  # max number of tokens in the response

    # Projector
    proj_width: int = 1024

    # Dropout
    dropout: float = 0.1

    # Decoding
    num_steps: int = 10

    # Number of cross attention tokens to pass from VLM to action expert
    # If None, then all image and prompt tokens are used; the response tokens are NOT used.
    n_cross_att_tokens: int | None = None

    # CONSTANT: total number of VLM layers
    TOTAL_VLM_LAYERS: int = 18

    """
    `cached_layer` specifies which VLM layer to cache the key and value for cross attention in the action expert.
    If it is a list, list[action_expert_layer_idx] = vlm_layer_idx means that
    the `action_expert_layer_idx`-th layer of the action expert will cross attend on
    the key/value cache from the `vlm_layer_idx`-th layer of the VLM.
    The list must be of length TOTAL_VLM_LAYERS.

    If it is an integer, then all action expert layers will cross attend on
    the key/value cache from the `cached_layer`-th layer of the VLM.

    NOTE: 0 indexed
    """
    use_cache_layer: list[int] | int = 17

    # Initialization strategy
    init_strategy: Literal["no_init", "full_he_init", "expert_only_he_init"] = "full_he_init"

    # Attention utils
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Mean latency of cloud VLM in seconds.
    cloud_vlm_latency_mean: float = 0.16
    # Standard deviation of latency of cloud VLM in seconds.
    cloud_vlm_latency_std: float = 0.05
    # Lower bound of latency of cloud VLM in seconds.
    cloud_vlm_latency_lower: float = 0.10
    # Upper bound of latency of cloud VLM in seconds.
    cloud_vlm_latency_upper: float = 0.25

    # Mean latency of action decoder in seconds.
    action_decoder_latency_mean: float = 0.032
    # Standard deviation of latency of action decoder in seconds.
    action_decoder_latency_std: float = 0.010
    # Lower bound of latency of action decoder in seconds.
    action_decoder_latency_lower: float = 0.020
    # Upper bound of latency of action decoder in seconds.
    action_decoder_latency_upper: float = 0.050

    # TODO: Add EMA

    def __post_init__(self):
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

        # process and validate use_cache_layer
        if isinstance(self.use_cache_layer, list):
            if len(self.use_cache_layer) != self.TOTAL_VLM_LAYERS:
                raise ValueError(
                    f"use_cache_layer must be a list of length {self.TOTAL_VLM_LAYERS}. Got {len(self.use_cache_layer)}."
                )
            for layer_idx in self.use_cache_layer:
                if layer_idx < 0 or layer_idx >= self.TOTAL_VLM_LAYERS:
                    raise ValueError(
                        f"use_cache_layer must be between 0 and {self.TOTAL_VLM_LAYERS - 1}. Got {layer_idx}."
                    )
        elif isinstance(self.use_cache_layer, int):
            if self.use_cache_layer < 0 or self.use_cache_layer >= self.TOTAL_VLM_LAYERS:
                raise ValueError(
                    f"use_cache_layer must be between 0 and {self.TOTAL_VLM_LAYERS - 1}. Got {self.use_cache_layer}."
                )
            self.use_cache_layer = [self.use_cache_layer] * self.TOTAL_VLM_LAYERS
        else:
            raise ValueError(
                f"use_cache_layer must be a list of integers or an integer. Got {type(self.use_cache_layer)}."
            )

        assert self.init_strategy in ["no_init", "full_he_init", "expert_only_he_init"], (
            f"Invalid init strategy: {self.init_strategy} must be one of ['no_init', 'full_he_init', 'expert_only_he_init']"
        )

        if self.init_strategy == "expert_only_he_init" and self.pretrained_path == "lerobot/pi0":
            raise ValueError(
                "You cannot load pretrained PI0 model when init_strategy is 'expert_only_he_init' due to differences in PaliGemma tokenizer vocab sizes."
            )

        if self.pretrained_path is not None and self.pretrained_path != "lerobot/pi0":
            logging.info("Setting init_strategy to 'no_init' because we are resuming from a checkpoint.")
            self.init_strategy = "no_init"

        assert (
            0 <= self.cloud_vlm_latency_lower <= self.cloud_vlm_latency_mean <= self.cloud_vlm_latency_upper
        )
        assert self.cloud_vlm_latency_std >= 0
        assert (
            0
            <= self.action_decoder_latency_lower
            <= self.action_decoder_latency_mean
            <= self.action_decoder_latency_upper
        )
        assert self.action_decoder_latency_std >= 0

        if self.cloud_vlm_latency_lower < self.action_decoder_latency_upper:
            logging.warning(
                "The lower bound of the cloud VLM latency is lower than the upper bound of the action decoder latency. "
                "You risk action decoder lagging behind the cloud VLM, which is out-of-distribution from real world."
            )

        if self.safety_buffer < self.frozen_actions:
            raise ValueError(
                f"The safety buffer must be greater than or equal to the number of frozen actions. Got {self.safety_buffer} for `safety_buffer` and {self.frozen_actions} for `frozen_actions`."
            )

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")
        pass

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
        return list(range(self.chunk_size + self.frozen_actions))

    @property
    def reward_delta_indices(self) -> None:
        return None
