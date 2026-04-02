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

"""Configuration module for the PI05 Mem Policy.

This module defines the ``PI05MemConfig`` class, which handles configuration
parameters for the PI05 Mem variant. This variant replaces the SigLIP image
encoder with a V-JEPA2 video encoder and processes temporal state sequences
(one continuous token per timestep).
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


@PreTrainedConfig.register_subclass("pi05_mem")
@dataclass
class PI05MemConfig(PreTrainedConfig):
    """Configuration class for the PI05 Mem Policy.

    This variant uses V-JEPA2 as a video encoder (replacing SigLIP) and projects
    temporal state sequences into per-timestep continuous tokens in the VLM
    embedding space.

    Args:
        n_obs_steps: Number of temporal video frames passed to V-JEPA2 per forward call.
            During training each sample has this many frames; during inference the
            observation-history buffer (controlled by ``n_obs_history`` and
            ``history_interval``) is stacked to produce exactly this many frames.
            Must equal ``n_obs_history`` when the latter is set.
        chunk_size: Size of the action chunk. The upper bound for n_action_steps. Defaults to 50.
        n_action_steps: Number of action steps to predict. Defaults to 50.
        normalization_mapping: Mapping of feature names to normalization modes.
        max_state_dim: Maximum dimension for state vectors. Shorter vectors are padded. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors. Shorter vectors are padded. Defaults to 32.
        resize_imgs_with_padding: Target size (height, width) for video frame resizing with padding.
            Defaults to (224, 224).
        empty_cameras: Number of empty camera inputs to add. Defaults to 0.
        prompt_max_length: Maximum length for tokenizer. Defaults to 256.
        discrete_action_max_length: Maximum length for discrete action tokens. Defaults to 32.
        proj_width: Width of the projection layer. Defaults to 1024.
        dropout: Dropout rate. Defaults to 0.1.
        num_steps: Number of flow matching steps for decoding. Defaults to 10.
        init_strategy: Initialization strategy. Defaults to "full_he_init".
        attention_implementation: Attention implementation ("eager" or "fa2"). Defaults to "eager".
        freeze_vision_encoder: Whether to freeze the V-JEPA2 encoder. Defaults to True.
        train_expert_only: Whether to train only the expert module. Defaults to False.
        vjepa2_model_name: HuggingFace repo for V-JEPA2 weights. Defaults to "facebook/vjepa2-vitl-fpc64-256".
        vjepa2_crop_size: Spatial resolution fed to V-JEPA2. Defaults to 224.
        vjepa2_num_video_tokens: Number of output tokens after reduction (must match SigLIP's 256). Defaults to 256.
        vjepa2_perceiver_heads: Number of attention heads in the Perceiver reducer. Defaults to 8.
        vjepa2_dtype: Torch dtype string for V-JEPA2 weights (e.g. "bfloat16", "float32").
            Defaults to None, which uses bfloat16 on CUDA and float32 on CPU.
    """

    # Input / output structure.
    n_obs_steps: int = 8
    chunk_size: int = 50
    n_action_steps: int = 50

    # Observation history for inference buffering.
    # ``n_obs_history`` controls how many evenly-spaced historical frames the
    # inference buffer keeps.  ``history_interval`` is the stride between those
    # frames.  Together they determine ``obs_buffer_size = (n_obs_history-1) *
    # history_interval + 1``.  Typically ``n_obs_history`` should equal
    # ``n_obs_steps`` so the V-JEPA2 encoder sees the same number of frames at
    # training and inference time.
    # Populated from DatasetMixtureConfig during training if unset.
    n_obs_history: int | None = None
    history_interval: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Video frame preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    empty_cameras: int = 0

    # Language Tokenizer
    prompt_max_length: int = 256

    # Maximum length of the action tokens
    discrete_action_max_length: int = 32

    # Projector
    proj_width: int = 1024

    # Dropout
    dropout: float = 0.1

    # Decoding
    num_steps: int = 10

    # Real Time Inference
    max_delay: int = 0

    # Initialization strategy
    init_strategy: Literal["no_init", "full_he_init", "expert_only_he_init"] = "full_he_init"

    # Attention utils
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False

    # V-JEPA2 settings
    vjepa2_model_name: str = "facebook/vjepa2-vitl-fpc64-256"
    vjepa2_crop_size: int = 224
    vjepa2_num_video_tokens: int = 256
    vjepa2_perceiver_heads: int = 8
    vjepa2_dtype: str | None = None

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    @property
    def obs_buffer_size(self) -> int:
        """Total raw frames the observation buffer must keep.

        With ``n_obs_history=T`` and ``history_interval=k``, the buffer stores
        the most recent ``(T-1)*k + 1`` frames so that ``T`` evenly-spaced
        frames can be selected.
        """
        if self.n_obs_history is None or self.n_obs_history <= 1:
            return 1
        return (self.n_obs_history - 1) * (self.history_interval or 1) + 1

    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()

        if self.n_obs_history is not None:
            if not isinstance(self.n_obs_history, int) or self.n_obs_history < 1:
                raise ValueError(
                    f"`n_obs_history` must be None or a positive integer, got {self.n_obs_history}."
                )
            if self.history_interval is None:
                self.history_interval = 1
        if self.history_interval is not None and (
            not isinstance(self.history_interval, int) or self.history_interval < 1
        ):
            raise ValueError(
                f"`history_interval` must be None or a positive integer, got {self.history_interval}."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        assert self.init_strategy in ["no_init", "full_he_init", "expert_only_he_init"], (
            f"Invalid init strategy: {self.init_strategy} must be one of ['no_init', 'full_he_init', 'expert_only_he_init']"
        )

        if self.init_strategy == "expert_only_he_init" and self.pretrained_path == "lerobot/pi05":
            raise ValueError(
                "You cannot load pretrained PI05 model when init_strategy is 'expert_only_he_init' due to differences in PaliGemma tokenizer vocab sizes."
            )

        if self.pretrained_path is not None and self.pretrained_path != "lerobot/pi05":
            logging.info("Setting init_strategy to 'no_init' because we are resuming from a checkpoint.")
            self.init_strategy = "no_init"

        if self.max_delay > self.chunk_size:
            raise ValueError(
                f"The max delay must be less than or equal to the chunk size. Got {self.max_delay} for `max_delay` and {self.chunk_size} for `chunk_size`."
            )

    def validate_features(self) -> None:
        """Validates the features and adds empty cameras if configured."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        """Returns the default optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        """Returns the default scheduler configuration."""
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
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
