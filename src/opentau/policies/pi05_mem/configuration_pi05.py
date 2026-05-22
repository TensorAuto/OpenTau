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
parameters for the PI05 Mem variant. This variant extends the SigLIP image
encoder from PaliGemma with space-time separable attention every
``spacetime_layer_stride``-th layer (per the MEM paper's low-level memory
architecture), and processes temporal state sequences (one continuous token
per timestep).
"""

from dataclasses import dataclass, field

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

    This variant uses PaliGemma's SigLIP as a space-time video encoder: every
    ``spacetime_layer_stride``-th ViT layer adds a causal temporal attention
    over frames (reusing the layer's existing Q/K/V/O projections — no new
    learnable parameters). Past-timestep tokens are dropped after the encoder
    so the prefix matches a single-frame VLA's 256 image tokens.

    Args:
        n_obs_steps: Number of temporal video frames the video encoder sees
            per forward call. During training the dataloader must be
            configured with ``dataset_mixture.n_obs_history = n_obs_steps``;
            during inference the observation-history buffer is stacked to
            produce exactly ``n_obs_steps`` frames (sampled at
            ``history_interval``).
        history_interval: Temporal stride between stacked frames, in
            environment steps. Defaults to 1.
        chunk_size: Trained action-chunk length, i.e. the prediction horizon the model
            always decodes at inference. Upper bound for n_action_steps. Defaults to 50.
        n_action_steps: Inference execution horizon -- how many actions from each
            predicted chunk are executed before the policy re-queries with fresh
            observations. Must be <= chunk_size. Defaults to 50.
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
        attention_implementation: Attention implementation ("eager", "sdpa", or "fa2"; "fa2"
            falls back to "eager" with a warning). Defaults to "eager".
        freeze_vision_encoder: Whether to freeze the SigLIP vision tower.
            When True the ``multi_modal_projector`` remains trainable, matching
            the semantics in ``pi05_continuous_state``. Defaults to True.
        train_expert_only: Whether to train only the expert module. Defaults to False.
        spacetime_layer_stride: Every ``stride``-th SigLIP encoder layer gets
            the temporal self-attention sublayer added. Defaults to 4, matching
            the MEM paper. The video encoder introduces no new learnable
            parameters and shares ``paligemma.vision_tower`` /
            ``multi_modal_projector`` with ``paligemma_with_expert``, so any
            pi05 checkpoint loads directly with unchanged state_dict keys.
    """

    # Input / output structure.
    n_obs_steps: int = 8
    chunk_size: int = 50
    n_action_steps: int = 50

    # Inference observation-history buffer: ``history_interval`` is the
    # temporal stride between the ``n_obs_steps`` stacked frames. Together they
    # determine ``obs_buffer_size = (n_obs_steps - 1) * history_interval + 1``.
    history_interval: int = 1

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

    # HF repo id or local path for the FAST action tokenizer
    # (``AutoProcessor.from_pretrained(..., trust_remote_code=True)``).
    # Override to use a tokenizer specialized to your mixture (see
    # ``opentau.scripts.fit_fast_tokenizer``).
    discrete_action_tokenizer_path: str = "physical-intelligence/fast"

    # Projector
    proj_width: int = 1024

    # Dropout
    dropout: float = 0.1

    # Decoding
    num_steps: int = 10

    # Real Time Inference
    max_delay: int = 0

    # Attention utils
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False

    # Wrap each transformer-layer forward in torch.utils.checkpoint to trade
    # ~25-33% same-batch compute for ~30-40 GB of activation memory per rank,
    # typically netting +10-25% throughput once the freed memory is spent on
    # a larger per-rank batch. Only supported with distributed_type=MULTI_GPU
    # (DDP), NO (single process), or DeepSpeed ZeRO-1/2 — src/opentau/scripts/
    # train.py raises if the accelerator's distributed_type is anything else
    # (ZeRO-3, FSDP) because the custom per-layer forward does not wire up
    # the backend-specific activation-checkpointing hooks those strategies
    # require. Defaults to False (no ckpt, lowest risk).
    gradient_checkpointing: bool = False

    # Space-time SigLIP video encoder settings (MEM paper low-level memory).
    # The encoder wraps paligemma_with_expert's own vision_tower / projector
    # by reference and adds zero new parameters, so there is no separate
    # model-name / dtype field here.
    spacetime_layer_stride: int = 4

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

        With ``n_obs_steps=T`` and ``history_interval=k``, the buffer stores
        the most recent ``(T-1)*k + 1`` frames so that ``T`` evenly-spaced
        frames can be selected.
        """
        if self.n_obs_steps <= 1:
            return 1
        return (self.n_obs_steps - 1) * self.history_interval + 1

    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()

        if not isinstance(self.n_obs_steps, int) or self.n_obs_steps < 1:
            raise ValueError(f"`n_obs_steps` must be a positive integer, got {self.n_obs_steps}.")
        if not isinstance(self.history_interval, int) or self.history_interval < 1:
            raise ValueError(f"`history_interval` must be a positive integer, got {self.history_interval}.")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.max_delay > self.chunk_size:
            raise ValueError(
                f"The max delay must be less than or equal to the chunk size. Got {self.max_delay} for `max_delay` and {self.chunk_size} for `chunk_size`."
            )

        if self.n_action_steps < self.chunk_size and self.max_delay != 0:
            raise ValueError(
                "A shortened execution horizon (n_action_steps < chunk_size) is not yet "
                "supported together with real-time inference delay (max_delay > 0); they "
                "would entangle the action-queue prefix logic. Got "
                f"n_action_steps={self.n_action_steps}, chunk_size={self.chunk_size}, "
                f"max_delay={self.max_delay}."
            )

        if not isinstance(self.spacetime_layer_stride, int) or self.spacetime_layer_stride < 1:
            raise ValueError(
                f"`spacetime_layer_stride` must be a positive integer, got {self.spacetime_layer_stride}."
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
