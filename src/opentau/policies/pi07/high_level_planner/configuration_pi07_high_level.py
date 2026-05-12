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
from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig


@PreTrainedConfig.register_subclass("pi07_high_level")
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
            resizing with aspect-ratio-preserving padding. Must match the
            Gemma 3 vision tower's ``image_size`` (the projector hardcodes
            ``patches_per_image = image_size // patch_size``). Defaults to
            ``(448, 448)`` to match ``vlm_config.gemma3_config.vision_config.image_size``.
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
            align CE slices with the prefix layout. MUST equal
            ``len(tokenizer.encode("Subtask: ", add_special_tokens=False))``
            for whatever tokenizer the model uses; otherwise the memory CE
            slice is misaligned. Defaults to 4.
        memory_indicator_max_length: Number of tokenizer pieces for the fixed
            ``"Updated Memory: "`` span. Used for documentation and layout
            checks. MUST equal
            ``len(tokenizer.encode("Updated Memory: ", add_special_tokens=False))``
            for whatever tokenizer the model uses. Defaults to 4.
        dropout: Dropout rate applied in the transformer expert.
            Defaults to 0.1.
        attention_implementation: Attention backend — ``"eager"``, ``"sdpa"``,
            or ``"fa2"``. Defaults to ``"eager"``. ``"sdpa"`` dispatches to
            ``torch.nn.functional.scaled_dot_product_attention`` (typically
            ~2x faster on A100/H100 + bf16). ``"fa2"`` is accepted for
            backward compatibility but logs a warning and falls back to
            eager. The value is propagated to ``vlm_config`` in
            ``__post_init__`` so a single ``--policy.attention_implementation``
            override reaches the engine.
        freeze_vision_encoder: Whether to freeze the SigLIP vision encoder
            during fine-tuning. Defaults to True.
        gradient_checkpointing: Wrap each interleaved Gemma 3 + expert decoder
            layer body in ``torch.utils.checkpoint.checkpoint`` during
            training. Trades roughly one extra forward pass per step
            (~25-33% compute) for a large slice of activation memory per
            rank, enabling larger per-rank batch sizes. Only safe under
            plain DDP (MULTI_GPU), single-process (NO), or DeepSpeed
            ZeRO-1/2 — ``src/opentau/scripts/train.py`` raises if the
            accelerator's distributed_type is anything else (ZeRO-3, FSDP).
            Propagated to ``vlm_config.gradient_checkpointing`` in
            ``__post_init__``. Defaults to False.
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

    # Image preprocessing. Must equal the Gemma 3 vision tower's image_size:
    # `Gemma3MultiModalProjector` hardcodes
    # `patches_per_image = image_size // patch_size`, so feeding a different
    # resolution crashes the projector's reshape.
    resize_imgs_with_padding: tuple[int, int] = (448, 448)

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

    # HF repo id or local path for the FAST action tokenizer
    # (``AutoProcessor.from_pretrained(..., trust_remote_code=True)``).
    # Override to use a tokenizer specialized to your mixture (see
    # ``opentau.scripts.fit_fast_tokenizer``).
    discrete_action_tokenizer_path: str = "physical-intelligence/fast"

    # Dropout
    dropout: float = 0.1

    # Attention utils
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True

    # Activation checkpointing for the engine (Gemma 3 backbone + Gemma-v1
    # expert per-layer body). Plumbed to vlm_config in __post_init__.
    gradient_checkpointing: bool = False

    vlm_config: Gemma3WithExpertConfig = field(
        default_factory=lambda: Gemma3WithExpertConfig(
            freeze_vision_encoder=True,
            train_expert_only=False,
            attention_implementation="eager",
            load_pretrained_gemma3=False,
            dropout=0.1,
            # The high-level planner predicts text autoregressively and never
            # feeds the expert stream (``inputs_embeds=[prefix_embs, None]``
            # at modeling_pi07_high_level.py). Skipping the expert removes
            # ~860M parameters of dead weight from the saved checkpoint and
            # from memory. Override to ``False`` only if you intend to wire
            # the expert into a downstream forward path.
            disable_action_expert=True,
        )
    )

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

        Also plumbs the policy-level ``attention_implementation`` and
        ``gradient_checkpointing`` flags into ``vlm_config`` so a single
        ``--policy.attention_implementation`` / ``--policy.gradient_checkpointing``
        CLI override reaches the engine. Direct overrides on
        ``--policy.vlm_config.*`` still work and are honoured as-is when the
        policy-level field is at its default.

        Raises:
            ValueError: If ``n_obs_steps`` is not 1.
        """
        super().__post_init__()

        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.attention_implementation != "eager":
            self.vlm_config.attention_implementation = self.attention_implementation
        if self.gradient_checkpointing:
            self.vlm_config.gradient_checkpointing = self.gradient_checkpointing

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
