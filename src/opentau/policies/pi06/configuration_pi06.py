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

"""Configuration module for the PI06 Policy.

This module defines the `PI06Config` class, which handles the configuration parameters
for the PI06 Vision-Language-Action model. π06 inherits the π05 training recipe
(FAST discrete action co-training, flow-matching continuous actions, Knowledge
Insulation gradient-stop) but upgrades the backbone to Gemma 3 4B with 448×448
vision and a larger action expert.
"""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


@PreTrainedConfig.register_subclass("pi06")
@dataclass
class PI06Config(PreTrainedConfig):
    """Configuration class for the PI06 Policy.

    Mirrors `PI05Config` but flips the defaults to match π0.6's architecture:
    Gemma 3 4B backbone, 448×448 image input, ~860M action expert, and
    5 flow-matching denoising steps.

    Args:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Trained action-chunk length, i.e. the prediction horizon the model
            always decodes at inference. Upper bound for n_action_steps. Defaults to 50.
        n_action_steps: Inference execution horizon -- how many actions from each
            predicted chunk are executed before the policy re-queries with fresh
            observations. Must be <= chunk_size. Defaults to 50.
        normalization_mapping: Mapping of feature names to normalization modes.
        max_state_dim: Maximum dimension for state vectors. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors. Defaults to 32.
        predict_response: Whether to predict the response. Defaults to False.
            Enabling this is required to reproduce the paper's hierarchical
            design (π0.6 model card §1: "preserves the hierarchical design of
            π0.5, providing high-level subtask prediction and low-level action
            generation"). When False, π0.6 reduces to a flat low-level-only
            model. The default is False because most LeRobot-style datasets do
            not carry subtask annotations. The same field is also used to
            supervise VQA / grounding-style textual targets during co-training
            (this is why the field is named "response" rather than "subtask" —
            it covers both uses, matching the π0.5 pretraining recipe).
        resize_imgs_with_padding: Target image size. Defaults to (448, 448).
        empty_cameras: Number of empty camera inputs to add. Defaults to 0.
            π0.6 pre-training uses up to 4 cameras (base + 2 wrist + optional
            backward for mobile manipulators); set this to match your robot.
        prompt_max_length: Maximum tokenizer length. Defaults to 256.
        response_max_length: Maximum response length. Defaults to 52.
        discrete_action_max_length: Maximum discrete action token length. Defaults to 32.
        proj_width: Width of the action projection layer. Defaults to 1280 to
            match the Gemma-v1 action expert hidden size.
        dropout: Dropout rate. Defaults to 0.1.
        num_steps: Number of flow matching denoising steps. Defaults to 5
            (halved from π0.5's 10, giving ~63 ms per chunk on an H100).
        attention_implementation: "eager", "sdpa", or "fa2". Defaults to "eager".
            "sdpa" dispatches to ``torch.nn.functional.scaled_dot_product_attention``
            (typically 2-3x faster on A100 + bf16). "fa2" is accepted for backward
            compatibility but logs a warning and falls back to eager.
        freeze_vision_encoder: Whether to freeze the vision encoder. Defaults to True.
        train_expert_only: Whether to train only the expert module. Defaults to False.
        optimizer_lr: AdamW learning rate. Defaults to 2.5e-5.
        optimizer_betas: AdamW betas. Defaults to (0.9, 0.95).
        optimizer_eps: AdamW epsilon. Defaults to 1e-8.
        optimizer_weight_decay: AdamW weight decay. Defaults to 1e-10.
        scheduler_warmup_steps: Warmup steps. Defaults to 1_000.
        scheduler_decay_steps: Decay steps. Defaults to 30_000.
        scheduler_decay_lr: Target decay learning rate. Defaults to 2.5e-6.
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
    predict_response: bool = False

    # Image preprocessing: π0.6 raises resolution to 448×448 (π0.5 used 224×224).
    resize_imgs_with_padding: tuple[int, int] = (448, 448)

    # π0.6 training uses up to 4 cameras (base + 2 wrist + optional back camera
    # for mobile manipulators). `empty_cameras` stubs in blank feeds for robots
    # that lack some of these views — same semantics as pi05.
    empty_cameras: int = 0

    # Language Tokenizer
    prompt_max_length: int = 256

    # Response Tokenizer
    response_max_length: int = 52

    # Maximum length of the action tokens
    discrete_action_max_length: int = 32

    # HF repo id or local path for the FAST action tokenizer
    # (``AutoProcessor.from_pretrained(..., trust_remote_code=True)``).
    # Override to use a tokenizer specialized to your mixture (see
    # ``opentau.scripts.fit_fast_tokenizer``).
    discrete_action_tokenizer_path: str = "physical-intelligence/fast"

    # Projector width matches the π0.6 action expert hidden size.
    proj_width: int = 1280

    # Dropout
    dropout: float = 0.1

    # Decoding: π0.6 halves the number of flow-matching denoising steps.
    num_steps: int = 5

    # Real Time Inference: maximum number of frozen actions.
    max_delay: int = 0

    # Attention implementation
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False

    # Wrap each interleaved transformer-layer forward in torch.utils.checkpoint
    # to trade ~25-33% same-batch compute for a large slice of activation memory
    # per rank, typically netting +10-25% throughput once the freed memory is
    # spent on a larger per-rank batch. Only supported with distributed_type=
    # MULTI_GPU (DDP), NO (single process), or DeepSpeed ZeRO-1/2 — src/opentau/
    # scripts/train.py raises if the accelerator's distributed_type is anything
    # else (ZeRO-3, FSDP) because pi06's custom interleaved per-layer forward
    # does not wire up the backend-specific activation-checkpointing hooks
    # those strategies require. Defaults to False (no ckpt, lowest risk).
    gradient_checkpointing: bool = False

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

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
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
