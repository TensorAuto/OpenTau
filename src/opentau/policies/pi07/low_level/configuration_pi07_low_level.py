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

"""Configuration module for the π07 Low-Level Component.

This module defines the ``PI07LowLevelConfig`` class, which handles
configuration parameters for the π07 low-level component. This component uses
SpaceTimeSiglip (the Gemma 3 SigLIP vision tower wrapped with space-time
separable attention) as a video encoder, processes temporal state sequences
(one continuous token per timestep), and supports optional subtask response,
subgoal image, and metadata conditioning.
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


@PreTrainedConfig.register_subclass("pi07_low_level")
@dataclass
class PI07LowLevelConfig(PreTrainedConfig):
    """Configuration for the π07 low-level component.

    The low-level component generates continuous action chunks via flow matching
    and discrete FAST action tokens through the VLM backbone. It uses
    :class:`SpaceTimeSiglipVideoEncoder` (the Gemma 3 SigLIP vision tower
    wrapped with space-time separable attention) as a video encoder and
    projects temporal state sequences into per-timestep continuous tokens.

    Args:
        n_obs_steps: Number of temporal video frames passed to the
            SpaceTimeSiglip encoder per forward call. Also the count of
            evenly-spaced historical frames the inference buffer keeps
            (the encoder sees the same number of frames at training and
            inference time). Must equal ``dataset_mixture.n_obs_history``.
        history_interval: Temporal stride between the ``n_obs_steps``
            stacked frames in the inference buffer, in environment steps.
            Defaults to 1.
        chunk_size: Trained action-chunk length, i.e. the prediction horizon the
            model always decodes at inference. Upper bound for ``n_action_steps``.
            Defaults to 50.
        n_action_steps: Inference execution horizon -- how many actions from each
            predicted chunk are executed before the policy re-queries with fresh
            observations. Must be <= ``chunk_size``. Defaults to 50.
        normalization_mapping: Mapping of feature names to normalization modes.
        max_state_dim: Maximum dimension for state vectors. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors. Defaults to 32.
        resize_imgs_with_padding: Target ``(H, W)`` for video frame resizing
            with padding. Must match the Gemma 3 vision tower's
            ``image_size`` (the projector hardcodes
            ``patches_per_image = image_size // patch_size``). Defaults to
            ``(448, 448)`` to match
            ``vlm_config.gemma3_config.vision_config.image_size``.
        empty_cameras: Number of empty camera inputs to add. Defaults to 0.
        prompt_max_length: Maximum token length for language prompts.
            Defaults to 256.
        discrete_action_max_length: Maximum length for FAST action tokens.
            Defaults to 32.
        discrete_action_tokenizer_path: HuggingFace repo id or local path of
            the FAST action tokenizer loaded into
            ``self.discrete_action_processor``. Accepts anything
            ``AutoProcessor.from_pretrained(..., trust_remote_code=True)`` can
            resolve. Override this to use a tokenizer specialized to a
            different dataset mixture (see
            ``opentau.scripts.fit_fast_tokenizer``); the value flows through
            to the policy's auxiliary cross-entropy target at training time
            but is unused at inference (the flow-matching head produces
            continuous actions). Defaults to
            ``"physical-intelligence/fast"``.
        metadata_max_length: Maximum token length for metadata strings.
            Defaults to 52.
        response_max_length: Maximum token length for high-level planner
            subtask responses. Defaults to 52.
        proj_width: Width of the action projection layer.  Must match the
            Gemma-v1 action expert's ``hidden_size`` so that suffix embeddings
            are compatible with the expert's transformer layers.  Defaults to
            1280 (the expert hidden size in ``Gemma3WithExpertConfig``).
        dropout: Dropout rate. Defaults to 0.1.
        num_steps: Number of flow-matching denoising steps. Defaults to 5.
        max_delay: Maximum number of prefix action steps for real-time
            inference. Defaults to 0.
        attention_implementation: Attention backend — ``"eager"``, ``"sdpa"``,
            or ``"fa2"``. Defaults to ``"eager"``. ``"sdpa"`` dispatches to
            ``torch.nn.functional.scaled_dot_product_attention`` (typically
            ~2x faster on A100/H100 + bf16). ``"fa2"`` is accepted for
            backward compatibility but logs a warning and falls back to
            eager. Propagated to ``vlm_config`` in ``__post_init__`` so a
            single ``--policy.attention_implementation`` override reaches
            the engine.
        freeze_vision_encoder: Whether to freeze the SigLIP vision tower.
            Defaults to True.
        train_expert_only: Whether to train only the action expert.
            Defaults to False.
        vlm_config: Bundled :class:`Gemma3WithExpertConfig` for the Gemma 3
            VLM backbone + Gemma-v1 action expert.
        spacetime_layer_stride: Wrap every Nth SigLIP encoder layer with
            space-time separable attention.  Defaults to ``4`` (every 4th
            of the 27 SigLIP layers, indices ``[3, 7, 11, 15, 19, 23]``),
            matching the MEM paper / pi05_mem (#171).
        gradient_checkpointing: If True, wrap each SpaceTimeSiglip video
            encoder layer **and** each interleaved Gemma 3 + expert decoder
            layer body in ``torch.utils.checkpoint.checkpoint`` during
            training. Trades roughly one extra forward pass per step
            (~25-33% compute) for a large slice of activation memory per
            rank, enabling larger per-rank batch sizes. Only safe under
            plain DDP (MULTI_GPU), single-process (NO), or DeepSpeed
            ZeRO-1/2 — ``src/opentau/scripts/train.py`` raises if the
            accelerator's distributed_type is anything else (ZeRO-3, FSDP).
            The engine half is plumbed via
            ``vlm_config.gradient_checkpointing`` in ``__post_init__``;
            the video-encoder half is plumbed at model construction time.
            Defaults to False.
    """

    # Input / output structure.
    n_obs_steps: int = 6
    chunk_size: int = 50
    n_action_steps: int = 50

    # ``history_interval`` is the stride between the ``n_obs_steps`` evenly-
    # spaced historical frames in the inference buffer.  Together they
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

    # Video frame preprocessing. Must equal the Gemma 3 vision tower's
    # image_size: `Gemma3MultiModalProjector` hardcodes
    # `patches_per_image = image_size // patch_size`, so feeding a different
    # resolution crashes the projector's reshape.
    resize_imgs_with_padding: tuple[int, int] = (448, 448)

    empty_cameras: int = 0

    # Language Tokenizer
    prompt_max_length: int = 256

    # Maximum length of the action tokens
    discrete_action_max_length: int = 32

    # HF repo id or local path for the FAST action tokenizer
    # (``AutoProcessor.from_pretrained(..., trust_remote_code=True)``).
    discrete_action_tokenizer_path: str = "physical-intelligence/fast"

    metadata_max_length: int = 52

    response_max_length: int = 52

    # Projector
    proj_width: int = 1280

    # Dropout
    dropout: float = 0.1

    # Decoding
    num_steps: int = 5

    # Real Time Inference
    max_delay: int = 0

    # Attention utils
    attention_implementation: str = "eager"

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False

    vlm_config: Gemma3WithExpertConfig = field(
        default_factory=lambda: Gemma3WithExpertConfig(
            freeze_vision_encoder=True,
            train_expert_only=False,
            attention_implementation="eager",
            load_pretrained_gemma3=False,
            dropout=0.1,
        )
    )

    # SpaceTime settings.  Stride 4 matches the MEM paper and pi05_mem
    # (#171): every 4th of the 27 SigLIP layers gets wrapped, indices
    # [3, 7, 11, 15, 19, 23].
    spacetime_layer_stride: int = 4
    gradient_checkpointing: bool = False

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Debug aid: during the training forward, emit a logging.warning when any
    # individual NORMALIZED state/action feature dim (after taking abs) exceeds
    # this value — a symptom of bad normalization stats (e.g. near-zero std on a
    # constant dim) or corrupt data. The warning names the offending
    # source/episode/frame when those batch fields are present, to point at the
    # dataset/frame to inspect. Set to 0.0 (or negative) to disable entirely and
    # skip the per-step device sync. The default 32 is deliberately permissive: the failure it
    # targets drives normalized values to ~1e8, so 32 flags those by a wide margin while
    # tolerating benign large-but-finite dims (the zero-variance guard and pad-aware masking
    # already remove the common false positives).
    warn_outlier_threshold: float = 32.0

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
        """Post-initialization validation and policy → vlm_config plumbing.

        Plumbs the policy-level ``attention_implementation`` and
        ``gradient_checkpointing`` flags into ``vlm_config`` so a single
        ``--policy.attention_implementation`` / ``--policy.gradient_checkpointing``
        CLI override reaches the engine. The video-encoder half of
        ``gradient_checkpointing`` is plumbed separately at model construction
        time (see ``modeling_pi07_low_level.py``). Direct overrides on
        ``--policy.vlm_config.*`` still work and are honoured as-is when the
        policy-level field is at its default.
        """
        super().__post_init__()

        if self.attention_implementation != "eager":
            self.vlm_config.attention_implementation = self.attention_implementation
        if self.gradient_checkpointing:
            self.vlm_config.gradient_checkpointing = self.gradient_checkpointing

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
