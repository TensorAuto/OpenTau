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

"""Configuration for the cosmos3 policy.

cosmos3 is the π0.5 flow-matching recipe with the PaliGemma backbone swapped for
a **frozen Qwen3-VL-32B** vision-language model (loaded from
``nvidia/Cosmos-Reason2-32B``) and a custom **sub-1B Qwen3-style action expert**.
Continuous actions only -- there is no FAST discrete-action branch and no
response/subtask head, so the discrete/response fields of ``PI05Config`` are
intentionally absent here.

Expert sizing note (the param budget is dominated by attention, not the MLP):
the expert key/value heads (``expert_num_key_value_heads``) and ``expert_head_dim``
**must** match the Qwen3-VL text tower (8 / 128) so the expert's keys/values
concatenate with the backbone's cached KV at every layer. The expert *query*
head count (``expert_num_attention_heads``) is free (any multiple of the KV head
count) because the backbone's prefix is run through stock transformers and only
its KV cache -- never its queries -- is consumed by the expert. With the defaults
below (hidden 1024, 16 query heads, 64 layers, intermediate 2048) the trainable
expert + projections total ~0.91B parameters, comfortably under 1B.
"""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


@PreTrainedConfig.register_subclass("cosmos3")
@dataclass
class Cosmos3Config(PreTrainedConfig):
    """Configuration class for the cosmos3 policy.

    Args:
        n_obs_steps: Number of observation steps. Only ``1`` is supported.
        chunk_size: Trained action-chunk length (prediction horizon). Upper bound
            for ``n_action_steps``. Defaults to 50.
        n_action_steps: Inference execution horizon (<= ``chunk_size``). Defaults to 50.
        normalization_mapping: Per-feature normalization modes. Visual identity,
            state/action mean-std (matches π0.5).
        max_state_dim: Padded proprioceptive-state dimension. Defaults to 32.
        max_action_dim: Padded action dimension. Defaults to 32.
        proj_width: Width of the action/time projection MLPs. Defaults to 1024.
        prompt_max_length: Maximum language-prompt token length. Defaults to 256.
        empty_cameras: Number of empty camera inputs to inject (sim adaptations).
        num_steps: Number of flow-matching Euler denoising steps. Defaults to 10.
        max_delay: Maximum number of frozen action-prefix steps (real-time inference).
        pretrained_backbone_repo_id: HF repo id (or local path) for the Qwen3-VL
            backbone weights. Defaults to ``nvidia/Cosmos-Reason2-32B`` (the
            Cosmos 3 reasoner, a Qwen3-VL-32B fine-tune that loads with stock
            ``transformers.Qwen3VLForConditionalGeneration``).
        load_pretrained_backbone: Whether to download/load the backbone weights on
            construction. Set ``False`` for CPU tests / tiny random configs.
        image_resize: Square side length (pixels) to resize every camera image to
            before the Qwen3-VL vision tower. Bounds the number of vision tokens to
            a fixed, deterministic count. Defaults to 224.
        attention_implementation: ``"eager"`` or ``"sdpa"`` for the expert
            attention. ``"flash_cuda"`` is unsupported (MRoPE/QK-norm). Defaults to ``"sdpa"``.
        freeze_vision_encoder: Freeze the Qwen3-VL vision tower. Defaults to True.
        train_expert_only: Freeze the entire backbone; train only the expert +
            projections. Defaults to True (cosmos3's intended regime).
        gradient_checkpointing: Checkpoint the expert decoder layers to trade
            compute for activation memory. Defaults to False.
        dropout: Dropout probability inside the expert. Defaults to 0.1.
        use_deepstack: Inject Qwen3-VL deepstack vision features into the prefix
            (fidelity to Cosmos-Reason2). Defaults to True.
        expert_hidden_size: Action-expert hidden width. Defaults to 1024.
        expert_intermediate_size: Action-expert SwiGLU MLP width. Defaults to 2048.
        expert_num_hidden_layers: Action-expert depth. MUST equal the backbone text
            tower depth (64 for Qwen3-VL-32B). Defaults to 64.
        expert_num_attention_heads: Action-expert query heads. Free (multiple of
            ``expert_num_key_value_heads``). Defaults to 16.
        expert_num_key_value_heads: Action-expert KV heads. MUST equal the backbone
            text tower (8). Defaults to 8.
        expert_head_dim: Per-head dimension. MUST equal the backbone (128). Defaults to 128.
        expert_adarms_cond_dim: Width of the AdaRMS (time) conditioning vector. Defaults to 256.
        expert_rms_norm_eps: RMSNorm epsilon for the expert. Defaults to 1e-6.
        expert_rope_theta: RoPE base for the expert (matches the backbone, 5e6).
            Note: when the backbone's rotary embedding is reused for the shared
            cos/sin this is informational; kept for standalone expert rotaries.
        optimizer_lr / optimizer_betas / optimizer_eps / optimizer_weight_decay:
            AdamW preset (π0.5 values).
        scheduler_warmup_steps / scheduler_decay_steps / scheduler_decay_lr:
            Cosine-decay-with-warmup preset (π0.5 values).
        use_torch_compile: Whether to ``torch.compile`` the model. Defaults to False
            (enable only after verifying bit-identical seeded runs).
    """

    # --- Input / output structure ---
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

    max_state_dim: int = 32
    max_action_dim: int = 32

    proj_width: int = 1024
    prompt_max_length: int = 256
    empty_cameras: int = 0

    # Flow-matching decoding
    num_steps: int = 10
    max_delay: int = 0

    # --- Backbone (Qwen3-VL-32B via Cosmos-Reason2-32B) ---
    pretrained_backbone_repo_id: str = "nvidia/Cosmos-Reason2-32B"
    load_pretrained_backbone: bool = True
    image_resize: int = 224
    attention_implementation: str = "sdpa"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    gradient_checkpointing: bool = False
    dropout: float = 0.1
    use_deepstack: bool = True

    # --- Action-expert sizing (see module docstring for the hard constraints) ---
    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 2048
    expert_num_hidden_layers: int = 64
    expert_num_attention_heads: int = 16
    expert_num_key_value_heads: int = 8
    expert_head_dim: int = 128
    expert_adarms_cond_dim: int = 256
    expert_rms_norm_eps: float = 1e-6
    expert_rope_theta: float = 5_000_000.0

    # --- Training presets ---
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    use_torch_compile: bool = False

    def __post_init__(self):
        """Validate the configuration."""
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                "for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

        if self.attention_implementation not in ("eager", "sdpa"):
            raise ValueError(
                "cosmos3 supports attention_implementation in {'eager', 'sdpa'} only "
                f"(MRoPE + QK-norm rule out 'flash_cuda'). Got '{self.attention_implementation}'."
            )

        if self.max_delay > self.chunk_size:
            raise ValueError(
                f"The max delay must be <= the chunk size. Got max_delay={self.max_delay} and "
                f"chunk_size={self.chunk_size}."
            )
        if self.n_action_steps < self.chunk_size and self.max_delay != 0:
            raise ValueError(
                "A shortened execution horizon (n_action_steps < chunk_size) is not supported "
                "together with real-time inference delay (max_delay > 0)."
            )

        # Hard concat-attention constraints vs the Qwen3-VL text tower. The exact
        # backbone values (64 layers / 8 KV heads / 128 head_dim) are re-validated
        # against the loaded backbone config at model-build time; here we only
        # enforce the GQA divisibility the expert attention itself requires.
        if self.expert_num_attention_heads % self.expert_num_key_value_heads != 0:
            raise ValueError(
                f"expert_num_attention_heads ({self.expert_num_attention_heads}) must be a multiple "
                f"of expert_num_key_value_heads ({self.expert_num_key_value_heads})."
            )

    def validate_features(self) -> None:
        """Add empty cameras to ``input_features`` if configured."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return the default AdamW optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        """Return the default cosine-decay-with-warmup scheduler configuration."""
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
