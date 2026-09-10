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

"""Configuration for the ``xr1`` policy (Xiaomi-Robotics-1).

``xr1`` is a faithful OpenTau port of **Xiaomi-Robotics-1**
(``XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365``, Apache-2.0), the top-ranked entry on
the RoboCasa365 leaderboard. Architecturally it is a **Qwen3-VL-4B** vision-language
model paired with a **36-layer DiT flow-matching action head** that cross-attends to the
VLM's per-layer key/value cache -- one DiT layer per VLM layer.

Differences from ``cosmos3`` (the closest sibling) that this config has to encode:

* **Observation history.** Four frames per camera at stride 2 (``n_obs_steps=4``,
  ``history_interval=2``), fed to the VLM as three two-frame *videos* rather than as
  still images. The prompt labels the cameras positionally -- Left / Right / **Wrist
  last** -- so camera order is load-bearing, not cosmetic.
* **A 0.95 centre crop** per frame, resized back to 256. ``int(256 * 0.95) = 243`` is
  odd, so the crop box is asymmetric (6 px dropped left, 7 px right).
* **Ascending flow time.** The Euler loop integrates tau 0 -> 1 with ``dt = +1/num_steps``;
  pi05 / cosmos3 descend 1 -> 0. Copying either sampler inverts the flow silently.
* **Identity normalization everywhere.** The reference's RoboCasa365 stats are mean 0 /
  std 1, and the state is not normalized at all -- the state adapter consumes *raw*
  quaternions, so normalizing state before the axis-angle conversion is silently
  catastrophic. ``__post_init__`` refuses a non-IDENTITY STATE/ACTION mapping.

Sizing: Qwen3-VL-4B (36 layers, hidden 2560, 32 query / 8 KV heads, head_dim 128; a
24-block ViT with DeepStack at 5/11/17) + a 0.60B DiT (36 layers, hidden 1024, head_dim
128, so 8 query heads and 8 KV heads -- ``repeat_kv`` is a no-op) = 5.05B total. The
DiT's KV-head count and head_dim **must** match the text tower so its queries attend
against the cached keys/values without reshaping.
"""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
from opentau.policies.utils import validate_state_action_representation_only_config

#: Prompt labels, in camera order. The reference hard-codes these strings *and* their
#: order; ``camera{i}`` keys map positionally onto ``env.camera_name``, so a config that
#: renders the wrist camera second silently tells the model it is looking right.
DEFAULT_CAMERA_PROMPT_LABELS: tuple[str, ...] = ("Left camera: ", "\nRight camera: ", "\nWrist camera: ")

#: Trailing instruction block, verbatim from the reference evaluator.
INSTRUCTION_TEMPLATE = "\n\nGenerate robot actions for the task:\n{instruction} /no_cot"

#: The assistant turn the reference always appends (a "no chain-of-thought" marker).
ASSISTANT_SUFFIX = "<cot></cot>"


@PreTrainedConfig.register_subclass("xr1")
@dataclass
class XR1Config(PreTrainedConfig):
    """Configuration class for the ``xr1`` policy.

    Args:
        n_obs_steps: Observation-history length in *frames*. The reference uses 4.
            A training config must set ``dataset_mixture.n_obs_history`` to the same
            value (``configs/train.py`` enforces the equality).
        history_interval: Temporal stride between the stacked frames, in dataset frames.
            Together with ``n_obs_steps`` this fixes ``obs_buffer_size =
            (n_obs_steps - 1) * history_interval + 1`` (7 at the defaults).
        chunk_size: Trained action-chunk length. 16.
        n_action_steps: Execution horizon. 16 -- the reference executes the whole chunk
            open-loop; a smaller value silently switches to receding-horizon replanning.
        normalization_mapping: Per-feature normalization. **All three must be IDENTITY**
            (see the module docstring); ``__post_init__`` raises otherwise.
        max_state_dim: Width of the *raw* per-frame state the env / dataset emits, before
            the state adapter runs. 16 for RoboCasa365's ``PandaOmron``
            (``base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) +
            gripper_qpos(2)``). This is deliberately **not** the DiT's projector width.
        state_token_dim: Width of each state token the DiT's ``state_projector`` consumes.
            60 in the reference: the 14-D adapted state zero-padded out to 60.
        max_action_dim: Padded action width the flow operates in. 60.
        state_adapter: Which raw-state -> policy-state adapter to apply.
            ``"robocasa_panda_omron"`` maps the 16-D RoboCasa observation to the
            reference's EE-first 14-D vector (two quaternion -> axis-angle conversions);
            ``"identity"`` passes the raw state straight through (zero-padded), for
            datasets that already store the adapted layout.
        quat_order: Quaternion component order in the raw state. ``"xyzw"`` (robosuite's
            convention, and what the reference's ``quat_xyzw_to_axis_angle`` assumes).
        image_size: Square side length every camera frame is fed to the ViT at. 256.
        center_crop_ratio: Centre-crop ratio applied before resizing back to
            ``image_size``. 0.95; set to 1.0 to disable.
        vision_patch_size / vision_temporal_patch_size / vision_merge_size /
            vision_image_mean / vision_image_std: Qwen3-VL patching parameters, mirroring
            the checkpoint's ``video_preprocessor_config.json``.
        video_timestamp_fps: Frame rate the ``<T.T seconds>`` prompt markers use (24, the
            no-metadata fallback the reference lands on).
        suffix_position_offset: Position-id offset for the non-prefix action tokens.
            0 (the released inference code); the 5B training recipe uses 10.
        num_steps: Flow-matching Euler steps at inference. 5.
        max_delay: Inference-time real-time-chunking prefix length (frozen leading rows
            of the chunk). 0 -- the reference evaluates fully open-loop. Bounded above by
            ``train_prefix_max``, since a longer frozen prefix than training ever showed
            the model is extrapolation.
        training_repeat: Number of flow timesteps evaluated per VLM prefix pass during
            training. 4 in the reference recipe. The prefix (~88 % of the FLOPs) runs
            once and the 21-token DiT runs ``training_repeat`` times against it.
        train_prefix_prob: Probability that a training sample gets a non-empty async
            action prefix. 0.5.
        train_prefix_max: Maximum async action-prefix length during training. 6.
        time_beta_alpha / time_beta_beta: ``Beta`` parameters for the training timestep
            draw. The reference samples ``(1 - Beta(1.5, 1).sample()) * 0.999``.
        val_deterministic_time: In validation, replace the random timestep draw with a
            deterministic grid and force ``training_repeat = 1``, so the validation curve
            is not dominated by tau variance. Defaults to True.
        mse_loss_scale: Weight on the masked flow MSE inside the ``"MSE"`` loss key. 0.5.
        freq_loss_weight: Weight on the rFFT-L1 term inside the ``"MSE"`` loss key. 0.5.
        freq_loss_excluded_dims: Action dimensions dropped from the frequency term. The
            reference's ``[17, 18, 19]`` is defined on its canonical 60-D layout, which
            RoboCasa365's flat-12 action has no columns in -- so the RoboCasa365 default
            is empty.
        weight_clamp: ``(low, high)`` clamp on the per-sample async-prefix loss weight.
        n_choices: Number of choice-policy candidates the VLM-side heads predict.
            5 in the reference's 5B checkpoint; the RoboCasa365 export dropped the heads,
            so ``enable_choice_heads`` defaults to False and this is inert until enabled.
        enable_choice_heads: Build and train the ``<state>`` / ``<a_i>`` / ``<score>``
            choice-policy heads. Inference never uses them. Defaults to False.
        pretrained_backbone_repo_id: HF repo id (or local path) supplying the **architecture
            and text assets**: the nested ``vlm_config`` (a ``Qwen3VLConfig``), the
            tokenizer, and the video processor. It is deliberately *not* a weights source
            -- the VLM weights arrive with the XR-1 checkpoint through
            ``XR1Policy.from_pretrained``, whose ``vlm.*`` keys no stock Qwen3-VL loader
            would match.
        load_pretrained_backbone: Whether to fetch that repo's config / tokenizer / video
            processor on construction. ``False`` for CPU tests, which pass an explicit tiny
            ``Qwen3VLConfig`` and never tokenize.
        backbone_weights_repo_id: Optional **stock** Qwen3-VL repo (e.g.
            ``"Qwen/Qwen3-VL-4B-Instruct"``) to warm-start the VLM from when training a DiT
            from scratch. ``None`` (default) leaves the VLM at its random init, which is
            correct whenever an XR-1 checkpoint is loaded on top.
        attention_implementation: Backbone + DiT attention kernel. ``"eager"`` is the
            default because it is what the reference-parity fixtures were captured under;
            ``"sdpa"`` is numerically close but **not** bit-identical.
        dit_*: DiT geometry. ``dit_num_key_value_heads`` and ``dit_head_dim`` must match
            the backbone text tower, and ``dit_num_hidden_layers`` must equal the backbone
            depth (layer *i* of the DiT reads cache layer ``start + i``).
        freeze_input_embeddings: Freeze the VLM's input-embedding table. True in the
            reference recipe -- and worth ~4.7 GB of optimizer state, since the table is
            389 M parameters that never receive a gradient.
        freeze_vision_encoder / train_expert_only / train_vision_encoder_only /
            train_state_action_representation_only: the standard OpenTau freezing matrix.
            ``train_expert_only=True`` here means "train the DiT + projectors only",
            which fits a full XR-1 fine-tune onto a single 80 GB card.
        gradient_checkpointing: Enable the three checkpointing mechanisms
            (ViT wholesale, LLM **MLP only**, DiT layers). Never call
            ``gradient_checkpointing_enable()`` on the text tower -- see
            ``qwen3vl_with_dit.py`` for why that silently empties the KV cache.
        optimizer_* / scheduler_*: the published recipe (LR 3e-5, wd **0.1**, 500-step
            warmup, cosine, clip 1.0).
    """

    # --- Input / output structure ---
    n_obs_steps: int = 4
    history_interval: int = 2
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    max_state_dim: int = 16
    state_token_dim: int = 60
    max_action_dim: int = 60

    state_adapter: str = "robocasa_panda_omron"
    quat_order: str = "xyzw"

    num_cams: int = 3
    camera_prompt_labels: tuple[str, ...] = DEFAULT_CAMERA_PROMPT_LABELS
    image_size: int = 256
    center_crop_ratio: float = 0.95
    prompt_max_length: int = 256

    # Qwen3-VL vision patching. These mirror the checkpoint's
    # ``video_preprocessor_config.json`` so the port does not have to carry that file:
    # 16-px patches, 2 frames per temporal patch, a 2x2 merge window, and the
    # symmetric [-1, 1] normalization Qwen3-VL uses.
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_merge_size: int = 2
    vision_image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    vision_image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Frame rate the ``<T.T seconds>`` prompt markers are computed at. It is a *fallback*:
    # the reference passes no video metadata, so Qwen3-VL warns and defaults to 24, and
    # reproducing the fallback is what makes the markers match.
    video_timestamp_fps: float = 24.0
    # Position-id offset added to the non-prefix action tokens. The released RoboCasa365
    # inference code applies **none**, so this defaults to 0; the 5B training recipe
    # (``xr1/mibot/models/VLA/XR1.py``) adds 10. Keeping training consistent with the
    # checkpoint we fine-tune matters more than matching a recipe that produced a
    # different checkpoint.
    suffix_position_offset: int = 0

    # --- Flow matching ---
    num_steps: int = 5
    max_delay: int = 0
    training_repeat: int = 4
    train_prefix_prob: float = 0.5
    train_prefix_max: int = 6
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    val_deterministic_time: bool = True

    # --- Losses ---
    mse_loss_scale: float = 0.5
    freq_loss_weight: float = 0.5
    freq_loss_excluded_dims: tuple[int, ...] = ()
    weight_clamp: tuple[float, float] = (0.5, 5.0)
    n_choices: int = 5
    enable_choice_heads: bool = False

    # --- Backbone ---
    pretrained_backbone_repo_id: str = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
    load_pretrained_backbone: bool = True
    backbone_weights_repo_id: str | None = None
    attention_implementation: str = "eager"

    # --- DiT geometry ---
    dit_num_hidden_layers: int = 36
    dit_hidden_size: int = 1024
    dit_intermediate_size: int = 4096
    dit_num_key_value_heads: int = 8
    dit_head_dim: int = 128
    dit_time_embed_dim: int = 256
    dit_rms_norm_eps: float = 1e-6

    # --- Freezing / memory ---
    freeze_input_embeddings: bool = True
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_vision_encoder_only: bool = False
    train_state_action_representation_only: bool = False
    gradient_checkpointing: bool = False

    # --- Training presets (the published Xiaomi-Robotics-1 recipe) ---
    optimizer_lr: float = 3e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 120_000
    scheduler_decay_lr: float = 3e-6

    use_torch_compile: bool = False

    def __post_init__(self):
        """Validate the configuration."""
        super().__post_init__()

        if self.train_vision_encoder_only and self.train_expert_only:
            raise ValueError(
                "`train_vision_encoder_only=True` and `train_expert_only=True` are mutually exclusive."
            )
        if self.train_vision_encoder_only and self.freeze_vision_encoder:
            raise ValueError(
                "`train_vision_encoder_only=True` requires `freeze_vision_encoder=False` — the vision "
                "encoder cannot be both frozen and the only trained component."
            )
        validate_state_action_representation_only_config(
            self, policy_name=self.type, has_discrete_actions=False
        )
        self.validate_action_horizon()

        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        if self.history_interval < 1:
            raise ValueError(f"history_interval must be >= 1, got {self.history_interval}.")

        if self.attention_implementation not in ("eager", "sdpa"):
            raise ValueError(
                "xr1 supports attention_implementation in {'eager', 'sdpa'} only (MRoPE + QK-norm "
                f"rule out 'flash_cuda'). Got '{self.attention_implementation}'."
            )

        if self.dit_hidden_size % self.dit_head_dim != 0:
            raise ValueError(
                f"dit_hidden_size ({self.dit_hidden_size}) must be a multiple of dit_head_dim "
                f"({self.dit_head_dim}); the DiT derives its query-head count as the quotient."
            )
        dit_num_attention_heads = self.dit_hidden_size // self.dit_head_dim
        if dit_num_attention_heads % self.dit_num_key_value_heads != 0:
            raise ValueError(
                f"dit query heads ({dit_num_attention_heads}) must be a multiple of "
                f"dit_num_key_value_heads ({self.dit_num_key_value_heads})."
            )

        if self.state_token_dim < self.max_state_dim:
            raise ValueError(
                f"state_token_dim ({self.state_token_dim}) must be >= max_state_dim "
                f"({self.max_state_dim}): the adapted state is zero-padded up to the projector width."
            )

        if self.state_adapter not in ("robocasa_panda_omron", "identity"):
            raise ValueError(
                f"Unknown state_adapter '{self.state_adapter}'; expected 'robocasa_panda_omron' "
                "or 'identity'."
            )
        if self.state_adapter == "robocasa_panda_omron" and self.max_state_dim != 16:
            raise ValueError(
                "state_adapter='robocasa_panda_omron' consumes the 16-D RoboCasa observation "
                "(base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)); "
                f"got max_state_dim={self.max_state_dim}."
            )
        if self.quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"quat_order must be 'xyzw' or 'wxyz', got '{self.quat_order}'.")

        window = self.vision_patch_size * self.vision_merge_size
        if self.image_size % window:
            raise ValueError(
                f"image_size ({self.image_size}) must be a multiple of vision_patch_size * "
                f"vision_merge_size ({window}); Qwen3-VL's merger needs whole merge windows."
            )
        if self.n_obs_steps % self.vision_temporal_patch_size:
            raise ValueError(
                f"n_obs_steps ({self.n_obs_steps}) must be a multiple of "
                f"vision_temporal_patch_size ({self.vision_temporal_patch_size}); the frames are "
                "fed to the vision tower as videos, which patch pairs of frames together."
            )

        if not 0.0 < self.center_crop_ratio <= 1.0:
            raise ValueError(f"center_crop_ratio must be in (0, 1], got {self.center_crop_ratio}.")

        if len(self.camera_prompt_labels) < self.num_cams:
            raise ValueError(
                f"camera_prompt_labels has {len(self.camera_prompt_labels)} entries but num_cams is "
                f"{self.num_cams}; the prompt labels cameras positionally, so every camera needs one."
            )

        if self.max_delay > self.train_prefix_max:
            raise ValueError(
                f"max_delay ({self.max_delay}) exceeds train_prefix_max ({self.train_prefix_max}): the "
                "async-prefix augmentation only ever showed the model prefixes up to "
                f"{self.train_prefix_max} rows long, so a longer frozen inference prefix is "
                "extrapolation. Raise train_prefix_max (and retrain) or lower max_delay."
            )

        # The state adapter consumes RAW quaternions. MEAN_STD- or MIN_MAX-normalizing a
        # quaternion before `quat -> axis-angle` produces a plausible-looking vector that
        # is silently wrong, and IDENTITY actions are what the reference stats encode
        # (mean 0 / std 1), so a non-identity mapping here means the checkpoint's actions
        # are being rescaled against stats it was never trained under.
        for feature in ("STATE", "ACTION", "VISUAL"):
            mode = self.normalization_mapping.get(feature)
            if mode is not None and mode != NormalizationMode.IDENTITY:
                raise ValueError(
                    f"xr1 requires normalization_mapping['{feature}'] == IDENTITY, got {mode}. "
                    "Xiaomi-Robotics-1's RoboCasa365 stats are mean 0 / std 1 and its state adapter "
                    "reads raw quaternions; any other mode silently changes what the model sees."
                )

    def validate_features(self) -> None:
        """Self-populate the input/output features and pin the camera contract.

        ``scripts/eval.py`` builds the policy with ``make_policy(cfg=cfg.policy)`` and no
        dataset metadata, so the features have to be derivable from the config alone.
        Anything already present (a checkpoint's ``config.json``, or an explicit eval
        config) wins; the defaults only fill gaps.
        """
        for i in range(self.num_cams):
            key = f"camera{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, self.image_size, self.image_size)
                )
        if "state" not in self.input_features:
            self.input_features["state"] = PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        if not self.output_features:
            self.output_features["actions"] = PolicyFeature(type=FeatureType.ACTION, shape=(12,))

        n_visual = len(self.image_features)
        if n_visual != self.num_cams:
            raise ValueError(
                f"xr1 expects exactly num_cams={self.num_cams} visual features, found {n_visual} "
                f"({sorted(self.image_features)}). The prompt labels cameras positionally "
                f"({[label.strip() for label in self.camera_prompt_labels[: self.num_cams]]}), so a "
                "mismatched camera count silently relabels them."
            )

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
    def obs_buffer_size(self) -> int:
        """Raw frames the inference observation buffer keeps.

        ``(n_obs_steps - 1) * history_interval + 1`` -- 7 at the defaults, which is
        exactly the reference evaluator's ``deque(maxlen=...)``.
        """
        if self.n_obs_steps <= 1:
            return 1
        return (self.n_obs_steps - 1) * self.history_interval + 1

    @property
    def dit_num_attention_heads(self) -> int:
        """DiT query-head count, derived the way the reference derives it."""
        return self.dit_hidden_size // self.dit_head_dim

    @property
    def dit_query_length(self) -> int:
        """DiT sequence length: ``[sink(1), state(n_obs_steps), noisy_action(chunk_size)]``."""
        return 1 + self.n_obs_steps + self.chunk_size

    @property
    def observation_delta_indices(self) -> None:
        # Genuinely unused: the dataset builds history timestamps from
        # `dataset_mixture.n_obs_history` + `history_interval`, not from this.
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
