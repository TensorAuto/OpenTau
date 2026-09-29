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

"""Configuration for the flux3_action policy.

flux3_action is Black Forest Labs' **FLUX 3 Action** (F3A), a 7B world-action model
ported from ``black-forest-labs/flux-action`` (see ``VENDOR.md``). Unlike every other
policy in this repo, F3A does **not** have a separable action head: its trunk denoises
action tokens and *video* tokens jointly in one packed sequence, and upstream marks the
video streams as required --

    REQUIRED_CONTENT_STREAMS = ("video", "video_cond")     # vendored config.py

-- so a "drop the video head and keep the actions" port is not a lighter variant of this
policy, it is a non-functional one. At inference the video tokens are denoised from
noise conditioned on the observed frame, which is why eval needs no future frames even
though training does (upstream's ``prepare`` demands ``window_frames`` per camera,
"observations plus future frames").

This config is the OpenTau-facing surface. The upstream-native settings live on
``flux_action.config.PolicyConfig``, which :meth:`Flux3ActionConfig.to_policy_config`
builds -- deliberately, so that upstream's own ``__post_init__`` validation is the single
source of truth for cross-field constraints rather than being paraphrased (and drifting)
here.

Checkpoint layout (three artifacts, mirroring upstream's Hugging Face repos):

* ``trunk_weights``   -- the action DiT (``black-forest-labs/flux-3-action-droid`` or
  ``-so101``), loaded by ``models/wiring.py::load_action_checkpoint``, which filters the
  unused content streams, remaps the co-trained ``action_prediction.*`` backbone onto
  this embodiment's modality, and seeds any missing embodiment heads deterministically
  from ``head_init_seed`` so every rank builds identical heads.
* ``video_vae_id``    -- the frozen video VAE (``black-forest-labs/flux-3-action-base``).
* ``text_encoder_id`` -- the frozen Qwen3-VL text tower (same base repo).
"""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)

from .config import PolicyConfig


@PreTrainedConfig.register_subclass("flux3_action")
@dataclass
class Flux3ActionConfig(PreTrainedConfig):
    """Configuration class for the flux3_action policy.

    Args:
        chunk_size: Trained action-chunk length. F3A's checkpoints are trained at 32
            (2.13s of motion at 15 fps), which is also what the released inference
            settings assume; changing it invalidates the pretrained action head.
        n_action_steps: Inference execution horizon (<= ``chunk_size``). The released
            DROID recipe runs the full 32 open loop.
        n_obs_steps: Observation frames. Only ``1`` is valid unless
            ``inference_profile="history"`` -- upstream enforces this pairing.
        action_dim: Real action width before OpenTau's padding to ``max_action_dim``.
        camera_layout: How cameras composite onto one canvas: ``"droid"`` (wrist
            full-res above two half-res exteriors), ``"single"``, ``"side_by_side"`` or
            ``"grid"``. The first three are the layouts the released checkpoints were
            trained with; ``"grid"`` runs anywhere but matches no checkpoint.
        camera_keys: Batch stream names in layout order. Upstream resolves cameras by a
            plain ``batch[key]`` lookup, so these are OpenTau's own
            ``observation.images.*`` keys rather than upstream's ``images.*`` spelling --
            which is what lets the frames be read where they already live, with no
            translation layer. The DROID layout expects [wrist, left exterior, right
            exterior] in that order.
        canvas_hw: Composited canvas size fed to the video VAE.
        fps: Control rate the action times are built against.
        action_parameterization: ``"absolute"`` (DROID) or ``"joint_delta"`` (SO-101).
        trunk_weights: Action-DiT checkpoint (repo id or local path).
        video_vae_id: Frozen video-VAE checkpoint. Required -- see the module docstring.
        text_encoder_id: Frozen Qwen3-VL text tower.
        quantization: ``None`` only. Upstream's ``"fp8r"`` path needs
            ``models/transformer_inf_fp8r.py``, which is deliberately not vendored.
    """

    # --- OpenTau horizon surface -------------------------------------------------
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # F3A does its own q01/q99 range normalization inside the policy
            # (``action_normalization`` / ``state_normalization``), so OpenTau's
            # normalizer is identity on every stream and must stay that way -- running
            # both would double-normalize the action targets.
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32

    # --- upstream-native geometry ------------------------------------------------
    action_dim: int = 8
    action_modality: str = "action"
    camera_layout: str = "droid"
    camera_keys: tuple[str, ...] = (
        "observation.images.wrist",
        "observation.images.left",
        "observation.images.right",
    )
    canvas_hw: tuple[int, int] = (544, 736)
    fps: float = 15.0
    action_scale: float = 2.0
    gripper_flip_dims: tuple[int, ...] = (-1,)
    action_parameterization: str = "absolute"
    absolute_action_dims: tuple[int, ...] = ()
    action_normalization: dict[str, list[float]] | None = None
    state_normalization: dict[str, list[float]] | None = None
    normalization_clip: float = 6.0
    camera_dropout: dict[str, float] = field(default_factory=dict)
    empty_cameras: int = 0

    # --- checkpoints -------------------------------------------------------------
    trunk_weights: str | None = None
    video_vae_id: str | None = None
    text_encoder_id: str | None = None
    dit_config: dict = field(default_factory=dict)
    content_streams: tuple[str, ...] | None = None
    head_init_seed: int = 0

    # --- runtime -----------------------------------------------------------------
    torch_dtype: str = "bfloat16"
    quantization: str | None = None
    attn_mode: str = "torch"
    compile_model: bool = False
    single_frame_encode: bool = True
    inference_profile: str = "default"
    history_snapshots: int = 1
    condition_on_past_actions: bool = False
    video_position_fps: float = 24.0
    text_fixed_length: int = 320

    # --- sampling (no preset is implied; these are explicit choices) --------------
    sampler: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_action: float | None = None
    sampler_shift: float | None = None
    inference_seed: int = 0

    # --- training recipe ---------------------------------------------------------
    train_timestep_width: float = 0.75
    train_timestep_shift: float = 42.0
    separate_timesteps: bool = False
    video_logit_mean: float = 1.08
    video_logit_std: float = 1.0
    conditioning_noise_max: float = 0.0
    loss_reduction: str = "joint_tokens"
    action_channel_weights: list[float] | None = None
    action_loss_weight: float = 50.0
    video_loss_weight: float = 1.0
    caption_dropout: float = 0.1
    augment: bool = True
    vae_batch_windows: int = 8

    # --- optimizer / scheduler ---------------------------------------------------
    optimizer_lr: float = 1.92e-4
    optimizer_lr_heads_multiplier: float = 5.0
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.quantization is not None:
            # The only import site is lazy (vendored ``policy.py``, under this exact
            # value), so an unvendored fp8r module would surface as a confusing
            # ImportError deep inside ``from_pretrained``. Fail here instead, naming
            # the reason.
            raise ValueError(
                f"quantization={self.quantization!r} is not supported: "
                "models/transformer_inf_fp8r.py is deliberately not vendored "
                "(see VENDOR.md). Use quantization=None."
            )
        non_identity = {
            feature: mode
            for feature, mode in self.normalization_mapping.items()
            if mode != NormalizationMode.IDENTITY
        }
        if non_identity:
            # The wrapper never invokes the Normalize modules -- F3A range-normalizes
            # state and actions itself from its own q01/q99 bounds. A non-IDENTITY mode
            # would therefore be accepted and then silently ignored, which reads as
            # "normalization is configured" while nothing applies it.
            raise ValueError(
                "flux3_action normalizes state and actions internally, so every "
                f"normalization_mapping mode must be IDENTITY; got {non_identity}. "
                "Set the q01/q99 bounds via action_normalization / state_normalization "
                "instead."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) must be <= max_action_dim ({self.max_action_dim})."
            )
        # Build the upstream config once so its own cross-field validation runs at
        # construction time rather than at the first forward.
        self.to_policy_config()

    def to_policy_config(self) -> PolicyConfig:
        """Build the upstream-native :class:`PolicyConfig` this config describes.

        Upstream's ``__post_init__`` owns every cross-field constraint (history/profile
        pairing, gripper flips, loss reduction, channel weights, ...). Delegating rather
        than restating them here is what keeps this file from drifting out of agreement
        with the vendored code it configures.
        """
        return PolicyConfig(
            action_dim=self.action_dim,
            action_modality=self.action_modality,
            camera_layout=self.camera_layout,
            camera_keys=tuple(self.camera_keys),
            canvas_hw=tuple(self.canvas_hw),
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
            fps=self.fps,
            action_scale=self.action_scale,
            gripper_flip_dims=tuple(self.gripper_flip_dims),
            action_parameterization=self.action_parameterization,
            absolute_action_dims=tuple(self.absolute_action_dims),
            action_normalization=self.action_normalization,
            state_normalization=self.state_normalization,
            normalization_clip=self.normalization_clip,
            camera_dropout=dict(self.camera_dropout),
            trunk_weights=self.trunk_weights,
            video_vae_id=self.video_vae_id,
            text_encoder_id=self.text_encoder_id,
            dit_config=dict(self.dit_config),
            content_streams=self.content_streams,
            head_init_seed=self.head_init_seed,
            torch_dtype=self.torch_dtype,
            quantization=self.quantization,
            attn_mode=self.attn_mode,
            compile_model=self.compile_model,
            single_frame_encode=self.single_frame_encode,
            inference_profile=self.inference_profile,
            n_obs_steps=self.n_obs_steps,
            history_snapshots=self.history_snapshots,
            condition_on_past_actions=self.condition_on_past_actions,
            video_position_fps=self.video_position_fps,
            text_fixed_length=self.text_fixed_length,
            sampler=self.sampler,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            guidance_scale_action=self.guidance_scale_action,
            sampler_shift=self.sampler_shift,
            inference_seed=self.inference_seed,
            train_timestep_width=self.train_timestep_width,
            train_timestep_shift=self.train_timestep_shift,
            separate_timesteps=self.separate_timesteps,
            video_logit_mean=self.video_logit_mean,
            video_logit_std=self.video_logit_std,
            conditioning_noise_max=self.conditioning_noise_max,
            loss_reduction=self.loss_reduction,
            action_channel_weights=self.action_channel_weights,
            action_loss_weight=self.action_loss_weight,
            video_loss_weight=self.video_loss_weight,
            caption_dropout=self.caption_dropout,
            augment=self.augment,
            vae_batch_windows=self.vae_batch_windows,
            optimizer_lr=self.optimizer_lr,
            optimizer_lr_heads_multiplier=self.optimizer_lr_heads_multiplier,
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
    def camera_delta_indices(self) -> list[int]:
        """Frames per camera the loader must fetch: the observation window plus futures.

        F3A supervises predicted frames, so training needs the frames *after* the
        observation as video targets -- upstream's ``prepare`` refuses a window that is
        not exactly ``window_frames`` per camera, "observations plus future frames".
        That window is ``chunk_size`` future frames on top of the observation window
        (one frame, or ``n_obs_steps`` under the history profile), so the offsets run
        from ``-(n_obs - 1)`` through ``chunk_size`` inclusive and this list is
        ``window_frames`` long by construction -- pinned by the CPU suite.

        Note the cost: this multiplies camera decodes per sample by roughly
        ``chunk_size``, across every camera in ``camera_keys`` (three for the DROID
        layout). That is inherent to a joint video-action objective, not an accident of
        this wiring, but it is the reason no other policy here returns anything but
        ``None``.
        """
        n_obs = self.n_obs_steps if self.inference_profile == "history" else 1
        return list(range(-(n_obs - 1), self.chunk_size + 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
