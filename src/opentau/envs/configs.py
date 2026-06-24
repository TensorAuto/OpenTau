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

r"""This module contains configuration files for different environments. LIBERO and RoboCasa365 are supported."""

import abc
import logging
from copy import copy
from dataclasses import dataclass, field
from typing import Literal, get_args

import draccus

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.constants import ACTION, OBS_IMAGES, OBS_STATE
from opentau.utils.accelerate_utils import get_proc_accelerator

# Single source of truth for the ``control_mode`` enum: the type alias is
# used in the dataclass annotation, the runtime tuple is derived from it
# via ``get_args`` so the two can't drift if a new mode is ever added.
ControlMode = Literal["joint", "ee"]
CONTROL_MODE_CHOICES: tuple[str, ...] = get_args(ControlMode)
# Step size between consecutive ``speed`` bucket labels. Must match the
# spacing of
# :data:`opentau.datasets.speed_percentiles.SPEED_BUCKET_LABELS`
# (``0, 10, 20, ..., 100``); update both call sites together if the
# label set ever changes. Named ``_STEP`` (not ``_SECONDS``) because
# the bucket label is no longer a duration in seconds — it is a
# percentile-rank index.
SPEED_BUCKET_STEP = 10


@dataclass
class EnvMetadataConfig:
    """Optional pi07 metadata fields, broadcast across the rollout batch.

    These describe properties of the environment / robot / demonstration
    style that the pi07 prefix tokenizes as a ``"Metadata: ..."`` segment.
    They live on the env config (not the eval config) because they're
    properties of *what is being run*, not *how many episodes to run*.

    Each field defaults to ``None`` — the corresponding batch key is omitted
    and the policy's ``prepare_metadata`` pad path produces no segment in
    the prefix. Set a value to inject it for every env in the rollout.
    Allowed values mirror the training-time distribution emitted by
    :meth:`BaseDataset._emit_optional_keys`.

    Only the pi07 family of policies consumes these keys today; setting
    them when evaluating another policy (e.g. pi0, pi05) will pass
    validation but the values will be ignored downstream.

    Args:
        speed: Integer in ``[0, 100]`` and a multiple of
            ``SPEED_BUCKET_STEP`` (= 10), or ``None``. Matches the
            per-task percentile-rank bucket used at training time
            (``0`` = fastest decile, ``100`` = slowest); see
            :mod:`opentau.datasets.speed_percentiles`.
        quality: Integer in ``[1, 5]``, or ``None``.
        mistake: ``True`` / ``False``, or ``None``. Note that ``False`` is
            semantically distinct from ``None``: ``False`` emits a
            ``"Mistake: False"`` segment into the prefix, ``None`` omits
            the segment entirely.
        robot_type: Non-empty robot identifier string (e.g. ``"UR5"``), or
            ``None``.
        control_mode: ``"joint"`` (joint-position control) or ``"ee"``
            (end-effector control), or ``None``.
        emit_fps: Whether to broadcast :attr:`EnvConfig.fps` as the ``fps``
            metadata field at inference (paralleling
            :attr:`DatasetMixtureConfig.emit_fps` at training time).
            Defaults to ``False`` — fps conditioning is opt-in so old
            checkpoints resume cleanly (no surprise ``FPS:`` segment in
            the policy's metadata prefix). Flip to ``True`` for
            checkpoints trained with the training-side ``emit_fps=True``.
    """

    speed: int | None = None
    quality: int | None = None
    mistake: bool | None = None
    robot_type: str | None = None
    control_mode: ControlMode | None = None
    emit_fps: bool = False

    def __post_init__(self) -> None:
        # `isinstance(x, bool)` guards exclude Python bools — `bool` is a
        # subclass of `int`, so without them `speed=True` would silently
        # pass the type check and fail later with a confusing value error.
        if self.speed is not None:
            if not isinstance(self.speed, int) or isinstance(self.speed, bool):
                raise TypeError(f"env.metadata.speed must be int, got {type(self.speed).__name__}")
            if self.speed < 0 or self.speed > 100 or self.speed % SPEED_BUCKET_STEP != 0:
                raise ValueError(
                    f"env.metadata.speed must be a non-negative multiple of "
                    f"{SPEED_BUCKET_STEP} in [0, 100], got {self.speed}"
                )
        if self.quality is not None:
            if not isinstance(self.quality, int) or isinstance(self.quality, bool):
                raise TypeError(f"env.metadata.quality must be int, got {type(self.quality).__name__}")
            if not 1 <= self.quality <= 5:
                raise ValueError(f"env.metadata.quality must be in [1, 5], got {self.quality}")
        if self.mistake is not None and not isinstance(self.mistake, bool):
            raise TypeError(f"env.metadata.mistake must be bool, got {type(self.mistake).__name__}")
        if self.robot_type is not None:
            if not isinstance(self.robot_type, str):
                raise TypeError(f"env.metadata.robot_type must be str, got {type(self.robot_type).__name__}")
            if self.robot_type == "":
                raise ValueError(
                    "env.metadata.robot_type must be a non-empty string; use ``None`` to leave it missing."
                )
        if self.control_mode is not None and self.control_mode not in CONTROL_MODE_CHOICES:
            raise ValueError(
                f"env.metadata.control_mode must be one of {CONTROL_MODE_CHOICES} or None, "
                f"got {self.control_mode!r}"
            )


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration for an environment.

    Args:
        import_name: Name under which the environment should be imported. For LIBERO, this doesn't need to be set.
        make_id: Gymnasium/Gym environment id (e.g., ``"CartPole-v1"``) when using ``gym.make``-style construction.
        task: Optional task or suite identifier understood by the environment.
        fps: Target stepping frequency in Hz. Exact meaning is env-specific; for
            LIBERO it is the robosuite control frequency (``LiberoEnv`` overrides
            the default to 20).
        features: Mapping from logical feature names (e.g., ``"action"``,
            ``"pixels/agentview_image"``) to :class:`~opentau.configs.types.PolicyFeature`
            definitions consumed by policies.
        features_map: Mapping from environment keys to standardized OpenTau keys
            (e.g., mapping env observations into ``OBS_IMAGES`` / ``OBS_STATE``).
        max_parallel_tasks: Maximum number of tasks to run in parallel within the env.
        disable_env_checker: Whether to disable Gymnasium environment checking.
        metadata: Optional pi07 metadata fields (speed/quality/mistake/
            robot_type/control_mode) broadcast across the eval batch.
            Defaults to all-``None`` (no metadata injected).

    """

    import_name: str = None
    make_id: str = None
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True
    metadata: EnvMetadataConfig = field(default_factory=EnvMetadataConfig)

    @property
    def type(self) -> str:
        """Return the registered choice name for this config.

        Returns:
            The draccus choice name used in configs/CLI.
        """
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        """Keyword arguments used to construct the environment.

        Subclasses must implement this to return the kwargs consumed by the project’s
        environment builder (often ``gym.make`` or an equivalent factory).

        Returns:
            A dict of keyword arguments for environment construction.
        """
        raise NotImplementedError()


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    r"""Configuration for the LIBERO environment.

    Args:
        task: The LIBERO task or suite to use (e.g., ``"libero_10"``).
        task_ids: Optional list of specific task IDs within the suite to use (if ``None``, all tasks in the suite are used).
        fps: Robosuite control frequency (Hz) for the LIBERO sim — the rate at
            which each ``env.step`` advances the simulation. Threaded through to
            ``OffScreenRenderEnv(control_freq=...)``. Defaults to 20, robosuite's
            native LIBERO rate (the value used before this field was wired up).
        episode_length: Maximum length of each episode in steps.
        obs_type: Type of observations to use (e.g., ``"pixels_agent_pos"``).
        render_mode: Rendering mode for the environment (e.g., ``"rgb_array"``).
        camera_name: Comma-separated LIBERO raw camera names to render — both
            count and ordering of LIBERO cameras at eval are driven by this
            string. Defaults to ``"agentview_image,robot0_eye_in_hand_image"``
            (agentview + wrist eye-in-hand). Set to ``"agentview_image"``
            (single camera) for agentview-only rollouts. When the underlying
            policy was trained with a larger ``cfg.num_cams`` (e.g. a
            multi-domain mixture with 4 camera slots),
            ``preprocess_observation`` zero-fills the remaining ``cameraN``
            slots so the train↔eval input structure stays aligned —
            independent of how many real LIBERO cameras this field renders.
        init_states: Whether to initialize states randomly.
        camera_name_mapping: Optional mapping from camera names to standardized keys.
        subgoal_source: HuggingFace repo id of the v2.1 LeRobot dataset to source
            subgoal images from at eval time, or ``None`` to disable subgoal
            injection. When set (today only ``"TensorAuto/libero"`` is exercised),
            :func:`opentau.scripts.eval.eval` constructs a
            :class:`~opentau.envs.subgoal.LiberoLastFrameSubgoalGenerator` that
            samples a random matching episode per env at each ``rollout()`` call
            and serves its last frame as the subgoal — matching the pi07
            low-level / pi07-paligemma training-time ``subgoal{k}`` input. Behavior
            with repos other than ``"TensorAuto/libero"`` is undefined.
        features: Mapping from logical feature names to :class:`~opentau.configs.types.PolicyFeature` definitions.
        features_map: Mapping from environment keys to standardized OpenTau keys.
    """

    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 20  # robosuite control frequency (Hz); see field docstring above
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    subgoal_source: str | None = None
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,  # TODO: fix the name
            "pixels/agentview_image": f"{OBS_IMAGES}.image",  # TODO: fix the name
            "pixels/robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",  # TODO: fix the name
        }
    )

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(
                f"LIBERO env.fps (robosuite control frequency in Hz) must be positive, got {self.fps}"
            )
        if self.obs_type == "pixels":
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        r"""Return the keyword arguments used to construct the LIBERO environment."""
        suite_names = [s.strip() for s in str(self.task).split(",") if s.strip()]

        accelerator = get_proc_accelerator()

        task_ids: dict[str, list[int] | None]

        if accelerator is None:
            task_ids = {suite: copy(self.task_ids) for suite in suite_names}
            logging.info(f"[LIBERO environment] No accelerator found, using {task_ids=}.")
        else:
            from opentau.envs.libero import _get_suite

            task_ids = {
                suite: _get_suite(suite).tasks if self.task_ids is None else self.task_ids
                for suite in suite_names
            }
            logging.info(f"[LIBERO environment] Before distributing, using {task_ids=}.")
            task_ids = {
                suite: [
                    tsk
                    for idx, tsk in enumerate(tasks)
                    if idx % accelerator.num_processes == accelerator.process_index
                ]
                for suite, tasks in task_ids.items()
            }
            logging.info(
                f"[LIBERO environment] After distributing, using {task_ids=} on {accelerator.process_index=}."
            )

        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "task_ids": task_ids,
            "control_freq": self.fps,
        }


@EnvConfig.register_subclass("robocasa")
@dataclass
class RoboCasaEnv(EnvConfig):
    r"""Configuration for the RoboCasa365 kitchen environment.

    RoboCasa runs on robosuite 1.5 (shared with LIBERO since the libero extra was
    bumped to robosuite 1.5.2), so it co-installs in the same venv. The default
    robot is the PandaOmron mobile manipulator — hence the 12-D action and 16-D
    state, distinct from LIBERO's 7-D/8-D. Set ``metadata.robot_type`` /
    ``eval.control_mode`` to select the matching per-(robot_type, control_mode)
    projection head when evaluating a co-trained policy.

    Args:
        task: A RoboCasa task name (e.g. ``"CloseFridge"``), a comma-separated
            list of task names, or a benchmark-group shortcut
            (``atomic_seen``/``composite_seen``/``composite_unseen``/
            ``pretrain50``/``pretrain100``/``pretrain200``/``pretrain300``), which
            auto-expands to the upstream task list and auto-sets ``split``.
        fps: RoboCasa control frequency (Hz); also the ``render_fps`` for videos.
        episode_length: Maximum steps per episode (``_max_episode_steps``).
            Defaults to 1000; set to ``null`` (None) to use RoboCasa's official
            per-task horizon from the dataset registry (e.g. OpenCabinet=1050,
            TurnOnMicrowave=450) instead of a single global cap.
        obs_type: ``"pixels"`` or ``"pixels_agent_pos"``.
        render_mode: Rendering mode for the environment.
        camera_name: Comma-separated raw RoboCasa camera names to render. The
            wrapper remaps them to ``camera0``/``camera1``/... so the policy input
            structure matches LIBERO regardless of the raw names; when the policy
            was trained with a larger ``cfg.num_cams``, ``preprocess_observation``
            zero-fills the remaining slots.
        observation_height: Height of observation images.
        observation_width: Width of observation images.
        visualization_height: Height of visualization frames.
        visualization_width: Width of visualization frames.
        split: RoboCasa dataset split (``None``/``"all"``/``"pretrain"``/
            ``"target"``). Defaults to ``"pretrain"`` when left ``None`` — every
            task-group shortcut and concrete single-task config resolves to the
            pretrain kitchen-scene distribution; set explicitly to override.
        obj_registries: Object-mesh registries to sample assets from. Defaults to
            ``["lightwheel"]`` (the pack the asset downloader ships by default);
            add ``"objaverse"`` only after downloading that ~30GB pack.
        assets_root: Directory to store/read RoboCasa kitchen assets, kept outside the
            (ephemeral) uv venv. ``None`` resolves to the ``ROBOCASA_ASSETS_ROOT`` env
            var, else ``HF_OPENTAU_HOME/robocasa/assets``.
        auto_download_assets: If ``True`` (default), the asset packs ``obj_registries``
            needs are downloaded automatically (once) on first env build.
        features: Mapping from logical feature names to ``PolicyFeature`` definitions.
        features_map: Mapping from environment keys to standardized OpenTau keys.
    """

    task: str = "CloseFridge"
    fps: int = 20
    episode_length: int | None = 1000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "robot0_agentview_left,robot0_eye_in_hand,robot0_agentview_right"
    observation_height: int = 256
    observation_width: int = 256
    visualization_height: int = 512
    visualization_width: int = 512
    split: str | None = None
    obj_registries: list[str] = field(default_factory=lambda: ["lightwheel"])
    assets_root: str | None = None
    auto_download_assets: bool = True
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
        }
    )

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"RoboCasa env.fps (control frequency in Hz) must be positive, got {self.fps}")
        if self.obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        # The wrapper remaps the i-th raw camera to ``camera{i}``; mirror that in
        # the feature map using OpenTau's ``image`` / ``image2`` / ... convention
        # (camera0 -> image, camera1 -> image2, ...), matching LIBERO.
        cams = [c.strip() for c in self.camera_name.split(",") if c.strip()]
        for i, cam in enumerate(cams):
            self.features[f"pixels/{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )
            mapped = "image" if i == 0 else f"image{i + 1}"
            self.features_map[f"pixels/{cam}"] = f"{OBS_IMAGES}.{mapped}"

        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        r"""Return the keyword arguments used to construct the RoboCasa environment.

        Task resolution and per-rank sharding live in ``create_robocasa_envs`` (they
        need the ``robocasa`` package for group expansion), so this stays sim-free
        and only carries the obs/render parameters plus an optional ``split``.
        """
        kwargs: dict = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "visualization_height": self.visualization_height,
            "visualization_width": self.visualization_width,
        }
        if self.split is not None:
            kwargs["split"] = self.split
        return kwargs
