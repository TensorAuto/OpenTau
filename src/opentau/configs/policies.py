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
"""Policy configuration module.

This module provides the base PreTrainedConfig class for policy models, which
defines the interface and common functionality for all policy configurations.
It includes support for feature definitions, normalization modes, and loading
configurations from pretrained models or local paths.
"""

import abc
import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError
from transformers import PretrainedConfig as _HFPretrainedConfig

from opentau.configs.refs import resolve_refs
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.optim.optimizers import OptimizerConfig
from opentau.optim.schedulers import LRSchedulerConfig
from opentau.utils.hub import HubMixin

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


# --- transformers.PretrainedConfig <-> draccus codec ---
# pi07's policy configs declare dataclass fields of type
# `Gemma3WithExpertConfig` (a `transformers.PretrainedConfig` subclass) so the
# VLM topology travels with the policy config and is serialised in
# `config.json` via `_save_pretrained`. draccus's choice-type encode/decode
# path traverses dataclass fields and looks up an encoder by field type —
# without a registered handler for `PretrainedConfig`, encoding raises
# `Exception("No parser for object ...")` because `PretrainedConfig` is not
# itself a dataclass. Registered with `include_subclasses=True` so any
# subclass dispatches here automatically; the decoder receives the declared
# subclass as its first argument, which is what lets `cls.from_dict`
# reconstruct the right concrete type. Lives in this module (rather than
# `configs/default.py` next to the np.ndarray codec) so it's loaded as a
# side-effect of any policy-config import path that hits `_save_pretrained`,
# including standalone scripts that don't go through `configs/default.py`.
def _decode_hf_pretrained_config(cls, raw_value, path=()):
    """Decode a dict back into the declared ``PretrainedConfig`` subclass.

    Args:
        cls: The declared subclass of ``transformers.PretrainedConfig`` (e.g.
            ``Gemma3WithExpertConfig``). Provided by draccus because this
            decoder was registered with ``include_subclasses=True``.
        raw_value: The serialised dict produced by the matching encoder
            (``obj.to_dict()``).
        path: Ignored; kept for compatibility with draccus's decoder
            signature.

    Returns:
        An instance of ``cls`` reconstructed from ``raw_value``.
    """
    return cls.from_dict(raw_value)


draccus.encode.register(_HFPretrainedConfig, lambda obj: obj.to_dict(), include_subclasses=True)
draccus.decode.register(_HFPretrainedConfig, _decode_hf_pretrained_config, include_subclasses=True)

# Monotonic schema version for the *behavioral conventions* a checkpoint's
# weights were trained under (NOT the package version). Bump only when a change
# alters what the model sees/produces and must be gated so old checkpoints keep
# their trained behavior. History lives in CHANGELOG.md.
#   0  legacy: MIN_MAX/QUANTILE map a zero-range band (zero-padded tail dim, or a
#      genuinely-constant real dim) to -1.0.
#   1  a zero-range band maps to 0.0 (openpi's pad-after-normalize output), via
#      `Normalize(zero_range_center=True)`. See `zero_range_centers_on_zero`.
# A config that predates the field (`config_version is None`) is resolved to
# CURRENT for a fresh run and to 0 (legacy) when loading pre-fix weights — see
# `PreTrainedPolicy.from_pretrained`.
CURRENT_CONFIG_VERSION = 1

_DEPRECATED_LATENCY_FIELDS = (
    "cloud_vlm_latency_mean",
    "cloud_vlm_latency_std",
    "cloud_vlm_latency_lower",
    "cloud_vlm_latency_upper",
    "action_decoder_latency_mean",
    "action_decoder_latency_std",
    "action_decoder_latency_lower",
    "action_decoder_latency_upper",
)

# Fields that have been removed from the policy config dataclass. Stripped from
# both incoming JSON (so old saved configs load without erroring on unknown
# fields) and outgoing JSON (defensive).
_REMOVED_POLICY_FIELDS = ("init_strategy",)

_STRIPPED_FIELDS = _DEPRECATED_LATENCY_FIELDS + _REMOVED_POLICY_FIELDS


def _strip_keys(data: dict, keys: tuple[str, ...]) -> bool:
    """Drop ``keys`` from ``data`` (top-level and under a ``"policy"`` sub-dict).

    Returns True if anything changed.
    """
    changed = False
    for key in keys:
        if key in data:
            del data[key]
            changed = True
        if isinstance(data.get("policy"), dict) and key in data["policy"]:
            del data["policy"][key]
            changed = True
    return changed


def strip_deprecated_fields_from_json(path: Path) -> None:
    """Remove deprecated and removed fields from a config JSON file in-place.

    Only safe to call on files we own (e.g. the JSON we just wrote inside a
    save directory). Do NOT use on user-supplied inputs or HF cache files —
    HF cache paths are symlinks into a content-addressed blob store, so an
    in-place rewrite mutates the blob and silently corrupts the cache. For
    incoming configs, use :func:`load_stripped_config_to_tempfile` instead.
    """
    with open(path) as f:
        data = json.load(f)

    if _strip_keys(data, _STRIPPED_FIELDS):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


def load_resolved_config_dict(source_path: str | Path) -> dict:
    """Resolve ``$ref`` includes in a config JSON and return the resulting dict.

    Centralizes both the ref resolution and the top-level shape check so that
    every loader in this module fails consistently on a non-object root, and
    so that a single ``from_pretrained`` call can resolve once and pass the
    dict to all downstream helpers (warnings, stripping) instead of re-walking
    the ref tree from disk for each.
    """
    data = resolve_refs(source_path)
    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level config at {source_path} must be a JSON object after "
            f"$ref resolution, got {type(data).__name__}"
        )
    return data


def write_stripped_config_to_tempfile(data: dict) -> Path:
    """Strip deprecated/removed fields from ``data`` and write to a temp file.

    ``data`` is treated as read-only (a defensive copy is made before
    stripping). Caller owns the returned path and must unlink it.
    """
    # `_strip_keys` only ever deletes from the top-level dict and the "policy"
    # sub-dict, so a two-level shallow copy is enough to keep the caller's
    # dict untouched without paying for a deep copy of (potentially large)
    # nested configs.
    data = dict(data)
    if isinstance(data.get("policy"), dict):
        data["policy"] = dict(data["policy"])
    _strip_keys(data, _STRIPPED_FIELDS)

    fd, tmp_path = tempfile.mkstemp(prefix="opentau_config_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=4)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def load_stripped_config_to_tempfile(source_path: str | Path) -> Path:
    """Read a config JSON, resolve ``$ref`` includes, strip deprecated/removed
    fields in memory, write to a fresh temp file, and return its path. Does
    not mutate ``source_path``.

    Use this on incoming config files (HF Hub downloads, user-supplied paths)
    before handing them to ``draccus.parse``: HF cache paths are symlinks into
    a content-addressed blob store, so an in-place rewrite would corrupt the
    cache; user-supplied paths shouldn't be mutated as a side effect of load.

    See :mod:`opentau.configs.refs` for ``$ref`` semantics.
    """
    return write_stripped_config_to_tempfile(load_resolved_config_dict(source_path))


def _find_present_keys(data: dict, keys: tuple[str, ...]) -> list[str]:
    """Return ``keys`` that appear at the top level or under a ``"policy"`` sub-dict."""
    found: list[str] = []
    for key in keys:
        if key in data:
            found.append(key)
        if isinstance(data.get("policy"), dict) and key in data["policy"]:
            found.append(f"policy.{key}")
    return found


def warn_deprecated_latency_fields_from_dict(data: dict, config_path: str | Path) -> None:
    """Like :func:`warn_deprecated_latency_fields` but operating on an
    already-resolved config dict, so callers that already loaded the dict
    don't pay for a second ``$ref`` walk. ``config_path`` is used only for
    the warning message."""
    found = _find_present_keys(data, _DEPRECATED_LATENCY_FIELDS)
    if found:
        warnings.warn(
            f"Config '{config_path}' contains deprecated latency fields that are no longer "
            f"used and will be ignored: {', '.join(found)}. "
            "Consider re-saving the config to remove them.",
            DeprecationWarning,
            stacklevel=4,
        )


def warn_removed_policy_fields_from_dict(data: dict, config_path: str | Path) -> None:
    """Like :func:`warn_removed_policy_fields` but operating on an
    already-resolved config dict. ``config_path`` is used only for the
    warning message."""
    found = _find_present_keys(data, _REMOVED_POLICY_FIELDS)
    if found:
        warnings.warn(
            f"Config '{config_path}' contains fields that have been removed and will be "
            f"ignored: {', '.join(found)}. Re-save the config to drop them.",
            DeprecationWarning,
            stacklevel=4,
        )


def warn_deprecated_latency_fields(config_path: str | Path) -> None:
    """Emit a deprecation warning if a config JSON file contains latency fields.

    Checks both top-level fields and fields nested under a ``"policy"`` key.
    Should be called before loading a config so users are aware the fields
    will be ignored. Resolves ``$ref`` includes so fields hidden behind a
    reference are still detected.
    """
    warn_deprecated_latency_fields_from_dict(load_resolved_config_dict(config_path), config_path)


def warn_removed_policy_fields(config_path: str | Path) -> None:
    """Emit a deprecation warning if a config JSON file contains removed fields.

    Removed fields (e.g. ``init_strategy``) are stripped at load time so old
    saved configs continue to parse, but the user's explicit choice is dropped
    on the floor — surface that with a warning so they know to re-save and
    update any downstream tooling that still emits the old key. Resolves
    ``$ref`` includes so fields hidden behind a reference are still detected.
    """
    warn_removed_policy_fields_from_dict(load_resolved_config_dict(config_path), config_path)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
        skip_normalization_weights: When loading via :py:meth:`~opentau.policies.pretrained.PreTrainedPolicy.from_pretrained`,
            drop the saved ``normalize_*`` / ``unnormalize_*`` buffer tensors from the state dict
            before ``load_state_dict``. The buffers freshly initialised from ``dataset_stats``
            then survive — use this when finetuning a checkpoint whose saved normalization stats
            were aggregated over a different dataset mixture than the finetuning data. The
            buffers are registered as ``nn.Parameter(requires_grad=False)``, so training alone
            cannot recover from inheriting the wrong stats. **Requires ``dataset_stats`` to be
            supplied to ``__init__``** (e.g. via :py:func:`opentau.policies.factory.make_policy`);
            otherwise the buffers stay at the ``inf`` sentinel from
            :py:func:`opentau.policies.normalize.create_stats_buffers` and the next forward
            crashes. **One-shot (in-memory only):** after the strip helper runs (whether keys
            were dropped or the "had no effect" warning fired), the flag is reset to ``False``
            on the *in-memory* model's config — a user who opted in once should not have to
            remember to flip it back before resume. Persistence requires
            ``save_pretrained``: an in-process resume that goes
            ``train.py`` → ``save_checkpoint`` → ``cfg.save_pretrained(...)`` writes ``False``
            to the new checkpoint's ``config.json``, so the next ``from_pretrained`` reads
            ``False`` and skips the strip. But re-running ``from_pretrained`` on the *original*
            source checkpoint (e.g. an interactive notebook session that hasn't yet saved) will
            re-read ``True`` from the source ``config.json`` and re-strip. The
            model was trained against the saved stats, so switching the normalization mid-training
            only makes sense when followed by further training, not when loading purely for
            inference. Honored by every policy whose ``from_pretrained`` (or ``_load_as_safetensor``)
            calls :py:meth:`~opentau.policies.pretrained.PreTrainedPolicy._strip_normalization_buffers_from_state_dict`.
            Defaults to ``False`` (no behaviour change).
        skip_input_resolution_check: Escape hatch for
            :py:meth:`validate_input_resolution`. When ``True``, a mismatch
            between the policy's ``resize_imgs_with_padding`` and the ``(H, W)``
            of the bound image features logs a loud warning instead of raising —
            even on the strict (training) path. Intended for deliberately
            resuming/finetuning a legacy checkpoint that was trained with the
            mismatch (i.e. with the policy silently letterboxing a second time
            inside ``prepare_images``/``prepare_videos``); leave ``False`` for
            everything else. Eval-shaped policy construction never raises
            regardless of this flag (it warns), so plain evaluation of legacy
            checkpoints does not need it. Note that ``True`` persists into the
            ``config.json`` of every checkpoint the run saves — coherent for
            resuming that same mismatch-trained lineage, but set it back to
            ``false`` when fine-tuning such a checkpoint against a *new*
            dataset mixture, or a fresh mismatch there will only warn instead
            of raising. Defaults to ``False``.
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # When True, training `torch.compile`s the policy's heavy compute submodule
    # (the flow-matching `self.model` for pi05 / pi07) *in place* via
    # `torch.nn.Module.compile`, leaving the host-side preprocessing in the
    # policy wrapper's own forward uncompiled. The compile is applied in
    # `PreTrainedPolicy.maybe_compile_for_training`, called from train.py after
    # the bf16 cast and before `accelerator.prepare`. Compatible with DDP /
    # single-process / DeepSpeed ZeRO-1/2; train.py rejects ZeRO-3 and FSDP
    # (both re-shard parameters during forward, which invalidates the traced
    # graphs). Defaults to False (no behaviour change; the default eager path
    # is byte-for-byte unchanged).
    use_torch_compile: bool = False
    # `torch.compile` mode forwarded to `torch.nn.Module.compile(mode=...)` when
    # `use_torch_compile` is True. One of "default", "reduce-overhead",
    # "max-autotune", "max-autotune-no-cudagraphs". Stick with "default" under
    # DeepSpeed — the cudagraph-based modes ("reduce-overhead", "max-autotune")
    # conflict with DeepSpeed's activation/parameter memory reuse.
    torch_compile_mode: str = "default"

    pretrained_path: str | None = None
    skip_normalization_weights: bool = False
    skip_input_resolution_check: bool = False

    # When False, `_save_pretrained` strips normalize_*.buffer_* / unnormalize_*.buffer_*
    # keys from the state_dict before writing model.safetensors. Reloading then requires
    # the caller to pass `ds_meta=` (or `stats=`) to `make_policy` so the buffers can be
    # repopulated; otherwise the inf-init assertion fires at first forward.
    save_normalization_stats: bool = True

    # Ordered list of norm-head identifiers this policy was trained on — one
    # per row of the stacked Normalize/Unnormalize buffers. With
    # `dataset_to_norm_index` set (new checkpoints), each entry is a norm key
    # ("<robot_type>::<control_mode>" or a fallback dataset name). Legacy
    # checkpoints (`dataset_to_norm_index is None`) carry one entry per
    # training dataset and the entries are dataset names; the per-dataset and
    # per-norm-head axes are then the same. `None` only for policies
    # constructed outside the standard `make_policy(ds_meta=...)` path (e.g.
    # legacy single-stats fallbacks).
    dataset_names: list[str] | None = None

    # `{training_dataset_name: norm_head_row}` map persisted at training
    # time. Lets inference resolve a `batch["dataset_repo_id"]` (a
    # training-time dataset name) to the right row when datasets share a
    # `(robot_type, control_mode)` head. `None` on legacy checkpoints; the
    # policy then falls back to treating `dataset_names` as a 1:1
    # per-dataset list (identity mapping) for inference-time lookup.
    dataset_to_norm_index: dict[str, int] | None = None

    # Schema version of the behavioral conventions this checkpoint's weights were
    # trained under (see `CURRENT_CONFIG_VERSION`). `None` is the "unresolved"
    # sentinel — a config that predates the field. It is resolved to a concrete
    # int in one of two directions depending on provenance, and NEVER in
    # `__post_init__` (which would erase the "was it absent?" signal the loader
    # needs to tell a fresh run from a pre-fix checkpoint):
    #   - loading pre-fix weights whose source carries no tag -> 0 (legacy), in
    #     `PreTrainedPolicy.from_pretrained`;
    #   - a fresh run / scratch policy -> CURRENT, in `factory.make_policy` and
    #     defensively at save time.
    # `None` behaves as CURRENT for normalization (`zero_range_centers_on_zero`),
    # so a config carrying `null` still runs new-behavior; the concretization
    # only matters so a *saved* checkpoint self-describes for later fine-tunes.
    config_version: int | None = None

    # Informational only — the `opentau` package version that last wrote this
    # config. Never read by any branch (it is "unknown" for a non-installed
    # source checkout, so it is a debugging hint, not a fact). Excluded from any
    # exact-config-equality assertion.
    opentau_version: str | None = None

    # Deprecated: latency fields are no longer used. Kept for backward-compatible
    # loading of old JSON configs. Must remain 0.0; non-zero values will raise.
    cloud_vlm_latency_mean: float = 0.0
    cloud_vlm_latency_std: float = 0.0
    cloud_vlm_latency_lower: float = 0.0
    cloud_vlm_latency_upper: float = 0.0
    action_decoder_latency_mean: float = 0.0
    action_decoder_latency_std: float = 0.0
    action_decoder_latency_lower: float = 0.0
    action_decoder_latency_upper: float = 0.0

    def __post_init__(self):
        """Initialize post-creation attributes.

        This method can be overridden by subclasses to perform additional
        initialization after the dataclass is created.
        """
        for field_name in _DEPRECATED_LATENCY_FIELDS:
            value = getattr(self, field_name)
            if value != 0.0:
                raise ValueError(
                    f"Deprecated config field '{field_name}' must be 0.0, got {value}. "
                    "Non-zero latency config fields are no longer supported."
                )

    def resolved_config_version(self) -> int:
        """Concrete config version, treating the unresolved ``None`` as CURRENT.

        Behavior (not persistence) reads through this: an unresolved config runs
        new-behavior. The stored ``config_version`` stays ``None`` until a save or
        an explicit provenance resolution concretizes it.
        """
        return self.config_version if self.config_version is not None else CURRENT_CONFIG_VERSION

    def zero_range_centers_on_zero(self) -> bool:
        """Whether ``Normalize``/``Unnormalize`` should map a zero-range MIN_MAX /
        QUANTILE band to ``0.0`` (config_version >= 1) rather than the legacy
        ``-1.0`` (config_version 0). Threaded into every ``Normalize`` /
        ``Unnormalize`` this policy builds via the ``zero_range_center`` kwarg.
        """
        return self.resolved_config_version() >= 1

    @property
    def type(self) -> str:
        """Get the type name of this configuration.

        Returns:
            The choice name of this configuration class.
        """
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        """Get indices for observation delta features.

        Returns:
            List of indices indicating which observation features should be
            treated as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        """Get indices for action delta features.

        Returns:
            List of indices indicating which action features should be treated
            as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        """Get indices for reward delta features.

        Returns:
            List of indices indicating which reward features should be treated
            as deltas, or None if no delta features are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        """Get the default optimizer configuration for this policy.

        Returns:
            An OptimizerConfig instance with default settings for this policy type.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Get the default learning rate scheduler configuration for this policy.

        Returns:
            An LRSchedulerConfig instance with default settings for this policy type,
            or None if no scheduler should be used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        """Validate that the feature configuration is correct.

        This method should check that all required features are present and
        have valid configurations.

        Raises:
            ValueError: If the feature configuration is invalid.
        """
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Get the robot state feature from input features.

        Returns:
            The PolicyFeature with type STATE if found, or None otherwise.
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """Get the environment state feature from input features.

        Returns:
            The PolicyFeature with type ENV if found, or None otherwise.
        """
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Get all visual/image features from input features.

        Returns:
            Dictionary mapping feature names to PolicyFeature instances with
            type VISUAL.
        """
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    def _bound_image_resolutions(self) -> dict[str, tuple[int, int]]:
        """``(H, W)`` per bound image feature.

        Skips the synthetic ``observation.images.empty_camera_*`` placeholders:
        their hard-coded shape describes no real data — each policy's
        ``prepare_images``/``prepare_videos`` fabricates those slots in-policy
        as masked placeholder frames cloned at the shape of the processed real
        camera tensors. Handles both the policy-side channels-first ``(C, H,
        W)`` convention (mixture-bound features) and the raw dataset
        channels-last ``(H, W, C)`` convention (bare
        ``LeRobotDatasetMetadata``); shapes that are recognizably neither are
        skipped rather than misread.
        """
        resolutions: dict[str, tuple[int, int]] = {}
        for key, ft in self.image_features.items():
            if key.startswith("observation.images.empty_camera"):
                continue
            shape = tuple(ft.shape)
            if len(shape) != 3:
                continue
            if shape[0] in (1, 3, 4) and shape[2] not in (1, 3, 4):
                resolutions[key] = (shape[1], shape[2])
            elif shape[2] in (1, 3, 4) and shape[0] not in (1, 3, 4):
                resolutions[key] = (shape[0], shape[1])
            elif shape[0] in (1, 3, 4) and shape[2] in (1, 3, 4):
                # Ambiguous (e.g. (3, H, 3)); prefer channels-first, the
                # policy-side convention.
                resolutions[key] = (shape[1], shape[2])
        return resolutions

    @property
    def input_image_size(self) -> tuple[int, int] | None:
        """``(H, W)`` the vision tower will actually receive.

        This is ``resize_imgs_with_padding`` when the policy defines and sets
        it (the in-policy letterbox target), else the resolution of the bound
        image features (native pass-through), else ``None`` when neither is
        known (e.g. a bare config before feature binding).
        """
        resize_target = getattr(self, "resize_imgs_with_padding", None)
        if resize_target is not None:
            return tuple(resize_target)
        for resolution in self._bound_image_resolutions().values():
            return resolution
        return None

    def validate_input_resolution(self, *, strict: bool) -> None:
        """Check ``resize_imgs_with_padding`` against the bound image features.

        The image features bound onto the config (from
        ``TrainPipelineConfig.resolution`` via the dataset mixture at train
        time, or carried inside a checkpoint's ``config.json`` at load time)
        are the ``(H, W)`` the policy actually receives. A differing
        ``resize_imgs_with_padding`` means every frame gets silently
        letterboxed a second time inside the policy — downscaled and padded
        with black bars — which is almost never intended.

        With ``resize_imgs_with_padding=None`` (native pass-through) there is
        no target to compare against, but the bound cameras must then agree
        with *each other* — with no in-policy resize step, a mixed-resolution
        camera set has no single geometry the vision tower could be built
        for, so that is flagged with the same strict/warn semantics.

        No-op when the policy has no ``resize_imgs_with_padding`` field or no
        comparable image feature is bound.

        Args:
            strict: ``True`` for training-shaped construction — raise on
                mismatch (unless ``skip_input_resolution_check``); ``False``
                for eval/inference — warn loudly but keep loading, so legacy
                checkpoints trained with the mismatch remain evaluable.

        Raises:
            ValueError: On mismatch when ``strict`` is ``True`` and
                ``skip_input_resolution_check`` is ``False``.
        """
        if not hasattr(self, "resize_imgs_with_padding"):
            return
        resize_target = self.resize_imgs_with_padding
        resolutions = self._bound_image_resolutions()
        if resize_target is None:
            if len(set(resolutions.values())) <= 1:
                return
            described = ", ".join(f"{key}={res}" for key, res in sorted(resolutions.items()))
            message = (
                f"resize_imgs_with_padding is null (native pass-through) but the bound image "
                f"features have mixed resolutions: {described}. With no in-policy resize step "
                "every camera must arrive at one resolution; set resize_imgs_with_padding to a "
                "single (H, W) target instead."
            )
        else:
            resize_target = tuple(resize_target)
            mismatched = {key: res for key, res in resolutions.items() if res != resize_target}
            if not mismatched:
                return
            described = ", ".join(f"{key}={res}" for key, res in sorted(mismatched.items()))
            message = (
                f"resize_imgs_with_padding={resize_target} (H, W) does not match the resolution of "
                f"the bound image features: {described}. The policy would silently letterbox every "
                "frame a second time (aspect-preserving resample + black padding), degrading the "
                "vision input. Set policy.resize_imgs_with_padding to the input resolution (and "
                "TrainPipelineConfig.resolution to the resolution you want to train at), or set it "
                "to null to pass frames through at the bound resolution (PaliGemma-family policies "
                "only; the Gemma3-family pi06/pi07 require the vision tower's square image_size)."
            )
        if strict and not self.skip_input_resolution_check:
            raise ValueError(
                message + " If you are deliberately resuming a legacy checkpoint trained with this "
                "mismatch, set policy.skip_input_resolution_check=true to downgrade this to a warning."
            )
        logging.warning(
            message + " Proceeding because this is %s.",
            "skip_input_resolution_check=true" if strict else "an eval/inference load",
        )

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Get the action feature from output features.

        Returns:
            The PolicyFeature with type ACTION if found, or None otherwise.
        """
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the configuration to a directory.

        Args:
            save_directory: Directory path where the configuration will be saved.
        """
        config_path = save_directory / CONFIG_NAME
        # Encode with `declared_type=PreTrainedConfig` so draccus inserts the
        # `type` choice-type discriminator at the top of the dict. Without
        # the declared type, `draccus.dump(self, ...)` infers it from the
        # runtime class (`PI06Config`, etc.) — already concrete, so the
        # discriminator is omitted — and the resulting `config.json` is
        # unloadable by `from_pretrained`, which dispatches off the parent
        # `PreTrainedConfig` and requires `type` to pick a subclass. Going
        # through `encode` + `json.dump` is equivalent to draccus's own JSON
        # dump path; the difference is the `declared_type` arg, which feeds
        # draccus's `encode_choice` code path (see draccus.parsers.encoding).
        data = draccus.encode(self, declared_type=PreTrainedConfig)
        # Concretize the version so the saved checkpoint self-describes for later
        # fine-tunes (a peeked `null` would be read as "no tag" -> legacy). A
        # config that reached save without provenance resolution is a fresh run,
        # hence CURRENT. Also stamp the informational package version.
        data["config_version"] = self.resolved_config_version()
        from opentau.__version__ import __version__ as _opentau_version

        # Stamp the current (last) writer, per the field's contract. Falls back to
        # any prior value only when this checkout can't report its own version
        # ("unknown" for a non-installed source tree), so the hint isn't lost.
        data["opentau_version"] = (
            _opentau_version if _opentau_version != "unknown" else (self.opentau_version or _opentau_version)
        )
        with open(config_path, "w") as f:
            json.dump(data, f, indent=4)
        strip_deprecated_fields_from_json(config_path)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """Load a policy configuration from a pretrained model or local path.

        Args:
            cls: The class to instantiate.
            pretrained_name_or_path: Can be either:

                - A string, the model id of a pretrained config hosted inside a model
                  repo on huggingface.co.
                - A path to a directory containing a configuration file saved using
                  the `_save_pretrained` method.
            force_download: Whether to force (re-)downloading the config files and
                configuration from the HuggingFace Hub. Defaults to False.
            resume_download: Whether to resume downloading the config files.
                Defaults to None.
            proxies: Dictionary of proxies to use for requests. Defaults to None.
            token: The token to use as HTTP bearer authorization. If True, will use
                the token generated when running `huggingface-cli login`. Defaults to None.
            cache_dir: Path to a directory in which a downloaded pretrained model
                configuration should be cached. Defaults to None.
            local_files_only: Whether to only look at local files (i.e., do not try
                to download the config). Defaults to False.
            revision: The specific model version to use. It can be a branch name, a
                tag name, or a commit id. Defaults to None.
            **policy_kwargs: Additional keyword arguments. May include 'cli_overrides'
                for command-line argument overrides.

        Returns:
            An instance of the configuration class loaded from the specified path.

        Raises:
            FileNotFoundError: If the configuration file is not found on the
                HuggingFace Hub or in the local path.
        """
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = policy_kwargs.pop("cli_overrides", [])

        if config_file is None:
            return draccus.parse(cls, config_file, args=cli_overrides)

        # Resolve $refs once and reuse — the warn helpers and the strip step
        # would otherwise each walk the full ref tree from disk.
        config_data = load_resolved_config_dict(config_file)
        warn_deprecated_latency_fields_from_dict(config_data, config_file)
        warn_removed_policy_fields_from_dict(config_data, config_file)
        # Strip deprecated/removed keys via a temp file rather than mutating the
        # source — `config_file` may be an HF cache symlink to a content-addressed
        # blob, where a rewrite would silently corrupt the cache.
        tmp_config = write_stripped_config_to_tempfile(config_data)
        try:
            return draccus.parse(cls, str(tmp_config), args=cli_overrides)
        finally:
            tmp_config.unlink(missing_ok=True)
