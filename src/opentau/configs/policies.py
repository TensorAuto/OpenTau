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
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mps
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False
    pretrained_path: str | None = None
    skip_normalization_weights: bool = False

    # When False, `_save_pretrained` strips normalize_*.buffer_* / unnormalize_*.buffer_*
    # keys from the state_dict before writing model.safetensors. Reloading then requires
    # the caller to pass `ds_meta=` (or `stats=`) to `make_policy` so the buffers can be
    # repopulated; otherwise the inf-init assertion fires at first forward.
    save_normalization_stats: bool = True

    # Ordered list of dataset names this policy was trained on. Used by the per-sample
    # Normalize/Unnormalize indexing path to map an inference-time
    # `batch["dataset_repo_id"]` (str) into the leading dim of the stacked stats
    # buffers. `None` only for policies constructed outside the standard
    # `make_policy(ds_meta=...)` path (e.g. legacy single-stats fallbacks).
    dataset_names: list[str] | None = None

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
