#!/usr/bin/env python

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

"""Factory functions for creating datasets and dataset mixtures.

This module provides factory functions to create individual datasets and
weighted dataset mixtures from configuration objects. It handles the setup
of delta timestamps, image transforms, and metadata configuration before
instantiating datasets.

The factory supports two types of datasets:
    1. LeRobot datasets: Standard robot learning datasets loaded from HuggingFace
       repositories with configurable delta timestamps for temporal alignment.
    2. VQA datasets: Vision-language vqa datasets (CLEVR, COCO-QA,
       VSR, etc.) for multimodal learning tasks.

Key Features:
    - Delta timestamp resolution: Automatically configures temporal offsets
      for features.
    - Image transform support: Applies configurable image transformations
      during dataset creation.
    - Imagenet stats override: Optionally replaces dataset statistics with
      ImageNet normalization statistics for camera features.
    - VQA dataset registration: Supports extensible vqa dataset
      registration through side-effect imports.

Functions:
    make_dataset: Creates a single dataset instance from a DatasetConfig,
        handling delta timestamp setup, image transforms, and metadata
        configuration.
    make_dataset_mixture: Creates a WeightedDatasetMixture from a
        TrainPipelineConfig containing multiple dataset configurations.
    resolve_delta_timestamps: Resolves delta timestamps configuration based
        on TrainPipelineConfig settings, mapping features to temporal groups.
    val_split_generator: Builds the rank-independent RNG that partitions a
        dataset into its train and validation halves.
    warn_if_resumed_split_differs: Warns when a resumed run's config would
        partition the data differently than the checkpoint it resumes from.

Constants:
    IMAGENET_STATS: ImageNet normalization statistics (mean, std, min, max)
        used for camera feature normalization when use_imagenet_stats is enabled.

Example:
    Create a single dataset:
        >>> dataset = make_dataset(dataset_cfg, train_cfg, return_advantage_input=False)

    Create a dataset mixture:
        >>> mixture = make_dataset_mixture(train_cfg, return_advantage_input=False)
        >>> dataloader = mixture.get_dataloader()
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

# NOTE: Don't delete; imported for side effects.
import opentau.datasets.vqa.clevr  # noqa: F401
import opentau.datasets.vqa.cocoqa  # noqa: F401
import opentau.datasets.vqa.dummy  # noqa: F401
import opentau.datasets.vqa.vsr  # noqa: F401
from opentau import available_vqa_datasets
from opentau.configs.default import DatasetConfig
from opentau.configs.policies import TRAIN_CONFIG_NAME
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.action_indexing import resolve_delta_map
from opentau.datasets.dataset_mixture import WeightedDatasetMixture
from opentau.datasets.delta_action_stats import (
    delta_stats_cache_key,
    load_or_compute_delta_action_stats,
)
from opentau.datasets.lerobot_dataset import (
    BaseDataset,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    suppress_control_mode_warning,
)
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, feature_mapping_key
from opentau.datasets.transforms import ImageTransforms
from opentau.datasets.utils import DeltaTimestampInfo
from opentau.utils.accelerate_utils import get_proc_accelerator

logger = logging.getLogger(__name__)

IMAGENET_STATS = {
    "min": [[[0.0]], [[0.0]], [[0.0]]],  # (c,1,1)
    "max": [[[1.0]], [[1.0]], [[1.0]]],  # (c,1,1)
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _apply_column_index_and_delta_config(
    dataset: BaseDataset, dataset_cfg: DatasetConfig, train_cfg: TrainPipelineConfig
) -> None:
    """Copy the per-dataset column-index / delta-action settings onto ``dataset``.

    Translates ``DatasetConfig.delta_action_state_map`` from **parquet** index space into the
    **post-index** space the dataset's emitted vectors use, so ``_apply_column_index_and_delta``
    can address the reindexed columns directly. See
    :mod:`opentau.datasets.action_indexing` for the two spaces.

    Args:
        dataset: The freshly-constructed dataset to configure.
        dataset_cfg: Its per-dataset config entry.
        train_cfg: Pipeline config, forwarded to the delta-stats computation.
    """
    if (
        dataset_cfg.state_index is None
        and dataset_cfg.action_index is None
        and not dataset_cfg.use_delta_joint_actions
    ):
        return
    dataset.state_index = dataset_cfg.state_index
    dataset.action_index = dataset_cfg.action_index
    if not dataset_cfg.use_delta_joint_actions:
        return

    # Width of the emitted action vector: the index length when subsetting, else the dataset's
    # own raw action dim. Only used to warn about kept-but-unmapped dims.
    action_dim = len(dataset_cfg.action_index) if dataset_cfg.action_index is not None else None
    if action_dim is None:
        feature = getattr(dataset, "meta", None)
        shapes = getattr(feature, "features", None) or {}
        name_map = dataset._get_name_map(strict=False)
        raw_key = name_map.get("actions", "action")
        raw_shape = (shapes.get(raw_key) or {}).get("shape")
        if raw_shape:
            action_dim = int(raw_shape[-1])

    dataset.delta_action_state_map = resolve_delta_map(
        dataset_cfg.delta_action_state_map,
        dataset_cfg.action_index,
        dataset_cfg.state_index,
        who=str(dataset_cfg.repo_id),
        action_dim=action_dim,
    )
    logging.info(
        "%s: delta joint actions ON. parquet map %s -> post-index map %s (state_index=%s, action_index=%s).",
        dataset_cfg.repo_id,
        dataset_cfg.delta_action_state_map,
        dataset.delta_action_state_map,
        dataset_cfg.state_index,
        dataset_cfg.action_index,
    )
    dataset.delta_action_stats = _compute_or_load_delta_stats(dataset, dataset_cfg, train_cfg)


def _compute_or_load_delta_stats(
    dataset: BaseDataset, dataset_cfg: DatasetConfig, train_cfg: TrainPipelineConfig
) -> dict[str, dict[str, np.ndarray]]:
    """Get this dataset's delta-action stats, from cache or by computing them.

    The dataset's ``meta/stats.json`` describes *absolute*, per-frame actions, so once the delta
    transform is on those numbers no longer describe the training targets and must be replaced.
    See :mod:`opentau.datasets.delta_action_stats`.

    Args:
        dataset: The configured dataset (already carrying its resolved delta map).
        dataset_cfg: Its config entry.
        train_cfg: Pipeline config, read for ``num_workers`` and
            ``dataset_mixture.delta_stats_max_rows``.

    Returns:
        ``{"actions": {...}, "state": {...}}`` in post-index space.
    """
    meta = dataset.meta
    episodes = list(dataset.episodes) if dataset.episodes is not None else list(meta.episodes)
    # One task per distinct data file: v3.0 consolidates many episodes into one parquet, so
    # mapping episode -> path without dedup would re-read (and double-count) the same file.
    parquet_paths = list(dict.fromkeys(str(meta.root / meta.get_data_file_path(ep)) for ep in episodes))

    # The action horizon in frame-index space. `delta_timestamps_params[0]` holds the *mean*
    # offsets; `get_delta_indices_soft` adds N(0, std) jitter per sample, so with a non-zero std
    # these stats describe the mean horizon rather than any single draw — an approximation
    # documented on the module.
    dt_mean = dataset.delta_timestamps_params[0]
    name_map = dataset._get_name_map()
    # `resolve_delta_timestamps` keys its output by RAW on-disk column names (it iterates
    # `ds_meta.features`), so the action horizon must be looked up under this dataset's own
    # action column — `"action"` for a standard LeRobot repo — not the `"actions"` alias.
    chunk_offsets = np.asarray(dt_mean[name_map["actions"]], dtype=np.float64) * dataset.fps

    # Bounds the O(frames x chunk_size) pass on very large sources. In the cache key too, so
    # flipping the cap recomputes instead of serving stats from a different sampling budget.
    max_rows = getattr(train_cfg.dataset_mixture, "delta_stats_max_rows", None)
    cache_key = delta_stats_cache_key(
        state_index=dataset_cfg.state_index,
        action_index=dataset_cfg.action_index,
        delta_map=dataset.delta_action_state_map,
        chunk_offsets=chunk_offsets,
        vector_resample_strategy=dataset.vector_resample_strategy,
        episodes=episodes,
        excluded_episodes=dataset_cfg.excluded_episodes,
        fps=dataset.fps,
        revision=dataset_cfg.revision,
        max_rows=max_rows,
    )
    # Only the main process computes, so sizing the pool from the full `num_workers` neither
    # oversubscribes the node nor leaves it idle.
    max_workers = max(1, int(getattr(train_cfg, "num_workers", 1) or 1))
    return load_or_compute_delta_action_stats(
        root=Path(meta.root),
        cache_key=cache_key,
        compute_kwargs={
            "parquet_paths": parquet_paths,
            "state_col": name_map["state"],
            "action_col": name_map["actions"],
            "state_index": dataset_cfg.state_index,
            "action_index": dataset_cfg.action_index,
            "delta_map": dataset.delta_action_state_map,
            "chunk_offsets": chunk_offsets,
            "strategy": dataset.vector_resample_strategy,
            "episodes": set(episodes),
            "max_workers": max_workers,
            "max_rows": max_rows,
        },
    )


def _apply_metadata_overrides(dataset: BaseDataset, dataset_cfg: DatasetConfig) -> None:
    """Apply ``robot_type`` / ``control_mode`` overrides from a DatasetConfig.

    The overrides are written through to ``dataset.meta.info`` so downstream
    consumers (``meta.control_mode``, ``_emit_optional_keys``) observe the
    overridden value. ``None`` means "do not override"; any string value
    (including ``""``) is applied.
    """
    if dataset_cfg.robot_type is not None:
        dataset.meta.info["robot_type"] = dataset_cfg.robot_type
    if dataset_cfg.control_mode is not None:
        dataset.meta.info["control_mode"] = dataset_cfg.control_mode
        # `LeRobotDataset.__init__` caches `self.control_mode = self.meta.control_mode`
        # before this override fires, so refresh the attribute when present.
        if hasattr(dataset, "control_mode"):
            dataset.control_mode = dataset_cfg.control_mode


def resolve_delta_timestamps(
    cfg: TrainPipelineConfig, dataset_cfg: DatasetConfig, ds_meta: LeRobotDatasetMetadata
) -> DeltaTimestampInfo:
    """Resolves per-feature delta_timestamps based on TrainPipelineConfig.

    Args:
        cfg (TrainPipelineConfig): The TrainPipelineConfig to read delta_indices from.
        dataset_cfg (DatasetConfig): The dataset configuration.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        A 4-tuple ``(mean, std, lower, upper)`` of dicts mapping feature names
        to lists of delta-timestamp values.  Keys that appear only in ``mean``
        will be filled with sensible defaults by
        ``LeRobotDataset.compute_delta_params``.
    """
    delta_timestamps: dict[str, list[float]] = {}
    action_freq = cfg.dataset_mixture.action_freq
    # Mixed-frequency training: `action_freq=None` opts out of resampling.
    # Substituting `ds_meta.fps` makes every delta-timestamp land exactly on
    # this dataset's native frame boundaries, so nearest-neighbor sampling is
    # a no-op and consecutive frames are returned unchanged.
    if action_freq is None:
        action_freq = ds_meta.fps

    if dataset_cfg.repo_id is None:
        raise ValueError("dataset_cfg.repo_id must not be None when resolving delta timestamps.")
    if dataset_cfg.data_features_name_mapping is not None:
        # This entry's own mapping — the registry may hold another entry's
        # mapping when two entries share a repo_id and control_mode (see
        # BaseDataset._get_name_map for the fetch-time counterpart).
        name_map = dataset_cfg.data_features_name_mapping
    else:
        # Runs before `_apply_metadata_overrides`, so prefer the config's control_mode
        # override, falling back to the on-disk value, to resolve dual-split columns.
        control_mode = (
            dataset_cfg.control_mode if dataset_cfg.control_mode is not None else ds_meta.control_mode
        )
        mkey = feature_mapping_key(dataset_cfg.repo_id, control_mode)
        name_map = DATA_FEATURES_NAME_MAPPING[
            mkey if mkey in DATA_FEATURES_NAME_MAPPING else dataset_cfg.repo_id
        ]
    reverse_name_map = {v: k for k, v in name_map.items()}
    for key in ds_meta.features:
        if key not in reverse_name_map:
            continue  # only process camera, state, and action features

        # Trajectory-sequence emission for recurrent policies. The window is
        # anchored at its *last* timestep, so timestep t sits at frame offset
        # `-(T - 1 - t) * stride`. Two reasons: it reuses the existing
        # history-window convention for observations verbatim (all offsets <= 0),
        # and it matches inference, where the memory is built from the past and
        # the policy predicts *now*.
        #
        # Episode-boundary clamping and per-entry padding flags come free from
        # the fetch layer — see `lerobot_dataset.py`, which notes that "per-frame
        # temporal padding info (from clamped episode boundaries) is tracked by
        # obs_history_is_pad", with `<key>_is_pad` doing the same for actions. So
        # a window running off the start of an episode is clamped and reported,
        # and no boundary arithmetic is needed here.
        # `getattr` because tests build the mixture as a SimpleNamespace stand-in
        # that only carries the fields the case under test needs.
        seq_len = getattr(cfg.dataset_mixture, "sequence_length", 1)
        seq_stride = getattr(cfg.dataset_mixture, "sequence_stride", None)
        if seq_stride is None:
            # RoboTTT semantics: one timestep IS one action chunk, so consecutive
            # timesteps tile the trajectory in disjoint chunks. Any other stride
            # overlaps adjacent timesteps' action targets and hands the mostly
            # teacher-forced context the current chunk's answers to copy —
            # `TrainPipelineConfig.validate` rejects explicit mismatches.
            # Only resolved in sequence mode: at seq_len == 1 the stride is never
            # read downstream, and test stubs legitimately omit `action_chunk`.
            seq_stride = cfg.action_chunk if seq_len > 1 else 1

        # `sequence_stride` is documented in *frames*, but every offset here is
        # converted with `action_freq` — the resampling rate — so the two agree
        # only when `action_freq == ds_meta.fps`. Oversampling makes consecutive
        # timesteps land inside one source frame, and the nearest-frame fetch
        # returns the *same* observation for both: the memory sees duplicates
        # and TTT has nothing to carry. Caught on droid_100, which is 15 fps
        # while the dev config asks for 30, giving bit-identical frames at
        # stride 1.
        if seq_len > 1 and action_freq > seq_stride * ds_meta.fps + 1e-6:
            raise ValueError(
                f"sequence_length={seq_len} with action_freq={action_freq} Hz on a dataset "
                f"recorded at {ds_meta.fps} Hz: a stride of {seq_stride} frame(s) is "
                f"{seq_stride / action_freq:.4f}s, shorter than one source frame "
                f"({1 / ds_meta.fps:.4f}s), so consecutive timesteps would resolve to the same "
                "observation. Set dataset_mixture.action_freq to the dataset's fps (the stride "
                "itself is derived from action_chunk and is not a tuning knob)."
            )

        standard_key = reverse_name_map[key]
        if (
            standard_key == "actions"
            and cfg.policy is not None
            and cfg.policy.action_delta_indices is not None
        ):
            chunk_offsets = list(cfg.policy.action_delta_indices)
            # T * H offsets, timestep-major so the reshape to (T, H, ...) is a
            # plain view and the row order stays batch-major downstream.
            # At seq_len == 1 this is exactly `chunk_offsets`, byte for byte.
            delta_timestamps[key] = [
                (-(seq_len - 1 - t) * seq_stride + h) / action_freq
                for t in range(seq_len)
                for h in chunk_offsets
            ]
        elif "camera" in standard_key or standard_key == "state":
            n_obs = cfg.dataset_mixture.n_obs_history
            if seq_len > 1:
                # One observation per supervised timestep. Config validation
                # already refuses combining this with `n_obs_history`.
                delta_timestamps[key] = [
                    -(seq_len - 1 - t) * seq_stride / action_freq for t in range(seq_len)
                ]
            elif n_obs is not None:
                interval = getattr(cfg.policy, "history_interval", 1)
                delta_timestamps[key] = [-(n_obs - 1 - i) * interval / action_freq for i in range(n_obs)]
            else:
                delta_timestamps[key] = [0.0]

    dt_mean = {k: np.array(v) for k, v in delta_timestamps.items()}
    return dt_mean, {}, {}, {}


# Fallback base seed for the train/val split when `cfg.seed is None`. The split
# must be identical on every rank, so it cannot fall back to the global RNG:
# without `cfg.seed`, `train.py` never calls `set_seed` at all and each process
# starts from its own entropy-derived torch seed. An arbitrary but fixed
# constant keeps the split reproducible in that case too.
DEFAULT_VAL_SPLIT_SEED = 8_675_309


def val_split_generator(train_cfg: TrainPipelineConfig) -> torch.Generator:
    """Build the rank-independent RNG that partitions a dataset into train/val.

    The train/val split **must** be identical on every rank. Each rank builds
    its own copy of the dataset and its own ``Subset`` views, so if the ranks
    disagree on the partition, one rank's validation frames are another rank's
    training frames — the gathered ``Validation/*`` metrics are then measured
    partly on memorized data and are optimistically biased, and the
    ``running_best`` checkpoint selection is driven by that contaminated number.

    Taking the split off the *global* torch RNG does exactly that, because
    :func:`opentau.utils.random_utils.set_seed` deliberately offsets the seed by
    ``process_index * 12345`` so that per-sample draws (augmentation, dropout,
    prompt substitution) are decorrelated across ranks. That offset is correct
    for per-sample draws and is intentionally left alone; the split is simply
    moved off the global RNG onto this dedicated, rank-independent generator.

    Stability contract — the split is a pure function of exactly three inputs:

    * ``train_cfg.seed`` (never ``process_index``, never ``num_processes``),
    * ``len(dataset)``,
    * the effective ``val_split_ratio`` for that dataset.

    Consequences that are deliberate:

    * **World-size change** (resuming on 4 GPUs instead of 8, or vice versa)
      leaves the split untouched, so validation numbers stay comparable across
      the resume boundary and across runs with different world sizes.
    * **Resume** reproduces the same split as long as those three inputs are
      unchanged. ``--config_path`` points at the checkpoint's ``train_config.json``
      on resume, so they carry over automatically — but a CLI override of
      ``seed``, of a ``val_split_ratio``, or of the fields that decide how many
      frames a dataset yields (``episodes``, ``excluded_episodes``, ``revision``)
      silently re-partitions the data. :func:`warn_if_resumed_split_differs`
      catches all of those. The one case it cannot catch is an out-of-band
      length change — the same repo at the same revision re-uploaded with more
      episodes — since the checkpoint records no length to compare against.
    * **Mixture composition is irrelevant**: every dataset draws from a
      generator seeded the same way, so adding or reordering datasets does not
      reshuffle the splits of the others. Two datasets of equal length get the
      same permutation, which is harmless (frame *i* of one dataset has nothing
      to do with frame *i* of another) and is in fact what you want when the
      same underlying data is listed twice under different configs — the shared
      permutation keeps the val frames aligned instead of leaking each entry's
      val half into the other's train half.

    Args:
        train_cfg: The training config supplying the base ``seed``.

    Returns:
        A fresh ``torch.Generator`` seeded identically on every rank.
    """
    seed = train_cfg.seed if train_cfg.seed is not None else DEFAULT_VAL_SPLIT_SEED
    return torch.Generator().manual_seed(int(seed))


def _split_determining_fields(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract the values the train/val split depends on from a *serialized* config.

    Takes an encoded config dict rather than a ``TrainPipelineConfig`` so that
    the live run and the checkpoint's ``train_config.json`` are read by the same
    code. Two separate extractors would drift, and a key present on only one
    side is silently dropped from the comparison — i.e. it fails open, which for
    a diagnostic is indistinguishable from a healthy resume.

    Covers the config half of all three inputs in the split contract (see
    :func:`val_split_generator`):

    * ``seed`` directly;
    * the **effective** ``val_split_ratio`` per dataset, resolving the
      ``None``-inherits-the-mixture-default rule the way ``make_dataset`` does,
      so a ratio moving between ``None`` and an explicit value equal to the
      default is correctly seen as no change;
    * the fields that determine ``len(dataset)`` without changing the dataset's
      identity — ``episodes`` / ``excluded_episodes`` select and drop episodes,
      and ``revision`` pins which version of the repo is read. ``repo_id`` comes
      along so that the positional per-dataset comparison is only made between
      like and like.

    Args:
        cfg_dict: A ``TrainPipelineConfig`` encoded via ``to_dict()``, or a
            ``train_config.json`` parsed from a checkpoint.

    Returns:
        A flat ``{field path: value}`` mapping, ready to diff against another
        config's mapping.
    """
    mixture = cfg_dict.get("dataset_mixture") or {}
    default_ratio = mixture.get("val_split_ratio")

    fields: dict[str, Any] = {"seed": cfg_dict.get("seed")}
    for i, ds in enumerate(mixture.get("datasets") or []):
        ratio = ds.get("val_split_ratio")
        # The mixture-wide default is folded in here rather than compared on its
        # own: changing it is a no-op for any dataset that overrides it.
        fields[f"datasets[{i}].val_split_ratio"] = default_ratio if ratio is None else ratio
        for key in ("repo_id", "revision", "episodes", "excluded_episodes"):
            fields[f"datasets[{i}].{key}"] = ds.get(key)
    return fields


def _describe_drift(field: str, before: Any, after: Any) -> str:
    """Render one drifted field for the warning, keeping episode lists readable."""

    def fmt(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) > 4:
            return f"[{len(value)} items]"
        return repr(value)

    return f"{field}: {fmt(before)} -> {fmt(after)}"


def warn_if_resumed_split_differs(cfg: TrainPipelineConfig) -> None:
    """Warn when a resumed run would partition its data differently than before.

    On resume the whole train config is loaded from the checkpoint's
    ``train_config.json`` and CLI overrides are layered on top, so a stray
    ``--seed=...`` or ``--dataset_mixture.val_split_ratio=...`` silently
    re-partitions every dataset. The resumed run then reports validation on
    frames the checkpoint was trained on, and its ``running_best`` numbers are
    not comparable to the pre-resume ones.

    Checks the *config* half of the contract, which includes the config-visible
    determinants of ``len(dataset)`` — ``episodes``, ``excluded_episodes`` and
    ``revision`` are all serialized into the checkpoint, so overriding them on
    resume is caught. What remains undetectable is an **out-of-band** length
    change: the same ``repo_id`` at the same ``revision`` re-uploaded with more
    episodes. Nothing in the checkpoint records the length it observed, so that
    case is documented on :func:`val_split_generator` rather than warned about.

    No-ops when not resuming, when validation is off, or when the checkpoint's
    ``train_config.json`` is missing or unreadable — a best-effort diagnostic
    must never be the thing that fails a resume. Every rank runs this (the
    dataset is built on all of them), so the message is emitted on the main
    process only rather than once per GPU.

    Args:
        cfg: The resolved training config for the run being started.
    """
    if not cfg.resume or cfg.val_freq <= 0 or cfg.checkpoint_path is None:
        return

    # None outside a training run (tests, offline scripts) — warn in that case.
    accelerator = get_proc_accelerator()
    if accelerator is not None and not accelerator.is_main_process:
        return

    saved_path = Path(cfg.checkpoint_path) / TRAIN_CONFIG_NAME
    try:
        with open(saved_path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read %s for the val-split drift check: %s", saved_path, exc)
        return

    saved_fields = _split_determining_fields(saved)
    current_fields = _split_determining_fields(cfg.to_dict())
    # Intersect on keys: a mixture that gained or lost a dataset changes far more
    # than the split, and comparing shifted positions would only add noise.
    drifted = [
        _describe_drift(key, saved_fields[key], value)
        for key, value in current_fields.items()
        if key in saved_fields and saved_fields[key] != value
    ]
    if drifted:
        logger.warning(
            "Resuming with train/val split settings that differ from the checkpoint at %s: %s. "
            "The split is a pure function of (seed, dataset length, val_split_ratio), so the "
            "resumed run validates on a different subset than the checkpoint was trained "
            "against — its Validation/* metrics and `running_best` are not comparable to the "
            "pre-resume ones, and frames held out before are now trained on.",
            cfg.checkpoint_path,
            ", ".join(sorted(drifted)),
        )


def _subset_meta(meta: LeRobotDatasetMetadata) -> LeRobotDatasetMetadata:
    """Per-subset metadata copy that shares read-only per-episode structures.

    Returns a shallow copy of ``meta`` that shares ``episodes`` /
    ``episodes_stats`` (and every other attribute) by reference, deep-copying
    only the small aggregated ``stats`` dict so the train/val halves can hold
    independent normalization stats without cross-contamination.

    ``random_split`` divides a dataset by frame index, so the per-episode
    ``episodes`` / ``episodes_stats`` are identical across the two subsets and
    are read-only on the training path (only the dataset-*creation*
    ``save_episode`` / stats-conversion paths write them). A blanket
    ``deepcopy(meta)`` cloned them per subset: for a large per-episode-stats
    dataset listed under many mixture configs that is tens of GB of needless
    copies per rank, and — worse — copy-on-write surface that blows up host RAM
    once forked across dataloader workers (every worker touches the clones).
    Sharing them by reference removes both costs. Mirrors the share-by-reference
    contract of :meth:`BaseDataset.shallow_copy_with_dropout`.

    Args:
        meta: The source dataset metadata to copy for a train/val subset.
    """
    subset_meta = copy.copy(meta)
    subset_meta.stats = copy.deepcopy(meta.stats)
    return subset_meta


def make_dataset(
    cfg: DatasetConfig,
    train_cfg: TrainPipelineConfig,
    return_advantage_input: bool = False,
) -> Union[BaseDataset, Tuple[BaseDataset, BaseDataset]]:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    A train and validation dataset are returned if `train_cfg.val_freq` is greater than 0.
    The validation dataset is a subset of the train dataset, and is used for evaluation during training.
    The validation dataset is created by splitting the train dataset into train and validation sets based on the
    effective split ratio: the per-dataset `cfg.val_split_ratio` when set, otherwise the mixture-wide
    `train_cfg.dataset_mixture.val_split_ratio` (the per-dataset value `None` inherits the mixture default).

    The split is drawn from a dedicated, rank-independent generator rather than the global torch RNG
    (which `set_seed` offsets per process), so every rank partitions the dataset identically and the
    validation half is held out from *all* ranks' training data. See `val_split_generator`.

    Args:
        cfg (DatasetConfig): A DatasetConfig used to create a LeRobotDataset.
        train_cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        return_advantage_input (bool): Whether the created dataset includes advantage inputs including "success",
            "episode_end_idx", "current_idx", "last_step", "episode_index", and "timestamp". Defaults to False.

    Raises:
        ValueError: If exactly one of `cfg.vqa` and `cfg.repo_id` is not provided.
        ValueError: If `cfg.vqa` is not a supported vqa dataset.

    Returns:
        BaseDataset or Tuple[BaseDataset, BaseDataset]: A single dataset or a tuple of (train_dataset, val_dataset) if val_freq > 0.
    """
    image_transforms = ImageTransforms(cfg.image_transforms) if cfg.image_transforms.enable else None

    if isinstance(cfg.vqa, str) + isinstance(cfg.repo_id, str) != 1:
        raise ValueError("Exactly one of `cfg.vqa` and `cfg.repo_id` should be provided.")

    if isinstance(cfg.vqa, str):
        ds_cls = available_vqa_datasets.get(cfg.vqa)
        if ds_cls is None:
            raise ValueError(
                f"Unknown vqa dataset '{cfg.vqa}'. Supported datasets are: {available_vqa_datasets.keys()}"
            )
        # TODO support dataset-specific arg / kwargs
        dataset = ds_cls(train_cfg)
    elif isinstance(cfg.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        dt_mean, dt_std, dt_lower, dt_upper = resolve_delta_timestamps(train_cfg, cfg, ds_meta)
        # Suppress the "missing control_mode" warning when the user is
        # providing an explicit override — they already know it's missing.
        # Ordering invariant: this MUST run before `LeRobotDataset(...)` below;
        # once `__init__` emits the warning the suppression is a no-op.
        if cfg.control_mode is not None:
            suppress_control_mode_warning(cfg.repo_id)
        # Per-dataset values win over the mixture-wide default; `None` means
        # "inherit". See `DatasetConfig` / `DatasetMixtureConfig` docstrings.
        effective_tolerance = (
            cfg.tolerance_s if cfg.tolerance_s is not None else train_cfg.dataset_mixture.tolerance_s
        )
        effective_skip = (
            cfg.skip_timestamp_check
            if cfg.skip_timestamp_check is not None
            else train_cfg.dataset_mixture.skip_timestamp_check
        )
        dataset = LeRobotDataset(
            train_cfg,
            cfg.repo_id,
            root=cfg.root,
            episodes=cfg.episodes,
            excluded_episodes=cfg.excluded_episodes,
            delta_timestamps=dt_mean,
            delta_timestamps_std=dt_std,
            delta_timestamps_lower=dt_lower,
            delta_timestamps_upper=dt_upper,
            tolerance_s=effective_tolerance,
            image_transforms=image_transforms,
            revision=cfg.revision,
            video_backend=cfg.video_backend,
            image_resample_strategy=train_cfg.dataset_mixture.image_resample_strategy,
            vector_resample_strategy=train_cfg.dataset_mixture.vector_resample_strategy,
            return_advantage_input=return_advantage_input,
            skip_timestamp_check=effective_skip,
            prompt_substitutions=cfg.prompt_substitutions,
            data_features_name_mapping=cfg.data_features_name_mapping,
        )
    else:
        raise ValueError("Exactly one of `cfg.vqa` and `cfg.repo_id` should be provided.")

    _apply_column_index_and_delta_config(dataset, cfg, train_cfg)
    _apply_metadata_overrides(dataset, cfg)

    # TODO vqa datasets implement stats in original feature names, but camera_keys are standardized names
    if (
        not isinstance(cfg.vqa, str)
        and isinstance(cfg.repo_id, str)
        and "dummy" not in cfg.repo_id
        and cfg.use_imagenet_stats
    ):
        if dataset.meta.stats is None:
            dataset.meta.stats = {}
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                if key not in dataset.meta.stats:
                    dataset.meta.stats[key] = {}
                dataset.meta.stats[key][stats_type] = np.array(stats, dtype=np.float32)

    if train_cfg.val_freq > 0:
        # Per-dataset value wins over the mixture-wide default; `None` means
        # "inherit". Mirrors the `tolerance_s` / `skip_timestamp_check`
        # resolution above. See `DatasetConfig` / `DatasetMixtureConfig` docs.
        effective_val_split = (
            cfg.val_split_ratio
            if cfg.val_split_ratio is not None
            else train_cfg.dataset_mixture.val_split_ratio
        )
        val_size = int(len(dataset) * effective_val_split)
        train_size = len(dataset) - val_size
        # Explicit, rank-independent generator: the global torch RNG is seeded
        # per-rank on purpose (`set_seed(..., accelerator=...)` offsets by
        # `process_index * 12345`), so splitting off it gives every rank a
        # different partition and the reported validation metrics end up
        # measured on other ranks' training frames. See `val_split_generator`
        # for the full stability contract.
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=val_split_generator(train_cfg)
        )
        # Share the large, read-only per-episode metadata (`episodes`,
        # `episodes_stats`) across the two halves; deep-copy only the small
        # aggregated `stats`. A blanket `deepcopy(meta)` here is tens of GB of
        # needless per-rank copies for a large per-episode-stats dataset and
        # becomes copy-on-write surface that OOMs the host across dataloader
        # workers. See `_subset_meta`.
        train_dataset.meta = _subset_meta(dataset.meta)  # type: ignore[assignment]
        val_dataset.meta = _subset_meta(dataset.meta)  # type: ignore[assignment]

        # Subset wraps the same underlying dataset by reference, so the
        # training and validation halves would share every instance attribute
        # — including the optional-key dropout and prompt-substitution flags.
        # Give the val subset its own shallow copy whose only divergent
        # attributes are those toggles. See
        # ``BaseDataset.shallow_copy_with_dropout`` for the contract on what
        # stays shared.
        val_dataset.dataset = dataset.shallow_copy_with_dropout(  # type: ignore[attr-defined]
            enable_dropout=train_cfg.dataset_mixture.val_enable_optional_key_dropout,
            enable_prompt_substitution=train_cfg.dataset_mixture.val_enable_prompt_substitution,
        )
        return train_dataset, val_dataset  # type: ignore[return-value]

    return dataset


def _resolve_weights(
    configured_weights: Optional[List[float]], datasets: list, label: str = "datasets"
) -> List[float]:
    """Return explicit weights or infer them from dataset lengths.

    Args:
        configured_weights: User-provided weights, or None to infer.
        datasets: The list of datasets whose lengths are used when
            ``configured_weights`` is None.
        label: Human-readable label used in the log message
            (e.g. "train" or "val").

    Returns:
        A list of float weights, one per dataset.
    """
    if configured_weights is not None:
        return configured_weights
    weights = [float(len(ds)) for ds in datasets]
    logger.info("No explicit weights provided; inferring %s weights from dataset lengths: %s", label, weights)
    return weights


def _validate_metadata_requirements(cfg: TrainPipelineConfig, datasets: list, label: str) -> None:
    """Raise if the mixture requires non-empty robot_type / control_mode and
    any dataset (after overrides) still has an empty value.
    """
    require_robot = cfg.dataset_mixture.require_non_empty_robot_type
    require_control = cfg.dataset_mixture.require_non_empty_control_mode
    if not (require_robot or require_control):
        return

    dataset_cfgs = cfg.dataset_mixture.datasets
    # Invariant from `make_dataset_mixture`: each dataset_cfg appends exactly
    # one entry to `datasets` (and at most one to `val_datasets`). Assert
    # rather than silently skipping so a future refactor that breaks the
    # invariant doesn't quietly bypass the require_non_empty_* checks.
    assert len(dataset_cfgs) == len(datasets), (
        f"dataset_cfgs ({len(dataset_cfgs)}) and {label} datasets ({len(datasets)}) "
        "must be 1:1; cannot validate metadata requirements."
    )

    bad: list[str] = []
    for dc, ds in zip(dataset_cfgs, datasets, strict=True):
        info = ds.meta.info
        identifier = dc.repo_id or dc.vqa or type(ds).__name__
        if require_robot and not (info.get("robot_type") or ""):
            bad.append(f"{identifier}: robot_type is empty")
        if require_control and not (info.get("control_mode") or ""):
            bad.append(f"{identifier}: control_mode is empty")

    if bad:
        raise ValueError(
            "DatasetMixtureConfig requires non-empty metadata fields, but the "
            f"following {label} datasets are missing values after overrides:\n  - "
            + "\n  - ".join(bad)
            + "\nSet `DatasetConfig.robot_type` / `DatasetConfig.control_mode` "
            "on the offending dataset(s) to provide an override."
        )


def _maybe_pair(dataset, dataset_cfg: DatasetConfig, cfg: TrainPipelineConfig):
    """Wraps one subset in the paired loader when ``pair_episodes`` is set.

    One dataset entry is one pairing key: the loader draws two distinct
    episodes from *this* subset and concatenates them, so a pair can never
    cross keys and the demonstration always describes the same variant as the
    rollout.

    Wrapping happens here rather than inside ``WeightedDatasetMixture`` so the
    mixture's weighting and hierarchical sampling keep operating on whole
    subsets, unchanged.

    Args:
        dataset: The constructed subset.
        dataset_cfg: Its config, supplying the ambiguous prompt.
        cfg: The pipeline config.

    Returns:
        The dataset, wrapped when pairing is enabled.

    Raises:
        ValueError: If pairing is on but the subset resolves to fewer than two
            episodes to pair. A missing ``ambiguous_prompt`` only warns: it is
            required for within-task ambiguity but wrong for cross-task
            transfer, and this cannot tell the two apart.
    """
    if not cfg.dataset_mixture.pair_episodes:
        return dataset

    from opentau.datasets.paired_sequence import PairedSequenceDataset

    key = dataset_cfg.repo_id or "subset"
    if dataset_cfg.episodes:
        key = f"{key}#{len(dataset_cfg.episodes)}eps"
    if not dataset_cfg.ambiguous_prompt:
        # Two legitimate designs reach here, and only one is a mistake.
        #
        # WITHIN-TASK ambiguity (e.g. NavigateKitchen): every episode of the key
        # shares a skill and differs only in target, so the episode's own prompt
        # names that target and the demonstration becomes redundant. An
        # ambiguous prompt is mandatory there.
        #
        # CROSS-TASK transfer (e.g. robolab bowl -> mug): the prompt supplies
        # WHAT and the demonstration supplies HOW for an object the model has
        # not manipulated. Overwriting the prompt would destroy the task
        # identity the model needs. Leaving it is correct.
        #
        # The loader cannot tell these apart, so warn rather than refuse and
        # let the experiment's author own the choice.
        logging.warning(
            "pair_episodes is on for %s with no `ambiguous_prompt`, so the sample keeps the "
            "rollout half's own instruction (a pair carries one prompt; the demonstration's "
            "is dropped). This is correct ONLY for cross-task transfer, "
            "where the prompt names the task and the demonstration shows how. If the "
            "episodes of this key differ only in their target, the prompt names that "
            "target and the demonstration is redundant — set `ambiguous_prompt`.",
            key,
        )
    episodes = list(getattr(dataset, "episodes", None) or dataset_cfg.episodes or [])
    if len(episodes) < 2:
        raise ValueError(
            f"pair_episodes is on but {key} resolves to {len(episodes)} episode(s); "
            "pairing needs at least two."
        )
    return PairedSequenceDataset(
        base=dataset,
        pairing_keys={key: episodes},
        # `or None` keeps the two layers agreeing on what "no override" means:
        # the falsy check above already treated `""` as absent and warned, but
        # `__getitem__` gates on `is not None` and would have blanked both
        # halves' prompt instead of leaving the rollout's in place.
        # None -> PairedSequenceDataset leaves the sample's own prompt in place.
        prompts={key: dataset_cfg.ambiguous_prompt or None},
        samples_per_epoch=len(dataset),
        seed=cfg.seed or 0,
    )


def make_dataset_mixture(
    cfg: TrainPipelineConfig, return_advantage_input: bool = False
) -> Union[WeightedDatasetMixture, Tuple[WeightedDatasetMixture, WeightedDatasetMixture]]:
    """Creates a dataset mixture from the provided TrainPipelineConfig.

    Args:
        cfg (TrainPipelineConfig): The configuration containing the datasets to mix.
            If `cfg.dataset_mixture.weights` is None, each dataset is weighted
            by its length (cast to float).
        return_advantage_input (bool): Whether the datasets should return advantage inputs including "success",
            "episode_end_idx", "current_idx", "last_step", "episode_index", and "timestamp". Defaults to False.

    Returns:
        WeightedDatasetMixture or Tuple[WeightedDatasetMixture, WeightedDatasetMixture]: An instance of WeightedDatasetMixture containing the datasets, or a tuple of (train_mixture, val_mixture) if val_freq > 0.
    """
    warn_if_resumed_split_differs(cfg)

    datasets = []
    val_datasets = []
    for dataset_cfg in cfg.dataset_mixture.datasets:
        res = make_dataset(dataset_cfg, cfg, return_advantage_input=return_advantage_input)
        if isinstance(res, tuple):
            datasets.append(_maybe_pair(res[0], dataset_cfg, cfg))
            val_datasets.append(_maybe_pair(res[1], dataset_cfg, cfg))
        else:
            datasets.append(_maybe_pair(res, dataset_cfg, cfg))

    _validate_metadata_requirements(cfg, datasets, label="train")
    if val_datasets:
        _validate_metadata_requirements(cfg, val_datasets, label="val")

    train_weights = _resolve_weights(cfg.dataset_mixture.weights, datasets, label="train")
    train_mixture = WeightedDatasetMixture(cfg, datasets, train_weights, cfg.dataset_mixture.action_freq)

    if val_datasets:
        val_weights = _resolve_weights(cfg.dataset_mixture.weights, val_datasets, label="val")
        val_mixture = WeightedDatasetMixture(cfg, val_datasets, val_weights, cfg.dataset_mixture.action_freq)
        return train_mixture, val_mixture

    return train_mixture
