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

"""Weighted dataset mixture for combining multiple datasets with controlled sampling.

This module provides functionality to combine multiple PyTorch datasets into a
single weighted mixture, enabling training on heterogeneous datasets with
controlled sampling proportions. It supports hierarchical sampling strategies
that efficiently handle large-scale dataset combinations while maintaining
memory efficiency.

The module implements a two-level sampling approach:
    1. Dataset-level sampling: Selects which dataset to sample from based on
       specified weights.
    2. Sample-level sampling: Uniformly samples within the selected dataset.

This hierarchical approach avoids expensive multinomial sampling over millions
of individual samples by operating at the dataset level, making it scalable
for large dataset mixtures.

Key Features:
    - Weighted sampling: Control relative sampling frequency of different
      datasets through configurable weights.
    - Memory-efficient sampling: Hierarchical sampler processes samples in
      chunks to minimize memory overhead.
    - Metadata aggregation: Automatically aggregates and standardizes metadata
      from multiple datasets, including statistics normalization and feature
      name mapping.
    - Format standardization: Converts dataset-specific feature formats to a
      common standard format, handling vector padding and missing cameras.

Classes:
    WeightedDatasetMixture: Main class for combining multiple datasets with
        weighted sampling. Creates concatenated datasets and provides DataLoader
        with hierarchical sampling.
    HierarchicalSampler: Custom PyTorch sampler that implements two-level
        weighted sampling (dataset selection, then uniform sample selection).
    DatasetMixtureMetadata: Aggregates metadata from multiple datasets,
        standardizes feature names, pads vectors, and combines statistics.

Functions:
    pad_vector: Pads the last dimension of a vector to a target size with zeros.

Example:
    Create a dataset mixture with two datasets resampled to a shared 30 Hz:
        >>> datasets = [dataset1, dataset2]
        >>> weights = [0.7, 0.3]  # 70% from dataset1, 30% from dataset2
        >>> mixture = WeightedDatasetMixture(cfg, datasets, weights, action_freq=30.0)
        >>> dataloader = mixture.get_dataloader()

    Mixed-frequency mixture (no resampling) — each dataset is sampled at its
    own native fps, so a single batch can contain samples drawn at different
    rates:
        >>> mixture = WeightedDatasetMixture(cfg, datasets, weights, action_freq=None)
"""

import functools
import logging
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.compute_stats import aggregate_stats
from opentau.datasets.lerobot_dataset import BaseDataset, DatasetMetadata
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, feature_mapping_key


class _TaggedDataset(Dataset):
    """Wraps a ``BaseDataset`` so every sample carries its norm-head identity.

    The wrapped sample dict gains two keys:

      - ``"dataset_repo_id"``: ``str`` — the deduplicated mixture-level name
        (matches an entry in ``DatasetMixtureMetadata.dataset_names``).
        Default PyTorch collate batches a per-sample ``str`` into a
        ``list[str]`` of length B.
      - ``"dataset_index"``: ``torch.long`` scalar — the policy's
        **norm-head row** for this dataset (i.e.
        ``DatasetMixtureMetadata.dataset_to_norm_index[dataset_repo_id]``).
        Default collate stacks into a ``(B,)`` long tensor. Note: the field
        name is retained for back-compat, but the value indexes the
        norm-head axis, not the per-dataset enumerate axis — two datasets
        sharing a ``(robot_type, control_mode)`` get the same value here.

    Policies read either key via ``PreTrainedPolicy._resolve_dataset_index``
    and route per-sample into the stacked Normalize/Unnormalize buffers.
    The training path always uses ``dataset_index`` (immune to the
    optional-key dropout that ``robot_type`` / ``control_mode`` undergo);
    inference can additionally supply ``dataset_repo_id`` or
    ``(robot_type, control_mode)``.

    The wrapper exposes the underlying dataset's ``.meta`` so callers of
    ``WeightedDatasetMixture`` (e.g. metadata validation, per-dataset val
    loaders) keep working unchanged. ``len`` delegates as well.
    """

    def __init__(self, base: Dataset, dataset_repo_id: str, dataset_index: int):
        self._base = base
        self._dataset_repo_id = dataset_repo_id
        # Pre-build the scalar long tensor once so __getitem__ doesn't
        # allocate per call.
        self._dataset_index_tensor = torch.tensor(int(dataset_index), dtype=torch.long)
        # Preserve `.meta` so `WeightedDatasetMixture.__init__`'s validation
        # (`hasattr(ds, "meta") and ds.meta is not None`) still works after
        # wrapping. `Subset` / `random_split` produce wrappers without a
        # `.meta`, so look one level deeper if needed.
        meta = getattr(base, "meta", None)
        if meta is None and hasattr(base, "dataset"):
            meta = getattr(base.dataset, "meta", None)
        self.meta = meta

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        item["dataset_repo_id"] = self._dataset_repo_id
        item["dataset_index"] = self._dataset_index_tensor
        return item


def pad_vector(vector: np.ndarray, new_dim: int) -> np.ndarray:
    """Pad the last dimension of a vector to a target size with zeros.

    Args:
        vector: Input numpy array to pad.
        new_dim: Target size for the last dimension.

    Returns:
        Padded array with the last dimension expanded to new_dim. If the
        vector already has the target dimension, returns it unchanged.
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype)
    new_vector[..., :current_dim] = vector
    return new_vector


# Case-insensitive sentinels that mean "no usable label" when resolving
# the (robot_type, control_mode) norm key:
#   - None: field absent from info.json.
#   - "" / whitespace-only: explicit clear via `DatasetConfig.{robot_type,control_mode}=""`.
#   - "unknown" (any casing): the `LeRobotDatasetMetadata.control_mode`
#     default for missing `info.json["control_mode"]`, and a common stand-in
#     for missing `info.json["robot_type"]` — not an explicit user-supplied tag.
# Both fields share the same sentinel set so a dataset labeled
# `robot_type="unknown"` doesn't silently anchor a `"unknown::<mode>"` head
# that pools across unrelated embodiments.
_NORM_KEY_MISSING_VALUES: frozenset[str] = frozenset({"unknown"})


def compute_norm_key(
    robot_type: str | None, control_mode: str | None, fallback_name: str
) -> tuple[str, bool]:
    """Compute the normalization-head key for a dataset.

    Datasets that share the same `(robot_type, control_mode)` are
    expected to share normalization stats because they share the units,
    axis count, and physical ranges of the proprio / action vectors.
    When either tag is missing, fall back to keying by the dataset's own
    name so the dataset still gets a head — it just won't share one with
    anything else.

    A value is treated as missing when it is `None`, empty after
    `strip()`, or matches `"unknown"` case-insensitively (the sentinel
    that `LeRobotDatasetMetadata.control_mode` returns when
    `info.json["control_mode"]` is absent — and the typical stand-in for
    missing `robot_type`).

    Args:
        robot_type: From `meta.info["robot_type"]` (after overrides).
        control_mode: From `meta.info["control_mode"]` (after overrides).
        fallback_name: The dataset's deduplicated mixture-level name,
            used as the head key when fallback fires.

    Returns:
        A `(norm_key, fallback_fired)` pair. `norm_key` is
        `"<robot_type>::<control_mode>"` (preserving the original casing
        of each tag) on the happy path and `fallback_name` otherwise;
        `fallback_fired` is True iff the fallback path was taken.
    """
    rt = (robot_type or "").strip()
    cm = (control_mode or "").strip()
    if not rt or not cm:
        return fallback_name, True
    if rt.casefold() in _NORM_KEY_MISSING_VALUES or cm.casefold() in _NORM_KEY_MISSING_VALUES:
        return fallback_name, True
    return f"{rt}::{cm}", False


def _apply_data_feature_name_mapping_overrides(
    worker_id: int, mapping_overrides: dict[str, dict[str, str]]
) -> None:
    """Apply repo-specific feature mapping overrides in each worker process.

    DataLoader workers run in separate processes, so they do not share module-level
    global mutations from the parent process when using spawn.
    """
    del worker_id  # required by DataLoader worker_init_fn signature
    DATA_FEATURES_NAME_MAPPING.update(mapping_overrides)


class DatasetMixtureMetadata:
    """Per-(robot_type, control_mode) normalization metadata for a mixture.

    Each underlying dataset's stats are normalised into the standard data
    format (feature renaming, state/action padding to ``cfg.max_state_dim`` /
    ``cfg.max_action_dim``, missing-camera zero placeholders). Datasets
    sharing the same ``(robot_type, control_mode)`` are then grouped into a
    single **norm head** whose stats are sample-count-pooled via
    :func:`~opentau.datasets.compute_stats.aggregate_stats`. The policy's
    ``Normalize`` / ``Unnormalize`` layers stack one row per norm head and
    use a per-sample index (chosen via ``dataset_to_norm_index``) to select
    the right row.

    Datasets whose ``(robot_type, control_mode)`` pair is missing — empty,
    ``None``, whitespace, or the ``"unknown"`` sentinel — fall back to
    keying by the dataset's own deduplicated mixture name, giving them a
    private head. See :func:`compute_norm_key`.

    Attributes:
        per_dataset_stats: One entry per underlying dataset, parallel to
            ``dataset_names``. Used by ``aggregated_action_stats()`` and
            kept for diagnostic / back-compat consumers.
        dataset_names: Ordered deduplicated mixture-level names (matches
            ``WeightedDatasetMixture._make_dataset_names`` output).
        dataset_name_to_index: ``{name: i}`` reverse lookup over
            ``dataset_names`` (per-dataset axis, NOT the norm-head axis).
        per_norm_key_stats: One entry per unique norm head, parallel to
            ``norm_keys``. This is what the policy's stacked
            Normalize / Unnormalize buffers consume.
        norm_keys: Ordered deduplicated norm-head identifiers
            (``"<robot_type>::<control_mode>"`` or fallback dataset name).
        norm_key_to_index: ``{norm_key: row}`` reverse lookup over
            ``norm_keys`` — the norm-head axis on the policy.
        dataset_to_norm_index: ``{dataset_name: norm_head_row}`` —
            the per-sample mapping consumed by ``_TaggedDataset`` at
            training time and by the policy at inference.
        norm_key_to_dataset_names: ``{norm_key: [dataset_name, ...]}`` —
            operator diagnostic showing which datasets share each head.
    """

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        metadatas: List[DatasetMetadata],
        dataset_weights: List[float],
        dataset_names: List[str] | None = None,
    ):
        self.cfg = cfg
        self._dataset_weights = list(dataset_weights)

        # Snapshot raw state / action dims BEFORE `_to_standard_data_format`
        # pads the standardized stats. Used to surface incompatible-dim
        # configurations (two datasets sharing a (robot_type, control_mode)
        # but with mismatched proprio / action widths). Done up front because
        # the standardize loop below clobbers `metadata.stats`.
        raw_dims: list[tuple[int, int]] = []
        for m in metadatas:
            key = feature_mapping_key(m.repo_id, getattr(m, "info", {}).get("control_mode"))
            name_map = DATA_FEATURES_NAME_MAPPING[key if key in DATA_FEATURES_NAME_MAPPING else m.repo_id]
            raw_dims.append(
                (
                    int(m.stats[name_map["state"]]["mean"].shape[-1]),
                    int(m.stats[name_map["actions"]]["mean"].shape[-1]),
                )
            )

        # convert each metadata stats to the standard data format
        for metadata in metadatas:
            metadata.stats = self._to_standard_data_format(
                metadata.repo_id, metadata.stats, getattr(metadata, "info", {}).get("control_mode")
            )

        # Per-dataset stats kept for diagnostic / back-compat consumers
        # (e.g. `aggregated_action_stats` for the BPE codec).
        self.per_dataset_stats: list[dict[str, dict[str, np.ndarray]]] = [m.stats for m in metadatas]

        # Names default to repo_id when WeightedDatasetMixture didn't supply
        # deduplicated names (e.g. tests that instantiate the metadata
        # directly). Duplicates would break the str -> index mapping; surface
        # them rather than silently keeping the last one.
        if dataset_names is None:
            dataset_names = [m.repo_id for m in metadatas]
        if len(dataset_names) != len(metadatas):
            raise ValueError(f"dataset_names ({len(dataset_names)}) must match metadatas ({len(metadatas)}).")
        if len(set(dataset_names)) != len(dataset_names):
            dups = [n for n in dataset_names if dataset_names.count(n) > 1]
            raise ValueError(
                f"dataset_names must be unique; got duplicates {sorted(set(dups))}. "
                "Use WeightedDatasetMixture's `_make_dataset_names` which appends "
                "`#N` suffixes for repeated repo ids."
            )
        self.dataset_names: list[str] = list(dataset_names)
        self.dataset_name_to_index: dict[str, int] = {n: i for i, n in enumerate(self.dataset_names)}

        # Compute per-dataset norm keys and group into norm heads.
        (
            self.norm_keys,
            self.per_norm_key_stats,
            self.norm_key_to_index,
            self.dataset_to_norm_index,
            self.norm_key_to_dataset_names,
        ) = self._build_norm_heads(metadatas, raw_dims)

    def _build_norm_heads(
        self,
        metadatas: List[DatasetMetadata],
        raw_dims: list[tuple[int, int]],
    ) -> tuple[
        list[str],
        list[dict[str, dict[str, np.ndarray]]],
        dict[str, int],
        dict[str, int],
        dict[str, list[str]],
    ]:
        """Aggregate per-dataset stats into per-(robot_type, control_mode) heads.

        See the class docstring for the full contract on the returned
        structures.
        """
        per_dataset_norm_keys: list[str] = []
        fallback_dataset_names: list[str] = []
        for ds_name, meta in zip(self.dataset_names, metadatas, strict=True):
            info = getattr(meta, "info", {}) or {}
            key, fallback_fired = compute_norm_key(info.get("robot_type"), info.get("control_mode"), ds_name)
            per_dataset_norm_keys.append(key)
            if fallback_fired:
                fallback_dataset_names.append(ds_name)

        # Insertion-order-preserving dedup; one head per unique key.
        norm_keys: list[str] = list(dict.fromkeys(per_dataset_norm_keys))
        norm_key_to_index: dict[str, int] = {k: i for i, k in enumerate(norm_keys)}
        dataset_to_norm_index: dict[str, int] = {
            ds_name: norm_key_to_index[k]
            for ds_name, k in zip(self.dataset_names, per_dataset_norm_keys, strict=True)
        }
        norm_key_to_dataset_names: dict[str, list[str]] = {k: [] for k in norm_keys}
        for ds_name, k in zip(self.dataset_names, per_dataset_norm_keys, strict=True):
            norm_key_to_dataset_names[k].append(ds_name)

        # Dimensional incompatibility check: two datasets sharing a
        # non-fallback (robot_type, control_mode) pair MUST have the same
        # raw state and action widths. Fallback-keyed groups are singletons
        # by construction (each fallback dataset is its own norm key) so
        # they cannot trip this check.
        for k in norm_keys:
            member_indices = [i for i, dnk in enumerate(per_dataset_norm_keys) if dnk == k]
            if len(member_indices) <= 1:
                continue
            # Fallback rows are singletons (a fallback key == the unique
            # dataset name), so a key with >1 member is necessarily a real
            # (robot_type, control_mode) pair.
            unique_dims = {raw_dims[i] for i in member_indices}
            if len(unique_dims) > 1:
                offenders = [
                    f"{self.dataset_names[i]} (state_dim={raw_dims[i][0]}, action_dim={raw_dims[i][1]})"
                    for i in member_indices
                ]
                raise ValueError(
                    f"Datasets sharing norm key {k!r} have incompatible raw "
                    "(pre-padding) state/action dims, so their normalization "
                    "stats cannot be pooled. Override `DatasetConfig.robot_type` "
                    "or `DatasetConfig.control_mode` to split them into "
                    f"distinct heads. Offenders: {offenders}"
                )

        # Per-norm-head stats. Singletons reuse the dataset's standardized
        # stats verbatim; multi-member groups go through
        # `aggregate_stats` weighted by each contributor's `info["total_frames"]`
        # (true sample-count pooling — passing `weights=` replaces the
        # internal `count` field, so we use total_frames rather than the
        # mixture sampling weights, which would conflate distributional
        # pooling with training-time prevalence).
        per_norm_key_stats: list[dict[str, dict[str, np.ndarray]]] = []
        for k in norm_keys:
            member_indices = [i for i, dnk in enumerate(per_dataset_norm_keys) if dnk == k]
            if len(member_indices) == 1:
                per_norm_key_stats.append(self.per_dataset_stats[member_indices[0]])
                continue
            member_stats = [self.per_dataset_stats[i] for i in member_indices]
            member_weights = [
                float(getattr(metadatas[i], "info", {}).get("total_frames", 1) or 1) for i in member_indices
            ]
            per_norm_key_stats.append(aggregate_stats(member_stats, weights=member_weights))

        # One aggregated warning for fallback datasets, capped at 10 names
        # so a 30-dataset mixture without tags doesn't spam the log.
        if fallback_dataset_names:
            shown = fallback_dataset_names[:10]
            suffix = (
                f", ... and {len(fallback_dataset_names) - 10} more"
                if len(fallback_dataset_names) > 10
                else ""
            )
            logging.warning(
                "DatasetMixtureMetadata: %d dataset(s) lack a usable "
                "(robot_type, control_mode) pair; falling back to "
                "dataset-name keying for their normalization heads. "
                "Set DatasetConfig.robot_type / control_mode (or fix "
                "info.json) to enable cross-dataset stat sharing. "
                "Fallback datasets: %s%s",
                len(fallback_dataset_names),
                shown,
                suffix,
            )

        # Operator diagnostic: log {norm_key: [dataset_repo_ids]} so the
        # train log shows exactly which datasets share each head.
        summary_lines = [
            f"Norm-head aggregation: {len(norm_keys)} heads over {len(self.dataset_names)} datasets."
        ]
        for k in sorted(norm_keys):
            ds_list = norm_key_to_dataset_names[k]
            shown_ds = ds_list[:10]
            ds_suffix = f", ... and {len(ds_list) - 10} more" if len(ds_list) > 10 else ""
            summary_lines.append(f"  - {k}: {shown_ds}{ds_suffix}")
        logging.info("\n".join(summary_lines))

        return (
            norm_keys,
            per_norm_key_stats,
            norm_key_to_index,
            dataset_to_norm_index,
            norm_key_to_dataset_names,
        )

    def aggregated_action_stats(self) -> dict[str, np.ndarray]:
        """Single mixture-wide action stats (mean/std/min/max/count).

        Backwards-compat helper for the rare consumers that genuinely need a
        single set of action stats across the whole mixture — currently only
        ``fit_fast_tokenizer.py``, which fits one BPE codec over a global
        action range. Most callers should consume ``per_dataset_stats`` /
        ``dataset_names`` directly.
        """
        # Co-iterate stats and weights so a non-trailing dataset lacking
        # ``actions`` keeps the weight alignment correct. Slicing
        # ``self._dataset_weights[:N]`` would silently misalign if e.g.
        # `per_dataset_stats[1]` lacked actions — the BPE codec would then
        # fit a weighted mean using dataset 0's weight applied to dataset 2's
        # action distribution.
        filtered: list[tuple[dict[str, np.ndarray], float]] = [
            ({"actions": s["actions"]}, w)
            for s, w in zip(self.per_dataset_stats, self._dataset_weights, strict=True)
            if "actions" in s
        ]
        if not filtered:
            raise ValueError(
                "No dataset in the mixture exposes 'actions' stats; aggregated_action_stats() is undefined."
            )
        agg = aggregate_stats(
            [s for s, _ in filtered],
            weights=[w for _, w in filtered],
        )
        return agg["actions"]

    def _to_standard_data_format(
        self, repo_id: str, stats: dict[str, dict[str, np.ndarray]], control_mode: str | None = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert statistics to the standard data format.

        Maps feature names from dataset-specific format to standard format,
        pads state and action vectors, and ensures all required cameras are present.

        Args:
            repo_id: Repository ID used to look up feature name mapping.
            stats: Statistics dictionary with dataset-specific feature names.
            control_mode: Control mode used (with ``repo_id``) to resolve the
                feature-name mapping, so dual-split entries (joint vs ee) pick
                their own action column. ``None`` falls back to the plain
                ``repo_id`` entry.

        Returns:
            Statistics dictionary with standard feature names and padded vectors.

        Raises:
            KeyError: If a required feature is missing from stats or if required
                statistics (mean, std, min, max) are missing.
        """
        key = feature_mapping_key(repo_id, control_mode)
        name_map = DATA_FEATURES_NAME_MAPPING[key if key in DATA_FEATURES_NAME_MAPPING else repo_id]
        features_without_stats = ["prompt", "response", "advantage"]

        standard_stats = {}
        for new_key, key in name_map.items():
            if new_key in features_without_stats:
                # skip features that do not have stats
                continue

            # ensure only the first num_cams is used
            if new_key.startswith("camera"):
                cam_idx = int(new_key[len("camera") :])
                if cam_idx >= self.cfg.num_cams:
                    continue
            if key in stats:
                standard_stats[new_key] = stats[key]
            else:
                raise KeyError(f"Key '{key}' not found in stats. Available keys: {list(stats.keys())}")

        # pad state and action vectors
        for stat in standard_stats["state"]:
            if stat in ["mean", "std", "min", "max"]:
                standard_stats["state"][stat] = pad_vector(
                    standard_stats["state"][stat], self.cfg.max_state_dim
                )
                standard_stats["actions"][stat] = pad_vector(
                    standard_stats["actions"][stat], self.cfg.max_action_dim
                )

        # pad missing cameras
        for cam_idx in range(self.cfg.num_cams):
            if f"camera{cam_idx}" in standard_stats:
                continue
            cam_stats: dict[str, np.ndarray] = {
                "min": np.zeros((3, 1, 1), dtype=np.float32),
                "max": np.ones((3, 1, 1), dtype=np.float32),
                "mean": np.zeros((3, 1, 1), dtype=np.float32),
                "std": np.zeros((3, 1, 1), dtype=np.float32),
            }
            # `count` is only present in v2.1+ stats; LeRobot v2.0-format
            # datasets carry only mean/std/min/max. Mirror it from the state stats when
            # available — the empty-camera placeholder doesn't actually
            # need a meaningful count, but downstream code that asserts
            # the field's presence shouldn't crash. Issue #264.
            if "count" in standard_stats.get("state", {}):
                cam_stats["count"] = np.array(standard_stats["state"]["count"])
            standard_stats[f"camera{cam_idx}"] = cam_stats

        # check for missing keys
        for data in standard_stats:
            missing_keys = {"mean", "std", "min", "max"} - standard_stats[data].keys()
            if missing_keys:
                raise KeyError(
                    f"The dataset {repo_id} is missing required statistics: {', '.join(sorted(missing_keys))}"
                )

        # Surface non-finite stats with the offending repo + dim, so that the
        # non-finite-tolerant aggregator in `aggregate_feature_stats` doesn't
        # silently drop a dataset's contribution. Cameras get
        # min=0/max=1/mean=0/std=0 placeholders for missing slots, so we only
        # check non-camera features.
        for data in standard_stats:
            if data.startswith("camera"):
                continue
            for stat_name in ("mean", "std", "min", "max"):
                arr = np.asarray(standard_stats[data][stat_name])
                bad = np.flatnonzero(~np.isfinite(arr.ravel()))
                if bad.size > 0:
                    logging.warning(
                        "Dataset %r: non-finite values in %r.%r at flat indices %s "
                        "(shape=%s); these dims are excluded from aggregation.",
                        repo_id,
                        data,
                        stat_name,
                        bad.tolist(),
                        tuple(arr.shape),
                    )

        return standard_stats

    @property
    def features(self) -> dict[str, dict]:
        """Return standard data format"""
        features = {
            "state": {
                "shape": (self.cfg.max_state_dim,),
                "dtype": "float32",
            },
            "actions": {
                "shape": (self.cfg.max_action_dim,),
                "dtype": "float32",
            },
        }
        # add camera features
        for i in range(self.cfg.num_cams):
            features[f"camera{i}"] = {
                "shape": (3, self.cfg.resolution[0], self.cfg.resolution[1]),
                "dtype": "image",
            }
        return features


class HierarchicalSampler(Sampler[int]):
    r"""With-replacement sampler for a ConcatDataset that first samples a dataset according to `dataset_probs`, and then
    samples uniformly within that dataset. This avoids multinomial over a huge number of categories (over 2^24)
    by operating at the dataset level.
    """

    def __init__(
        self,
        dataset_lengths: List[int],
        dataset_probs: List[float],
        num_samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 262144,
    ):
        super().__init__()

        if len(dataset_lengths) != len(dataset_probs):
            raise ValueError("dataset_lengths and dataset_probs must have the same length.")
        self.num_samples = int(num_samples)
        self.chunk_size = int(chunk_size)

        lens = torch.as_tensor(dataset_lengths, dtype=torch.long)
        probs = torch.as_tensor(dataset_probs, dtype=torch.double)

        if (lens < 0).any():
            raise ValueError("dataset_lengths must be non-negative.")

        # Offsets for mapping local indices to global ConcatDataset indices
        self._full_offsets = torch.zeros(len(lens), dtype=torch.long)
        if len(lens) > 0:
            self._full_offsets[1:] = lens.cumsum(0)[:-1]

        # Keep only non-empty datasets with positive probability
        valid_mask = (lens > 0) & (probs > 0)
        if not bool(valid_mask.any()):
            raise ValueError("All datasets are empty or have zero probability.")

        self._valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
        self._valid_lens = lens[self._valid_ids]
        valid_probs = probs[self._valid_ids]
        self._valid_probs = (valid_probs / valid_probs.sum()).to(dtype=torch.double)

        self._num_valid = int(self._valid_ids.numel())
        self._gen = generator if generator is not None else torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        # Generate indices in memory-friendly chunks
        total = self.num_samples
        cs = self.chunk_size
        for start in range(0, total, cs):
            m = min(cs, total - start)

            # Choose dataset ids according to probs (over valid ids only)
            ds_choices_valid = torch.multinomial(self._valid_probs, m, replacement=True, generator=self._gen)

            # For each chosen dataset, draw uniform local indices and map to global indices
            out = torch.empty(m, dtype=torch.long)
            for k in range(self._num_valid):
                mask = ds_choices_valid == k
                k_count = int(mask.sum().item())
                if k_count == 0:
                    continue
                local_idx = torch.randint(0, int(self._valid_lens[k].item()), (k_count,), generator=self._gen)
                orig_ds_id = int(self._valid_ids[k].item())
                out[mask] = local_idx + self._full_offsets[orig_ds_id]

            # Yield one by one to conform to Sampler API
            for idx in out.tolist():
                yield int(idx)


class WeightedDatasetMixture:
    """
    A class to combine multiple PyTorch Datasets and create a DataLoader
    that samples from them according to specified weightings.
    """

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        datasets: List[BaseDataset],
        dataset_weights: List[float],
        action_freq: Optional[float],
    ):
        """
        Initializes the WeightedDatasetMixture.

        Args:
            cfg (TrainPipelineConfig): Configuration for the training pipeline.
            datasets (List[Dataset]): A list of PyTorch Dataset objects.
            dataset_weights (List[float]): A list of weights corresponding to each dataset.
                                          These determine the relative sampling frequency.
            action_freq (Optional[float]): Common action frequency (Hz) the
                mixture's datasets are resampled to. ``None`` means no
                resampling — each dataset is sampled at its native fps, so a
                single batch may mix samples from sources running at different
                rates (mixed-frequency training). Stored as informational
                state and forwarded to downstream consumers (e.g.
                ``BaseDataset._action_freq``); not used arithmetically here.
        """
        if not datasets:
            raise ValueError("The list of datasets cannot be empty.")
        if len(datasets) != len(dataset_weights):
            raise ValueError("The number of datasets must match the number of dataset_weights.")
        if any(w < 0 for w in dataset_weights):
            raise ValueError("Dataset weights must be non-negative.")
        if sum(dataset_weights) == 0 and any(len(ds) > 0 for ds in datasets):
            # If all weights are zero, but there's data, sampler will fail.
            # If all datasets are empty, sum of weights being zero is fine.
            logging.warning(
                "Warning: All dataset weights are zero. The sampler might not behave as expected if datasets have samples."
            )

        self.cfg = cfg
        # Common resample rate (Hz); None = mixed-frequency (native fps per dataset).
        self.action_freq: Optional[float] = action_freq
        self.dataset_weights = dataset_weights
        self.dataset_names = self._make_dataset_names(cfg, datasets)  # For logging + tagging

        # Validate meta presence on the UN-wrapped inputs (so the error
        # message points to the real source) before wrapping.
        if not all(hasattr(ds, "meta") and ds.meta is not None for ds in datasets):
            raise ValueError("All datasets must have a 'meta' attribute with valid metadata.")

        # Build the norm-head mapping FIRST so `_TaggedDataset` can tag each
        # sample with its norm-head row, not the per-dataset enumerate index.
        # `DatasetMixtureMetadata` clobbers `metadata.stats` via
        # `_to_standard_data_format`; this must run before any downstream
        # consumer that reads the raw per-feature stats.
        self.meta = DatasetMixtureMetadata(
            cfg,
            [ds.meta for ds in datasets],
            dataset_weights,
            dataset_names=self.dataset_names,
        )

        # Wrap every underlying dataset so __getitem__ injects
        # `dataset_repo_id: str` and `dataset_index: torch.long`. The
        # `dataset_index` value is the norm-head row index (datasets sharing
        # a `(robot_type, control_mode)` get the same value).
        self.datasets = [
            _TaggedDataset(ds, name, self.meta.dataset_to_norm_index[name])
            for ds, name in zip(datasets, self.dataset_names, strict=True)
        ]

        logging.info("Initializing WeightedDatasetMixture...")
        self._log_dataset_info()

        self.concatenated_dataset: ConcatDataset = ConcatDataset(self.datasets)
        logging.info(f"Total length of concatenated dataset: {len(self.concatenated_dataset)}")

        self.sample_weights: torch.Tensor = self._calculate_sample_weights()
        if self.sample_weights is None and len(self.concatenated_dataset) > 0:
            raise ValueError("Sample weights could not be calculated, but concatenated dataset is not empty.")
        elif self.sample_weights is not None and len(self.sample_weights) != len(self.concatenated_dataset):
            raise ValueError(
                f"Length of sample_weights ({len(self.sample_weights)}) "
                f"must match concatenated_dataset length ({len(self.concatenated_dataset)})."
            )
        logging.info("-" * 30)

    @staticmethod
    def _make_dataset_names(cfg: TrainPipelineConfig, datasets: List[BaseDataset]) -> List[str]:
        """Derive human-readable names for each dataset in the mixture.

        Uses each ``DatasetConfig``'s ``repo_id`` or ``vqa`` identifier when
        available (ordered to match ``cfg.dataset_mixture.datasets``). Duplicates
        get a per-name sequential ``#<i>`` suffix (so ``['A','B','A']`` becomes
        ``['A#0','B','A#1']``). Falls back to the dataset class name plus index
        when the config list cannot be lined up with ``datasets`` (e.g. in tests
        that construct a mixture directly).
        """
        dataset_cfgs = getattr(getattr(cfg, "dataset_mixture", None), "datasets", None)
        if dataset_cfgs is None or len(dataset_cfgs) != len(datasets):
            return [type(ds).__name__ + f"_{i}" for i, ds in enumerate(datasets)]

        raw_names = [
            (dc.repo_id or dc.vqa or type(ds).__name__) for dc, ds in zip(dataset_cfgs, datasets, strict=True)
        ]
        counts = Counter(raw_names)
        seen: dict[str, int] = {}
        out: list[str] = []
        for name in raw_names:
            if counts[name] > 1:
                i = seen.get(name, 0)
                out.append(f"{name}#{i}")
                seen[name] = i + 1
            else:
                out.append(name)
        return out

    def _log_dataset_info(self) -> None:
        """Log information about all datasets in the mixture."""
        logging.info("Dataset information:")
        for i, ds in enumerate(self.datasets):
            logging.info(f"  - {self.dataset_names[i]}: Length={len(ds)}, Weight={self.dataset_weights[i]}")
        logging.info("-" * 30)

    def _calculate_sample_weights(self) -> Optional[torch.Tensor]:
        """Calculate the weight for each individual sample in the concatenated dataset.

        Samples from datasets with higher weights or smaller sizes (for a given weight)
        will have higher individual sample weights. Weight per sample = dataset_weight / dataset_length.

        Returns:
            Tensor of sample weights, or None if all datasets are empty or have zero weight.

        Raises:
            RuntimeError: If there's a mismatch between concatenated dataset length
                and calculated sample weights.
        """
        if not self.concatenated_dataset:  # Handles case where all input datasets are empty
            logging.warning("Warning: Concatenated dataset is empty. No sample weights to calculate.")
            return None

        logging.info("Calculating per-sample weights...")
        all_sample_weights: List[float] = []
        dataset_lengths = [len(ds) for ds in self.datasets]

        for i, length in enumerate(dataset_lengths):
            dataset_name = self.dataset_names[i]
            current_dataset_weight = self.dataset_weights[i]

            if length == 0:
                logging.info(f"  Skipping {dataset_name} (length 0).")
                continue  # Skip empty datasets

            if current_dataset_weight == 0:
                # Assign zero weight to all samples in this dataset
                weight_per_sample = 0.0
                logging.info(
                    f"  Weight for each sample in {dataset_name} (size {length}): {weight_per_sample:.10f} (dataset weight is 0)"
                )
            else:
                # Standard calculation: dataset_weight / num_samples_in_dataset
                weight_per_sample = current_dataset_weight / length
                logging.info(
                    f"  Weight for each sample in {dataset_name} (size {length}): {weight_per_sample:.10f}"
                )

            all_sample_weights.extend([weight_per_sample] * length)

        if not all_sample_weights:  # All datasets were empty or had 0 weight
            if len(self.concatenated_dataset) > 0:  # Should not happen if logic is correct
                raise RuntimeError(
                    "Mismatch: concatenated_dataset has samples but all_sample_weights is empty."
                )
            logging.warning(
                "Warning: All datasets are effectively empty or have zero weight. Sample weights list is empty."
            )
            return None  # No samples to weight

        return torch.DoubleTensor(all_sample_weights)

    def _get_worker_name_mapping_overrides(self) -> dict[str, dict[str, str]]:
        """Collect per-dataset mapping overrides from config for worker init.

        Emits both the plain ``repo_id`` key (back-compat fallback) and the
        control-mode-aware ``repo_id::control_mode`` key, so that after a
        ``spawn`` each worker can resolve dual-split entries (joint vs ee) to
        their own action column instead of last-wins on ``repo_id``.
        """
        overrides: dict[str, dict[str, str]] = {}
        for dataset_cfg in self.cfg.dataset_mixture.datasets:
            if dataset_cfg.repo_id and dataset_cfg.data_features_name_mapping is not None:
                overrides[dataset_cfg.repo_id] = dataset_cfg.data_features_name_mapping
                key = feature_mapping_key(dataset_cfg.repo_id, dataset_cfg.control_mode)
                if key != dataset_cfg.repo_id:
                    overrides[key] = dataset_cfg.data_features_name_mapping
        return overrides

    def get_dataloader(self) -> DataLoader:
        """Create and return a PyTorch DataLoader with weighted sampling.

        Uses HierarchicalSampler to first sample a dataset according to weights,
        then uniformly sample within that dataset.

        Returns:
            DataLoader configured for weighted hierarchical sampling.

        Raises:
            ValueError: If no non-empty dataset has a positive sampling weight.
        """
        worker_name_mapping_overrides = self._get_worker_name_mapping_overrides()
        worker_init_fn = None
        if worker_name_mapping_overrides:
            worker_init_fn = functools.partial(
                _apply_data_feature_name_mapping_overrides,
                mapping_overrides=worker_name_mapping_overrides,
            )

        if len(self.concatenated_dataset) == 0:
            logging.warning("Warning: Concatenated dataset is empty. DataLoader will produce no batches.")
            # Return an empty dataloader or raise error, depending on desired behavior.
            # For now, let it create an empty dataloader.
            return DataLoader(
                self.concatenated_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                worker_init_fn=worker_init_fn,
            )

        # Validate there is at least one non-empty dataset with positive weight
        if not any(len(ds) > 0 and w > 0 for ds, w in zip(self.datasets, self.dataset_weights, strict=True)):
            logging.error("Error: No non-empty dataset has a positive sampling weight.")
            raise ValueError("No non-empty dataset has a positive sampling weight.")

        num_samples_per_epoch = len(self.concatenated_dataset)
        logging.info("\nCreating DataLoader...")
        logging.info(f"  Batch size: {self.cfg.batch_size}")
        logging.info(f"  Samples per epoch (num_samples for sampler): {num_samples_per_epoch}")

        # Hierarchical sampling: choose dataset by weight, then uniform within it (both with replacement)
        ds_lengths = [len(ds) for ds in self.datasets]
        sampler = HierarchicalSampler(
            dataset_lengths=ds_lengths,
            dataset_probs=self.dataset_weights,
            num_samples=num_samples_per_epoch,
        )

        dataloader = DataLoader(
            self.concatenated_dataset,
            batch_size=self.cfg.dataloader_batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            prefetch_factor=self.cfg.prefetch_factor,
            worker_init_fn=worker_init_fn,
        )
        logging.info("DataLoader created successfully.")
        logging.info("-" * 30)
        return dataloader

    def get_per_dataset_dataloaders(self) -> dict[str, DataLoader]:
        """Create one sequential DataLoader per underlying dataset.

        Intended for per-dataset evaluation (e.g. per-dataset validation loss),
        where each dataset should be iterated exactly once rather than mixed
        via weighted hierarchical sampling. Empty datasets are skipped.

        Returns:
            Mapping from ``dataset_name`` to its DataLoader.
        """
        worker_name_mapping_overrides = self._get_worker_name_mapping_overrides()
        worker_init_fn = None
        if worker_name_mapping_overrides:
            worker_init_fn = functools.partial(
                _apply_data_feature_name_mapping_overrides,
                mapping_overrides=worker_name_mapping_overrides,
            )

        loaders: dict[str, DataLoader] = {}
        for name, ds in zip(self.dataset_names, self.datasets, strict=True):
            if len(ds) == 0:
                logging.info(f"Skipping per-dataset DataLoader for empty dataset '{name}'.")
                continue
            loaders[name] = DataLoader(
                ds,
                batch_size=self.cfg.dataloader_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                prefetch_factor=self.cfg.prefetch_factor,
                worker_init_fn=worker_init_fn,
            )
        return loaders
