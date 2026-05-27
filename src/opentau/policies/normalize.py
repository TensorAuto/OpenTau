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

"""Per-dataset Normalize / Unnormalize for VLA policies.

Each feature buffer has shape ``(D, *feat_shape)`` where ``D`` is the number of
datasets the policy was trained on. ``forward`` accepts a per-sample
``dataset_index: torch.LongTensor`` of shape ``(B,)`` and gathers the right
row per sample before broadcasting.
"""

import sys

import numpy as np
import torch
from torch import Tensor, nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature

EPS = 1e-8  # Small epsilon value for numerical stability in normalization


def _materialize(tensor: Tensor) -> Tensor:
    """Return the full local Tensor for ``tensor``, materializing if it is a DTensor.

    Under FSDP2 (``fully_shard``), parameters and buffers attached to a
    ``fully_shard``-wrapped module are exposed as ``DTensor`` instances.
    Mixing a ``DTensor`` with a regular ``Tensor`` in arithmetic raises:

        RuntimeError: aten.sub.Tensor got mixed torch.Tensor and DTensor,
        need to convert all torch.Tensor to DTensor before calling
        distributed operators!

    Normalize / Unnormalize stats (``mean``/``std``/``min``/``max``) are tiny
    and replicated across ranks, so the cost of redistributing to a full
    Tensor for the per-feature arithmetic is negligible. Under FSDP1, DDP,
    or single-process this helper is a fast no-op (the Parameter is already
    a regular Tensor).
    """
    if hasattr(tensor, "full_tensor"):
        return tensor.full_tensor()
    return tensor


def warn_missing_keys(features: dict[str, PolicyFeature], batch: dict[str, Tensor], mode: str) -> None:
    """Warns if expected features are missing from the batch.

    Args:
        features: Dictionary of expected policy features.
        batch: Dictionary containing the data batch.
        mode: The operation mode (e.g., "normalization" or "unnormalization") for the warning message.
    """
    for missing_key in set(features) - set(batch):
        red_seq = "\033[91m"
        reset_seq = "\033[0m"
        print(
            f"{red_seq}Warning: {missing_key} was missing from the batch during {mode}.{reset_seq}",
            file=sys.stderr,
        )


def resolve_num_datasets(
    per_dataset_stats: list | None,
    dataset_names: list | None,
    config,
) -> int:
    """Decide the leading dim for the stacked stats buffers.

    Preference order:
        1. ``len(per_dataset_stats)`` if provided.
        2. ``len(dataset_names)`` if provided.
        3. ``len(config.dataset_names)`` if the policy config remembers a
           trained-on list (checkpoint-load construction path).
        4. ``1`` — legacy single-dataset default.
    """
    if per_dataset_stats is not None:
        return len(per_dataset_stats)
    if dataset_names is not None:
        return len(dataset_names)
    cfg_names = getattr(config, "dataset_names", None)
    if cfg_names:
        return len(cfg_names)
    return 1


def _stat_to_float32_tensor(value: np.ndarray | Tensor) -> Tensor:
    """Convert one per-dataset stat (np or torch) to a float32 ``Tensor``.

    Mirrors the np/torch dispatch the legacy single-stats path used: caller
    receives a fresh tensor (cloned for torch inputs) so the stacked buffer
    doesn't alias the input.
    """
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(dtype=torch.float32)
    if isinstance(value, Tensor):
        # Clone so the stacked buffer doesn't alias an input the caller may mutate.
        return value.clone().to(dtype=torch.float32)
    raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type(value)}' instead.")


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    per_dataset_stats: list[dict[str, dict[str, Tensor | np.ndarray]]] | None = None,
    num_datasets: int | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """Create per-feature stat buffers shaped ``(D, *feat_shape)``.

    Args:
        features: Mapping feature name -> PolicyFeature.
        norm_map: Mapping FeatureType -> NormalizationMode.
        per_dataset_stats: Ordered list (one entry per dataset) of dicts shaped
            ``{feature_name: {"mean": ..., "std": ...}}`` or ``{... "min": ..., "max": ...}``.
            When ``None`` the buffers are initialised to ``+inf`` so the
            assertion in ``Normalize.forward`` fires until a checkpoint load
            or ``_inject_stats`` overwrites them.
        num_datasets: Required when ``per_dataset_stats is None``; ignored
            otherwise. Sets the leading dim of the inf-init buffers.

    Returns:
        dict feature_name -> nn.ParameterDict({"mean": param, "std": param}) etc.
    """
    if per_dataset_stats is None and num_datasets is None:
        raise ValueError(
            "create_stats_buffers requires either `per_dataset_stats` or `num_datasets` "
            "so the stacked stats buffers can be sized correctly."
        )
    if per_dataset_stats is not None:
        num_ds = len(per_dataset_stats)
        if num_ds == 0:
            raise ValueError("per_dataset_stats must contain at least one dataset.")
        if num_datasets is not None and num_datasets != num_ds:
            raise ValueError(
                f"num_datasets ({num_datasets}) disagrees with len(per_dataset_stats) ({num_ds})."
            )
    else:
        num_ds = num_datasets
        if num_ds <= 0:
            raise ValueError(f"num_datasets must be positive, got {num_ds}.")

    stats_buffers: dict[str, nn.ParameterDict] = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        stacked_shape = (num_ds, *shape)

        if norm_mode is NormalizationMode.MEAN_STD:
            stat_names = ("mean", "std")
        elif norm_mode is NormalizationMode.MIN_MAX:
            stat_names = ("min", "max")
        else:
            raise ValueError(norm_mode)

        params: dict[str, nn.Parameter] = {}
        for name in stat_names:
            if per_dataset_stats is None:
                # Inf-init across all D rows. Asserted-finite at forward time.
                tensor = torch.full(stacked_shape, torch.inf, dtype=torch.float32)
            else:
                rows = []
                for i, ds_stats in enumerate(per_dataset_stats):
                    if key not in ds_stats:
                        raise KeyError(
                            f"per_dataset_stats[{i}] is missing feature '{key}'. "
                            f"Available keys: {list(ds_stats)}."
                        )
                    if name not in ds_stats[key]:
                        raise KeyError(
                            f"per_dataset_stats[{i}]['{key}'] is missing stat '{name}'. "
                            f"Available stats: {list(ds_stats[key])}."
                        )
                    row = _stat_to_float32_tensor(ds_stats[key][name])
                    if tuple(row.shape) != shape:
                        raise ValueError(
                            f"per_dataset_stats[{i}]['{key}']['{name}'] has shape "
                            f"{tuple(row.shape)}, expected {shape}."
                        )
                    rows.append(row)
                tensor = torch.stack(rows, dim=0)
            params[name] = nn.Parameter(tensor, requires_grad=False)

        stats_buffers[key] = nn.ParameterDict(params)
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `per_dataset_stats` as an "
        "argument, call `policy._inject_stats(...)`, or load a pretrained model that contains "
        "the normalization buffers."
    )


def _gather_and_broadcast(stat: Tensor, dataset_index: Tensor, batch_val: Tensor) -> Tensor:
    """Index the stacked stat by per-sample dataset and broadcast to the batch tensor.

    ``stat`` is shape ``(D, *feat_shape)``. After ``index_select`` we get
    ``(B, *feat_shape)``. If ``batch_val`` has extra interior dims (temporal
    history axis, or full spatial HxW for an image batch), we pad ``mean``
    with 1-axes between the batch axis and the feature axes so broadcasting
    aligns. ``reshape`` with computed dims is used here rather than einops
    because the inserted axis count is data-dependent and varies per-feature.
    """
    gathered = stat.index_select(0, dataset_index)  # (B, *feat_shape)
    extra = batch_val.ndim - gathered.ndim
    if extra > 0:
        gathered = gathered.reshape(gathered.shape[0], *((1,) * extra), *gathered.shape[1:])
    return gathered


class Normalize(nn.Module):
    """Normalizes data (e.g. ``observation.image``) per-sample using the row of the
    stacked stat buffer selected by ``dataset_index``.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        per_dataset_stats: list[dict[str, dict[str, Tensor | np.ndarray]]] | None = None,
        dataset_names: list[str] | None = None,
        num_datasets: int | None = None,
    ):
        """Initializes the Normalize module.

        Args:
            features: Mapping feature name -> PolicyFeature.
            norm_map: Mapping FeatureType -> NormalizationMode.
            per_dataset_stats: Ordered list of per-dataset stat dicts. ``None``
                during checkpoint-load construction (the safetensors load
                overwrites the buffers afterwards); in that case
                ``num_datasets`` must be supplied so the inf-init buffers have
                the right leading dim.
            dataset_names: Ordered list of dataset name strings, parallel to
                ``per_dataset_stats``. Stored as a Python attribute, not a
                buffer (strings aren't tensors). Persisted into ``config.json``
                via ``PreTrainedConfig.dataset_names``.
            num_datasets: Required when ``per_dataset_stats is None``.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.dataset_names = list(dataset_names) if dataset_names is not None else None
        if (
            per_dataset_stats is not None
            and dataset_names is not None
            and len(per_dataset_stats) != len(dataset_names)
        ):
            raise ValueError(
                f"per_dataset_stats and dataset_names must have the same length, got "
                f"{len(per_dataset_stats)} vs {len(dataset_names)}."
            )
        stats_buffers = create_stats_buffers(
            features, norm_map, per_dataset_stats=per_dataset_stats, num_datasets=num_datasets
        )
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor], dataset_index: Tensor) -> dict[str, Tensor]:
        """Normalizes the batch data per-sample.

        Args:
            batch: Dictionary containing the data to normalize.
            dataset_index: LongTensor of shape ``(B,)`` giving the source
                dataset row for each sample.

        Returns:
            The normalized batch (shallow-copied; inputs are not mutated).
        """
        warn_missing_keys(self.features, batch, "normalization")
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            batch_val = batch[key]
            if batch_val.numel() == 0:  # skip empty tensors, which won't broadcast well
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = _gather_and_broadcast(_materialize(buffer["mean"]), dataset_index, batch_val)
                std = _gather_and_broadcast(_materialize(buffer["std"]), dataset_index, batch_val)
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch_val - mean) / (std + EPS)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = _gather_and_broadcast(_materialize(buffer["min"]), dataset_index, batch_val)
                max_ = _gather_and_broadcast(_materialize(buffer["max"]), dataset_index, batch_val)
                assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                batch[key] = (batch_val - min_) / (max_ - min_ + EPS)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)
        return batch


class Unnormalize(nn.Module):
    """Inverse of ``Normalize``: scales outputs back to the per-sample dataset range."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        per_dataset_stats: list[dict[str, dict[str, Tensor | np.ndarray]]] | None = None,
        dataset_names: list[str] | None = None,
        num_datasets: int | None = None,
    ):
        """See :class:`Normalize.__init__` for argument semantics."""
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.dataset_names = list(dataset_names) if dataset_names is not None else None
        if (
            per_dataset_stats is not None
            and dataset_names is not None
            and len(per_dataset_stats) != len(dataset_names)
        ):
            raise ValueError(
                f"per_dataset_stats and dataset_names must have the same length, got "
                f"{len(per_dataset_stats)} vs {len(dataset_names)}."
            )
        stats_buffers = create_stats_buffers(
            features, norm_map, per_dataset_stats=per_dataset_stats, num_datasets=num_datasets
        )
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor], dataset_index: Tensor) -> dict[str, Tensor]:
        """Unnormalizes the batch data per-sample.

        Args:
            batch: Dictionary containing the data to unnormalize.
            dataset_index: LongTensor of shape ``(B,)``.
        """
        warn_missing_keys(self.features, batch, "unnormalization")
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            batch_val = batch[key]
            if batch_val.numel() == 0:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = _gather_and_broadcast(_materialize(buffer["mean"]), dataset_index, batch_val)
                std = _gather_and_broadcast(_materialize(buffer["std"]), dataset_index, batch_val)
                if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
                    assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                    assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch_val * (std + EPS) + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = _gather_and_broadcast(_materialize(buffer["min"]), dataset_index, batch_val)
                max_ = _gather_and_broadcast(_materialize(buffer["max"]), dataset_index, batch_val)
                if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
                    assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                    assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                batch[key] = (batch_val + 1) / 2
                batch[key] = batch[key] * (max_ - min_ + EPS) + min_
            else:
                raise ValueError(norm_mode)
        return batch
