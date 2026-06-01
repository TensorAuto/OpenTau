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

import logging
import sys

import numpy as np
import torch
from torch import Tensor, nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature

EPS = 1e-8  # Small epsilon value for numerical stability in normalization

# A genuinely-zero std (|std| < EPS) marks a constant/padding feature dim. The guard in
# ``Normalize`` / ``Unnormalize`` snaps such a std (or a zero MIN_MAX range) to 1 so a value
# that *deviates* from that "constant" can't divide by ~EPS and blow up to ~1e8 (the
# "Outlier normalized state" failure). When the dim is truly constant and its stats are
# right, every value equals the mean, the deviation is 0, and the snap is a no-op.
#
# ``_SNAP_WARN_TOL`` is the raw-unit |value - mean| above which a snapped dim counts as a real
# deviation worth warning about. Set it <= 0 to disable the snap warning (and skip its per-step
# host sync) entirely — mirroring the state/action outlier warning's ``threshold <= 0``
# off-switch. The snap itself always runs, so disabling the warning never weakens the guard.
_SNAP_WARN_TOL = 1e-2

# ``(feature_key, dim)`` snap offender -> the largest raw deviation already warned about, so a
# pair is warned the first time it snaps and again only when a strictly larger deviation appears
# (mirrors ``_WARNED_OUTLIER_KEYS`` in ``opentau.scripts.train``).
_WARNED_SNAP_KEYS: dict[tuple[str, int], float] = {}


def _warn_snapped_deviation(
    key: str,
    zero_var_mask: Tensor,
    deviation: Tensor,
    dataset_index: Tensor,
    dataset_names: list[str] | None,
) -> None:
    """Loudly warn when the zero-variance guard changed the result (a real deviation).

    The guard snaps a ``std`` (or ``max - min``) of ~0 to 1. That only changes the output when
    the input *deviates* from the dim's constant ``mean`` (resp. ``min``): a genuinely-constant
    dim with correct stats has ``deviation == 0`` and stays silent. A non-trivial ``deviation``
    means the stats are stale or the value was zeroed upstream (e.g. a dropped frame) — surface
    the offending ``feature``/``dim`` so it can be traced.

    Log-only and called behind ``Normalize._snapping_possible()``; reads tensors without
    mutating them and fires no collective, so the ``forward`` graph stays identical across
    ranks (CLAUDE.md rule 5).

    Args:
        key: Feature name (e.g. ``"observation.state"``).
        zero_var_mask: Bool ``(B, *feat)`` — True where the gathered std/range was snapped.
        deviation: ``batch_val - mean`` (MEAN_STD) or ``batch_val - min`` (MIN_MAX), the raw
            pre-scale numerator; broadcasts against ``zero_var_mask`` onto the batch shape.
        dataset_index: ``(B,)`` per-sample dataset row, used to name the offending dataset.
        dataset_names: Ordered dataset names parallel to the buffer rows, or ``None``.
    """
    trigger = zero_var_mask & (deviation.abs() > _SNAP_WARN_TOL)
    if not bool(trigger.any()):
        return
    # Per-dim max deviation magnitude. The inserted-axis count is data-dependent (see
    # ``_gather_and_broadcast``), so a computed dim tuple is clearer than a fixed einops pattern.
    magnitude = deviation.abs() * trigger
    feat_axes = tuple(range(magnitude.ndim - 1))
    per_dim_mag = magnitude.amax(dim=feat_axes) if feat_axes else magnitude  # (dim,)
    dims = torch.nonzero(per_dim_mag > 0, as_tuple=False).flatten().tolist()
    # Warn a (feature, dim) the first time it snaps and again only when its deviation exceeds
    # the largest already warned for it.
    fresh = []
    for d in dims:
        val = float(per_dim_mag[d])
        prev = _WARNED_SNAP_KEYS.get((key, d))
        if prev is None or val > prev:
            _WARNED_SNAP_KEYS[(key, d)] = val
            fresh.append(d)
    if not fresh:
        return
    # Provenance: the worst fresh dim and the sample carrying the largest deviation there.
    worst_d = max(fresh, key=lambda d: float(per_dim_mag[d]))
    worst_val = float(per_dim_mag[worst_d])
    col = magnitude.select(-1, worst_d)
    sample_axes = tuple(range(1, col.ndim))
    worst_sample = int(torch.argmax(col.amax(dim=sample_axes) if sample_axes else col))
    ds = None
    if dataset_names is not None:
        idx = int(dataset_index[worst_sample])
        if 0 <= idx < len(dataset_names):
            ds = dataset_names[idx]
    logging.warning(
        "Normalize zero-variance guard fired: feature '%s' dim(s)=%s have std≈0 but a value "
        "deviates from the constant by up to %.3g (raw units, dataset=%s). std was snapped to 1 "
        "so only the mean is subtracted (no ~1e8 blow-up), but this almost always means stale "
        "normalization stats or a value zeroed upstream — inspect this feature's stats. "
        "Re-warned only when a (feature,dim) pair's deviation exceeds its last warned value.",
        key,
        fresh,
        worst_val,
        ds,
    )


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
    if extra < 0:
        # Batch tensor has fewer dims than the buffer — silent broadcasting
        # would produce a shape mismatch later, or worse, broadcast along
        # the wrong axis. Surface this loudly so the misconfiguration is
        # caught at the feature in question rather than mid-step.
        raise ValueError(
            f"Stats buffer rank ({gathered.ndim}) exceeds batch tensor rank "
            f"({batch_val.ndim}); expected batch_val to have at least as many "
            f"dims as the (D, *feat_shape) buffer. Stats shape={tuple(stat.shape)}, "
            f"gathered shape={tuple(gathered.shape)}, batch shape={tuple(batch_val.shape)}."
        )
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

    def _snapping_possible(self) -> bool:
        """Whether any stat buffer has a zero-variance dim, so the guard could ever snap.

        Cached after the first ``forward``. Lets ``forward`` skip the per-step deviation check
        (and its host sync) entirely on the common healthy path where no dim is constant —
        only a model that actually has a constant dim pays for the warning bookkeeping.

        The cache assumes stats are finalized before the first ``forward`` (OpenTau's
        load-then-train order); a zero-variance dim introduced by a *later* ``_inject_stats`` /
        ``load_state_dict`` would not flip the cached value. That only affects the warning — the
        snap itself runs unconditionally every ``forward``, so numerical safety never depends on it.
        """
        cached = getattr(self, "_snap_active", None)
        if cached is not None:
            return cached
        active = False
        for key, ft in self.features.items():
            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            buffer = getattr(self, "buffer_" + key.replace(".", "_"), None)
            if buffer is None:
                continue
            if norm_mode is NormalizationMode.MEAN_STD:
                std = _materialize(buffer["std"])
                if bool((torch.isfinite(std) & (std.abs() < EPS)).any()):
                    active = True
                    break
            elif norm_mode is NormalizationMode.MIN_MAX:
                rng = _materialize(buffer["max"]) - _materialize(buffer["min"])
                if bool((torch.isfinite(rng) & (rng.abs() < EPS)).any()):
                    active = True
                    break
        self._snap_active = active
        return active

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
                # Snap a genuinely-zero std (constant/padding dim) to 1 so a deviating value
                # can't become (x - mean)/EPS ~ 1e8. std >= EPS dims stay bit-identical.
                std_is_zero = std.abs() < EPS
                std = torch.where(std_is_zero, torch.ones_like(std), std)
                deviation = batch_val - mean
                batch[key] = deviation / (std + EPS)
                # Loud warning (deduped per (feature,dim), re-fired on a larger deviation), but
                # only when the snap actually changed the output (a real deviation from a
                # "constant" dim). The _SNAP_WARN_TOL>0 / compile / ONNX checks short-circuit before
                # the precheck so a disabled or traced forward adds no data-dependent op / sync.
                if (
                    _SNAP_WARN_TOL > 0
                    and not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export())
                    and self._snapping_possible()
                ):
                    _warn_snapped_deviation(key, std_is_zero, deviation, dataset_index, self.dataset_names)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = _gather_and_broadcast(_materialize(buffer["min"]), dataset_index, batch_val)
                max_ = _gather_and_broadcast(_materialize(buffer["max"]), dataset_index, batch_val)
                assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                # Snap a zero range (max == min: constant dim) to 1 — same blow-up class.
                denom = max_ - min_
                denom_is_zero = denom.abs() < EPS
                denom = torch.where(denom_is_zero, torch.ones_like(denom), denom)
                deviation = batch_val - min_
                batch[key] = deviation / (denom + EPS)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
                if (
                    _SNAP_WARN_TOL > 0
                    and not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export())
                    and self._snapping_possible()
                ):
                    _warn_snapped_deviation(key, denom_is_zero, deviation, dataset_index, self.dataset_names)
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
                # Mirror Normalize's guard so Unnormalize(Normalize(x)) round-trips on a snapped
                # dim: with std=1 the inverse x*(1+EPS)+mean recovers the deviation; the legacy
                # x*EPS+mean would collapse it to ~mean. std >= EPS dims stay bit-identical.
                std = torch.where(std.abs() < EPS, torch.ones_like(std), std)
                batch[key] = batch_val * (std + EPS) + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min_ = _gather_and_broadcast(_materialize(buffer["min"]), dataset_index, batch_val)
                max_ = _gather_and_broadcast(_materialize(buffer["max"]), dataset_index, batch_val)
                if not (torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()):
                    assert not torch.isinf(min_).any(), _no_stats_error_str("min")
                    assert not torch.isinf(max_).any(), _no_stats_error_str("max")
                # Snap a zero range to 1 to mirror Normalize (keeps the round-trip exact).
                denom = max_ - min_
                denom = torch.where(denom.abs() < EPS, torch.ones_like(denom), denom)
                batch[key] = (batch_val + 1) / 2
                batch[key] = batch[key] * (denom + EPS) + min_
            else:
                raise ValueError(norm_mode)
        return batch
