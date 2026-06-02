#!/usr/bin/env python
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
"""Fit a FAST action tokenizer on an OpenTau dataset mixture (CPU only).

The FAST tokenizer (``physical-intelligence/fast``) is used during pi0.5 /
pi0.7 training as an auxiliary cross-entropy target on top of the
flow-matching MSE loss -- it tokenizes action chunks via DCT + BPE. The
published upstream checkpoint was fit on a generic robotics mixture; this
script lets you specialize the BPE to your own action distribution by
fitting on the same min-max-normalized, FPS-resampled action chunks the
policy will see at training time.

Output is a ``processor`` directory loadable via
``AutoProcessor.from_pretrained(out_dir, trust_remote_code=True)``.

Pipeline (all CPU):

    1. Resolve ``$ref`` includes in the mixture JSON and parse it as a
       ``DatasetMixtureConfig``.
    2. Build a minimal ``TrainPipelineConfig`` (stub policy with
       ``action_delta_indices = list(range(chunk_size))``, ``num_cams=0``
       to skip video decode) and pass it to ``make_dataset_mixture`` so
       we reuse the same weighted-sampling + ``action_freq`` resampling
       pipeline the training loop uses. This is the ONLY way to guarantee
       the tokenizer is fit on the same chunk distribution the policy
       will see.
    3. Drain the weighted dataloader until ``--total-chunks`` action
       chunks have been collected; each batch yields chunks already
       resampled to ``mixture.action_freq`` (or to each dataset's native
       fps when ``action_freq is None`` -- mixed-frequency mixtures) and
       right-padded to ``max_action_dim``.
    4. Min-max-normalize each chunk to ``[-1, 1]`` using the per-
       ``(robot_type, control_mode)`` norm-head stats by default
       (``--per-head-norm``) -- the same stats the training policy
       applies via ``Normalize({"ACTION": NormalizationMode.MIN_MAX})``
       with per-head stacked buffers (PR #347). Per-dataset raw stats
       are zero-padded (matching ``pad_vector`` in
       ``_to_standard_data_format``) then pooled across head members
       with ``nanmin``/``nanmax`` (with ``±Inf`` masked first, like
       ``aggregate_stats``). Failure modes mirror training: any
       dataset whose stats fail to load -> ``RuntimeError`` (training
       would crash too via ``_to_standard_data_format``);
       ``require_non_empty_robot_type/control_mode`` -> same
       ``ValueError`` as ``datasets.factory._validate_metadata_requirements``.
       Pass ``--no-per-head-norm`` for the legacy global aggregate
       path, which is what older fits (pre-#347) used; the global
       path under-spreads each head's distribution and produces
       shorter fit-time chunks than the policy actually feeds the
       tokenizer at training, so token-length analysis on the global
       fit systematically underestimates the truncation rate.
       Passing ``--use-mixture-dataloader`` silently degrades to
       global normalization with a warning (the dataloader path
       doesn't surface per-sample dataset_index here yet).
    5. Call ``UniversalActionProcessor.fit(...)`` (DCT + Rust BpeTrainer)
       and ``save_pretrained`` the result. The upstream remote-code
       source ``processing_action_tokenizer.py`` is copied alongside so
       the saved directory loads via ``trust_remote_code=True``.
    6. Round-trip a held-out sample of chunks for a sanity-check MSE and
       average token length; write ``fit_report.json``.

Usage:

    python -m opentau.scripts.fit_fast_tokenizer \\
        --mixture-json /path/to/mixture.json \\
        --out-dir /path/to/output_dir/ \\
        --chunk-size 50 \\
        [--total-chunks 1000000] [--action-dim 32] \\
        [--vocab-size 2048] [--scale 10] [--seed 0] [--num-workers 8] \\
        [--dataloader-batch-size 256] [--pilot] \\
        [--no-per-head-norm]  # legacy global normalization (default: per-head)

The mixture JSON may use ``$ref`` includes -- see
``opentau.configs.refs.resolve_refs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import draccus
import numpy as np

from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.datasets.dataset_mixture import compute_norm_key

logger = logging.getLogger(__name__)

UPSTREAM_REPO_ID = "physical-intelligence/fast"
UPSTREAM_SOURCE_FILE = "processing_action_tokenizer.py"
# Commit SHA of physical-intelligence/fast that ``_fit_tokenizer`` was ported
# from. When upstream changes (new preprocessing, different defaults), bump
# this and re-port the body of ``UniversalActionProcessor.fit`` in
# ``_fit_tokenizer`` accordingly.
UPSTREAM_PINNED_SHA = "ec4d7aa71691cac0b8bed6942be45684db2110f4"

# How much merge headroom (in tokens) we want above the initial-alphabet floor
# before warning the user. Lower -> the BPE has very few real merges; raise
# ``--vocab-size`` to fix.
_MIN_MERGE_HEADROOM = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a FAST action tokenizer on an OpenTau dataset mixture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mixture-json",
        type=Path,
        required=True,
        help="Path to a DatasetMixtureConfig JSON (may use $ref).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the fitted tokenizer.",
    )
    p.add_argument("--total-chunks", type=int, default=1_000_000)
    p.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help=(
            "Time horizon per action chunk. MUST match the downstream policy's "
            "n_action_steps -- the BPE merges depend on chunk length. "
            "Every policy config in this repo (pi0, pi0.5, pi0.5-mem, pi0.6, "
            "pi0.7 low-level, pi0.7-paligemma low-level) defaults n_action_steps "
            "to 50; configs/examples/pi07_libero.json overrides to 10."
        ),
    )
    p.add_argument(
        "--action-dim",
        type=int,
        default=32,
        help=(
            "Padded action dimension. Must match the production policy's "
            "max_action_dim (pi0.5/pi0.6/pi0.7 default to 32). Datasets whose "
            "native action dim exceeds this value have their extra dims "
            "silently dropped from the BPE corpus -- those dims are never "
            "used at inference, so they shouldn't appear at fit time either."
        ),
    )
    p.add_argument(
        "--max-state-dim",
        type=int,
        default=128,
        help=(
            "Padded state dimension. Must be >= the largest native state dim in "
            "the mixture (same non-truncating pad path). We never read state, "
            "but the value flows through DatasetMixtureMetadata stats."
        ),
    )
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument(
        "--max-token-length",
        type=int,
        default=64,
        help=(
            "Cap on the byte length of any single learned BPE merge. The "
            "upstream FAST fit uses 10000 (effectively uncapped), which on a "
            "high-zero-pad corpus causes the BPE to spend its entire merge "
            "budget on one runaway zero-run token (lengths 2,3,4,...,N). "
            "Capping at ~chunk_size keeps merges focused on meaningful "
            "patterns. Set to >chunk_size*action_dim to reproduce the "
            "upstream behaviour."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dataloader-batch-size",
        type=int,
        default=256,
        help="Per-iteration batch size from the WeightedDatasetMixture dataloader.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Dataloader worker processes (passed to PyTorch DataLoader).",
    )
    p.add_argument(
        "--use-mixture-dataloader",
        action="store_true",
        help=(
            "Use the full WeightedDatasetMixture dataloader for sampling "
            "(safer but ~1000x slower at ~25 chunks/s due to per-sample "
            "standardization). Default: a hand-rolled equivalent that "
            "weight-allocates a budget per dataset, reads action+timestamp "
            "columns directly from parquet, and resamples to "
            "mixture.action_freq via scipy.interp1d -- or to each dataset's "
            "native fps when mixture.action_freq is None (mixed-frequency "
            "mode). Both paths respect the mixture's weights and the same "
            "fps convention."
        ),
    )
    p.add_argument(
        "--per-head-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize each chunk to [-1, 1] using its own "
            "(robot_type, control_mode) norm head's pooled min/max -- "
            "matches what the policy does at training time after PR #347. "
            "Pass --no-per-head-norm to fall back to the legacy global "
            "aggregate (only correct for single-head mixtures). The "
            "global path systematically under-spreads each head's "
            "distribution, so fit-time token-length analysis "
            "underestimates the actual policy-side truncation rate. "
            "Only supported on the default (manual sampler) path; "
            "passing --use-mixture-dataloader silently falls back to "
            "global normalization with a warning."
        ),
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: cap total-chunks at 50_000 and write to out_dir/pilot/.",
    )
    p.add_argument(
        "--skip-fit",
        action="store_true",
        help="Skip BPE fit (sampling pipeline only; useful for timing).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_mixture(mixture_json: Path) -> tuple[DatasetMixtureConfig, str]:
    """Resolve ``$ref`` includes, parse, return (config, sha256 of resolved JSON)."""
    tmp = resolve_refs_to_tempfile(mixture_json)
    try:
        mixture_cfg = draccus.parse(
            config_class=DatasetMixtureConfig,
            config_path=str(tmp),
            args=[],
        )
        content_hash = hashlib.sha256(tmp.read_bytes()).hexdigest()
    finally:
        tmp.unlink(missing_ok=True)
    return mixture_cfg, content_hash


def _resolve_native_action_key(
    repo_id: str | None, override_mapping: dict | None, available_keys: set[str] | None = None
) -> str:
    """Return the native parquet / stats column name for actions.

    OpenTau's standard name is ``"actions"`` (plural); most upstream LeRobot
    datasets use ``"action"`` (singular). ``DATA_FEATURES_NAME_MAPPING`` maps
    standard -> native; per-DatasetConfig overrides are upserted into that
    global at parse time.

    When the caller knows the dataset's actual stats / column keys, pass
    ``available_keys`` so we can return whichever of ``"action"`` /
    ``"actions"`` is actually present instead of guessing -- handles datasets
    that aren't registered in ``DATA_FEATURES_NAME_MAPPING`` and use either
    convention. Without ``available_keys`` we still fall back to ``"action"``
    (the more common upstream convention).
    """
    from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

    if override_mapping is not None and "actions" in override_mapping:
        return override_mapping["actions"]
    # TODO: dual-split repos (same repo_id, joint vs ee) share this repo_id key; thread
    # control_mode through `resolve_feature_mapping` for a precise per-mode action column.
    mapping = DATA_FEATURES_NAME_MAPPING.get(repo_id, {}) if repo_id else {}
    if "actions" in mapping:
        return mapping["actions"]
    # No registered mapping. If we know the available keys, prefer whichever
    # exists; otherwise default to "action".
    if available_keys is not None:
        for candidate in ("action", "actions"):
            if candidate in available_keys:
                return candidate
    return "action"


def _load_metadata_stats_and_info(
    item: tuple[int, Any],
) -> tuple[int, str, dict[str, Any], dict[str, np.ndarray] | None, str | None]:
    """Shared worker: load a dataset's ``LeRobotDatasetMetadata``, return
    ``(idx, repo_id, info_with_overrides, stats_or_None, err_or_None)``.

    ``info_with_overrides`` is a copy of ``meta.info`` with
    ``DatasetConfig.{robot_type,control_mode}`` overrides applied -- mirrors
    ``factory._apply_metadata_overrides`` so the caller can derive the same
    norm key the training policy does. The action stats are aggregated over the
    SELECTED episodes (``DatasetConfig.{episodes,excluded_episodes}``) so the
    codec range matches training-time normalization. On full construction
    failure, the returned ``info`` is empty.
    """
    idx, cfg = item
    repo_id = cfg.repo_id or "<no-repo-id>"
    try:
        from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata, aggregate_selected_stats

        meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        info = dict(getattr(meta, "info", {}) or {})
        if cfg.robot_type is not None:
            info["robot_type"] = cfg.robot_type
        if cfg.control_mode is not None:
            info["control_mode"] = cfg.control_mode
        # Fit the codec over the SELECTED-episode action range so it matches the
        # episodes the policy trains on (honors `episodes` + `excluded_episodes`).
        stats = aggregate_selected_stats(meta, cfg.episodes, cfg.excluded_episodes)
        key = _resolve_native_action_key(
            cfg.repo_id,
            cfg.data_features_name_mapping,
            available_keys=set(stats or []),
        )
        if not stats or key not in stats:
            return (
                idx,
                repo_id,
                info,
                None,
                f"key {key!r} missing from stats (keys={sorted(stats or [])})",
            )
        s = stats[key]
        return (
            idx,
            repo_id,
            info,
            {
                "min": np.asarray(s["min"], dtype=np.float64).ravel(),
                "max": np.asarray(s["max"], dtype=np.float64).ravel(),
            },
            None,
        )
    except Exception as e:  # noqa: BLE001
        return idx, repo_id, {}, None, f"{type(e).__name__}: {e}"


def _load_dataset_stats(
    item: tuple[int, Any],
) -> tuple[int, str, dict | None, str | None]:
    """Worker: read action min/max only (used by the legacy global-norm path)."""
    idx, repo_id, _info, stats, err = _load_metadata_stats_and_info(item)
    return idx, repo_id, stats, err


def _aggregate_stats_manual(
    mixture_cfg: DatasetMixtureConfig, action_dim: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
    """NaN-tolerant nanmin / nanmax of per-dataset action stats.

    Mirrors what ``DatasetMixtureMetadata`` does for ``actions`` -- pads each
    dataset's stats to ``action_dim`` (NaN sentinel for padded slots),
    aggregates with nanmin / nanmax, falls back to ``[-1, 1]`` for dims that
    are NaN across the entire mixture.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(
        "Loading action stats for %d datasets (workers=%d)",
        len(mixture_cfg.datasets),
        num_workers,
    )
    t0 = time.perf_counter()
    mins_padded: list[np.ndarray] = []
    maxs_padded: list[np.ndarray] = []
    per_dataset: dict[int, dict] = {}
    n_ok = 0
    n_fail = 0
    work = list(enumerate(mixture_cfg.datasets))
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_load_dataset_stats, item): item[0] for item in work}
        for fut in as_completed(futs):
            idx, repo_id, stats, err = fut.result()
            per_dataset[idx] = {"repo_id": repo_id, "ok": stats is not None, "error": err}
            if stats is None:
                n_fail += 1
                logger.warning("Dataset %d (%s): no stats -- %s", idx, repo_id, err)
                continue
            n_ok += 1
            native_dim = int(stats["min"].shape[0])
            per_dataset[idx]["native_action_dim"] = native_dim
            mn = np.full(action_dim, np.nan, dtype=np.float64)
            mx = np.full(action_dim, np.nan, dtype=np.float64)
            clip = min(action_dim, native_dim)
            mn[:clip] = stats["min"][:clip]
            mx[:clip] = stats["max"][:clip]
            mins_padded.append(mn)
            maxs_padded.append(mx)
    if not mins_padded:
        raise RuntimeError("No dataset stats loaded successfully.")
    # Surface --action-dim / mixture-native dim mismatches loudly: any dataset
    # whose native action dim exceeds ``action_dim`` will have its high dims
    # silently dropped from the BPE corpus *and* normalized with truncated
    # stats. That's only correct if production's ``max_action_dim`` matches.
    over_dim = [
        (i, info.get("native_action_dim", 0), info["repo_id"])
        for i, info in per_dataset.items()
        if info.get("native_action_dim", 0) > action_dim
    ]
    if over_dim:
        max_over = max(d for _, d, _ in over_dim)
        logger.warning(
            "%d/%d datasets have native action_dim > --action-dim=%d (max %d). "
            "Their extra dims will be silently dropped. Confirm this matches "
            "the production policy's max_action_dim (mismatch => the fitted "
            "BPE doesn't cover those dims at inference). Sample: %s",
            len(over_dim),
            n_ok,
            action_dim,
            max_over,
            [r for _, _, r in over_dim[:5]],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        action_min = np.nanmin(np.stack(mins_padded), axis=0)
        action_max = np.nanmax(np.stack(maxs_padded), axis=0)
    nan_dims = np.where(~np.isfinite(action_min) | ~np.isfinite(action_max))[0]
    if nan_dims.size > 0:
        logger.info(
            "Dims with no stats anywhere in mixture: %s -- using [-1, 1] fallback.",
            nan_dims.tolist(),
        )
        action_min[nan_dims] = -1.0
        action_max[nan_dims] = 1.0
    logger.info(
        "Stats aggregation done in %.1fs (%d ok, %d fail).",
        time.perf_counter() - t0,
        n_ok,
        n_fail,
    )
    return action_min, action_max, per_dataset


def _aggregate_stats_per_head(
    mixture_cfg: DatasetMixtureConfig, action_dim: int, num_workers: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, np.ndarray]]:
    """Aggregate per-``(robot_type, control_mode)`` action stats.

    Mirrors what ``DatasetMixtureMetadata._build_norm_heads`` does for the
    actions stat at training time: each dataset gets the pooled min/max of
    its norm head, so chunks normalized at fit time match what
    ``Normalize({"ACTION": NormalizationMode.MIN_MAX})`` produces from the
    same data at training time. The pooling for min/max is straight
    ``nanmin``/``nanmax`` across head members -- count-weighted aggregation
    (which ``aggregate_stats`` does in dataset_mixture) only matters for
    ``mean``/``std`` and is a no-op for ``min``/``max``.

    To match the production ``Normalize`` path exactly, per-dataset stats are
    zero-padded (not NaN-padded) to ``action_dim`` before pooling -- this
    mirrors ``pad_vector`` (zero-pad) applied in
    ``DatasetMixtureMetadata._to_standard_data_format``. Trailing slots
    therefore pool to ``min=max=0``, which makes the production
    ``(x - min) / (max - min + EPS) * 2 - 1`` evaluate to ``-1`` for the
    zero-padded action suffix at both fit and training time.

    Failure modes (matching training-time behaviour):

    - If any dataset's action stats fail to load, raise ``RuntimeError``.
      Training would also crash on such a dataset (``_to_standard_data_format``
      raises ``KeyError``), so silently producing a tokenizer that the policy
      will refuse to consume is sneaky. Drop the offending dataset from the
      mixture (or use ``--no-per-head-norm`` to fall back to the legacy
      global path) before retrying.
    - If ``mixture_cfg.require_non_empty_robot_type`` /
      ``require_non_empty_control_mode`` is set and any dataset still has
      an empty value after overrides, raise the same ``ValueError`` that
      ``datasets.factory._validate_metadata_requirements`` raises.

    Args:
        mixture_cfg: Mixture config.
        action_dim: Padded action dim (chunks are padded to this width
            before normalization, so the returned ``min``/``max`` arrays
            are sized ``(action_dim,)``).
        num_workers: ProcessPool size for parallel stats loading.

    Returns:
        ``(per_ds_min, per_ds_max, per_ds_key, per_head_stats)``:

        - ``per_ds_min[i]``/``per_ds_max[i]``: ``(action_dim,)`` float32
          arrays for dataset ``i`` (the i-th entry in
          ``mixture_cfg.datasets``). Datasets sharing a non-fallback
          ``(robot_type, control_mode)`` get identical pooled arrays.
        - ``per_ds_key[i]``: the norm key for dataset ``i`` (always a str
          after the all-stats-required guard above).
        - ``per_head_stats``: deduplicated head -> ``{"min": ..., "max": ...}``
          map (diagnostics + report).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = len(mixture_cfg.datasets)
    raw_min: list[np.ndarray | None] = [None] * n
    raw_max: list[np.ndarray | None] = [None] * n
    # Use list comprehensions (not `[{}] * n`) for mutable defaults so a future
    # `per_ds_info[idx]["override_X"] = ...` doesn't silently fan out across
    # all slots. The `[None] * n` / `["<no-repo-id>"] * n` patterns above are
    # safe because their element types are immutable.
    per_ds_info: list[dict[str, Any]] = [{} for _ in range(n)]
    per_ds_native_dim: list[int | None] = [None] * n
    per_ds_repo: list[str] = ["<no-repo-id>"] * n
    failures: list[tuple[int, str, str]] = []

    logger.info(
        "Loading per-(rt, cm) action stats for %d datasets (workers=%d)",
        n,
        num_workers,
    )
    t0 = time.perf_counter()
    work = list(enumerate(mixture_cfg.datasets))
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_load_metadata_stats_and_info, item): item[0] for item in work}
        for fut in as_completed(futs):
            idx, repo_id, info, stats, err = fut.result()
            per_ds_repo[idx] = repo_id
            per_ds_info[idx] = info
            if stats is None:
                failures.append((idx, repo_id, err or "<unknown>"))
                continue
            native_dim = int(stats["min"].shape[0])
            per_ds_native_dim[idx] = native_dim
            # Zero-pad to action_dim -- matches `pad_vector` in
            # `_to_standard_data_format`. Trailing slots get min=max=0, so the
            # production `(0 - 0) / (EPS) * 2 - 1 = -1` matches our fit-time
            # output bit-for-bit (in float64) for the padded suffix.
            mn = np.zeros(action_dim, dtype=np.float64)
            mx = np.zeros(action_dim, dtype=np.float64)
            clip = min(action_dim, native_dim)
            mn[:clip] = stats["min"][:clip]
            mx[:clip] = stats["max"][:clip]
            raw_min[idx] = mn
            raw_max[idx] = mx

    if failures:
        sample = ", ".join(f"{repo}: {err}" for _i, repo, err in failures[:5])
        raise RuntimeError(
            f"Per-head normalization requires all datasets' action stats to load, "
            f"but {len(failures)}/{n} failed. Training would also crash on these "
            f"datasets (`_to_standard_data_format` raises on missing stats). "
            f"Drop them from the mixture or fix their stats. Sample: {sample}"
        )

    # Match `_validate_metadata_requirements` in datasets.factory: if the
    # mixture demands non-empty robot_type / control_mode, surface that at
    # fit time so the operator doesn't burn 90s on a fit that the very next
    # training launch refuses to start.
    require_robot = bool(getattr(mixture_cfg, "require_non_empty_robot_type", False))
    require_control = bool(getattr(mixture_cfg, "require_non_empty_control_mode", False))
    if require_robot or require_control:
        bad: list[str] = []
        for i in range(n):
            info = per_ds_info[i]
            if require_robot and not (info.get("robot_type") or "").strip():
                bad.append(f"{per_ds_repo[i]}: robot_type is empty")
            if require_control and not (info.get("control_mode") or "").strip():
                bad.append(f"{per_ds_repo[i]}: control_mode is empty")
        if bad:
            raise ValueError(
                "DatasetMixtureConfig requires non-empty metadata fields, but the "
                f"following {len(bad)} datasets are missing values after overrides:\n  - "
                + "\n  - ".join(bad)
                + "\nSet `DatasetConfig.robot_type` / `DatasetConfig.control_mode` "
                "on the offending dataset(s) to provide an override."
            )

    # Derive norm keys (now that all stats loaded). Track fallback-fired
    # datasets and surface them like `_build_norm_heads` does -- otherwise the
    # operator silently gets singleton-per-dataset heads instead of pooled
    # ones, which is almost never what they want.
    per_ds_key: list[str] = [""] * n
    fallback_datasets: list[str] = []
    for i in range(n):
        info = per_ds_info[i]
        key, fallback_fired = compute_norm_key(
            info.get("robot_type"),
            info.get("control_mode"),
            per_ds_repo[i],
        )
        per_ds_key[i] = key
        if fallback_fired:
            fallback_datasets.append(per_ds_repo[i])
    if fallback_datasets:
        shown = fallback_datasets[:10]
        suffix = f", ... and {len(fallback_datasets) - 10} more" if len(fallback_datasets) > 10 else ""
        logger.warning(
            "%d/%d datasets lack non-empty robot_type / control_mode and were "
            "given a per-dataset fallback norm head (one singleton head each). "
            "Set `DatasetConfig.robot_type` / `DatasetConfig.control_mode` to "
            "pool them into shared heads. Affected: %s%s",
            len(fallback_datasets),
            n,
            shown,
            suffix,
        )
        # Tighter signal for the specific divergence the deferred fallback-name
        # dedup finding flagged: when a fallback-keyed repo_id appears more
        # than once in the mixture, fit-time pools them under one shared key
        # (since `compute_norm_key` returns the same string both times) while
        # training keeps them as separate singleton heads (since
        # `_make_dataset_names` deduplicates with `#N` suffixes). Flag the
        # divergence explicitly so the operator can fix it before it ships.
        from collections import Counter

        fallback_counts = Counter(fallback_datasets)
        duplicates_in_fallback = {k: v for k, v in fallback_counts.items() if v > 1}
        if duplicates_in_fallback:
            dup_preview = dict(list(duplicates_in_fallback.items())[:10])
            logger.warning(
                "%d fallback-keyed repo_id values appear in the mixture more "
                "than once. Fit-time POOLS these under one shared head per "
                "repo_id, but training (via `_make_dataset_names`'s `#N` dedup) "
                "keeps them as separate singleton heads -- the fit-time "
                "normalization will diverge from training. Set "
                "`DatasetConfig.robot_type` / `DatasetConfig.control_mode` on "
                "each entry, or deduplicate the mixture, to close the gap. "
                "Duplicates (repo_id -> count): %s",
                len(duplicates_in_fallback),
                dup_preview,
            )

    # Restore the over-dim warning from the global path: datasets whose
    # native action dim exceeds --action-dim will have high dims silently
    # dropped. The mixture won't include them at training either if
    # max_action_dim < native, but the warning is still load-bearing because
    # the fit produces a tokenizer with no token coverage for the dropped
    # dims at inference.
    over_dim = [
        (i, per_ds_native_dim[i], per_ds_repo[i])
        for i in range(n)
        if per_ds_native_dim[i] is not None and per_ds_native_dim[i] > action_dim
    ]
    if over_dim:
        max_over = max(d for _, d, _ in over_dim if d is not None)
        logger.warning(
            "%d/%d datasets have native action_dim > --action-dim=%d (max %d). "
            "Their extra dims will be silently dropped. Confirm this matches "
            "the production policy's max_action_dim (mismatch => the fitted "
            "BPE doesn't cover those dims at inference). Sample: %s",
            len(over_dim),
            n,
            action_dim,
            max_over,
            [r for _, _, r in over_dim[:5]],
        )

    per_ds_min, per_ds_max, per_head_stats = _pool_per_head_stats(raw_min, raw_max, per_ds_key, action_dim)
    logger.info(
        "Per-(rt, cm) stats aggregated in %.1fs: %d heads across %d datasets.",
        time.perf_counter() - t0,
        len(per_head_stats),
        n,
    )
    return per_ds_min, per_ds_max, per_ds_key, per_head_stats


def _pool_per_head_stats(
    raw_min: list[np.ndarray | None],
    raw_max: list[np.ndarray | None],
    per_ds_key: list[str | None],
    action_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Pure pooling: aggregate per-dataset raw (min, max) into per-head stats,
    then broadcast back per-dataset so each dataset gets the stats of its
    head. Fallback keys (a dataset's repo_id used in lieu of a real
    ``(rt, cm)`` pair) are singletons by construction, so they get their
    own stats verbatim.

    Pooling for ``min``/``max`` is ``nanmin``/``nanmax`` across the head's
    members; for those two fields ``aggregate_stats``'s count-weighted
    aggregation reduces to the unweighted nanmin/nanmax (the weights only
    matter for ``mean``/``std``). ``aggregate_stats`` masks ``±Inf`` to
    ``NaN`` first so they don't poison the reduction; we mirror that.

    Preconditions (enforced by ``_aggregate_stats_per_head``): every entry
    in ``per_ds_key`` is a non-None string, and every entry in
    ``raw_min``/``raw_max`` is a non-None ``(action_dim,)`` array.

    Returns ``(per_ds_min, per_ds_max, per_head_stats)`` where the third
    element is the deduplicated per-head stats map
    ``{key: {"min": ..., "max": ...}}``.
    """
    n = len(per_ds_key)
    from collections import defaultdict

    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(per_ds_key):
        # Invariant from `_aggregate_stats_per_head`: every per_ds_key is a
        # non-empty string and every raw_min/raw_max entry is set.
        assert k is not None and raw_min[i] is not None and raw_max[i] is not None, (
            f"_pool_per_head_stats: row {i} has stale None (key={k!r}, "
            f"raw_min={'None' if raw_min[i] is None else 'set'}). "
            "Callers must populate every row before pooling."
        )
        key_to_indices[k].append(i)

    per_head_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, indices in key_to_indices.items():
        stacked_min = np.stack([raw_min[i] for i in indices])
        stacked_max = np.stack([raw_max[i] for i in indices])
        # Mirror `aggregate_stats` (compute_stats.py:350): mask `±Inf` to
        # `NaN` before reduction so a single Inf entry doesn't poison the
        # pool (-Inf would still survive nanmin; +Inf would still survive
        # nanmax). NaN is then skipped by nanmin/nanmax.
        stacked_min = np.where(np.isfinite(stacked_min), stacked_min, np.nan)
        stacked_max = np.where(np.isfinite(stacked_max), stacked_max, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mn = np.nanmin(stacked_min, axis=0)
            mx = np.nanmax(stacked_max, axis=0)
        # Defensive: a head where *every* member had Inf in some slot would
        # leave NaN here. Fill with [-1, 1] so downstream `Normalize` doesn't
        # see NaN. (Should not happen on healthy data; assert and continue.)
        nan_dims = np.where(~np.isfinite(mn) | ~np.isfinite(mx))[0]
        if nan_dims.size > 0:
            logger.warning(
                "Norm head %r has dims with no finite stats after Inf-mask "
                "+ nanmin/nanmax: %s. Filling with [-1, 1]; expect a model "
                "that ignores those output dims.",
                key,
                nan_dims.tolist(),
            )
            mn[nan_dims] = -1.0
            mx[nan_dims] = 1.0
        per_head_stats[key] = {"min": mn, "max": mx}

    per_ds_min: list[np.ndarray] = []
    per_ds_max: list[np.ndarray] = []
    for i in range(n):
        head = per_head_stats[per_ds_key[i]]
        per_ds_min.append(head["min"].astype(np.float32))
        per_ds_max.append(head["max"].astype(np.float32))
    return per_ds_min, per_ds_max, per_head_stats


def _normalize_chunks_per_head(
    stacked: np.ndarray,
    per_dataset_chunks: list[int],
    per_ds_min: list[np.ndarray],
    per_ds_max: list[np.ndarray],
) -> np.ndarray:
    """Apply per-dataset min/max to chunks in dataset-config order.

    ``_sample_via_manual`` concatenates per-dataset chunk buckets in
    mixture-config order, so a cumulative sum over ``per_dataset_chunks``
    tells us which slice of ``stacked`` belongs to each dataset. Each slice
    gets normalized with its own ``(min, max)``; datasets that share a norm
    head get identical stats by construction (see ``_aggregate_stats_per_head``).
    """
    total = int(sum(per_dataset_chunks))
    # Defensive: a future refactor to `_sample_via_manual` that changes the
    # concatenation order or drops chunks must not silently corrupt the BPE
    # corpus by misaligning the per-dataset normalization windows.
    assert total == stacked.shape[0], (
        f"_normalize_chunks_per_head: per_dataset_chunks sums to {total} but "
        f"stacked has {stacked.shape[0]} rows. Sampler/normalizer drifted "
        "out of sync; review `_sample_via_manual` concatenation order."
    )
    assert len(per_dataset_chunks) == len(per_ds_min) == len(per_ds_max), (
        "per_dataset_chunks, per_ds_min, per_ds_max must be 1:1; got "
        f"{len(per_dataset_chunks)} / {len(per_ds_min)} / {len(per_ds_max)}."
    )
    out = np.zeros_like(stacked)
    offset = 0
    for i, count in enumerate(per_dataset_chunks):
        if count <= 0:
            continue
        out[offset : offset + count] = _normalize_chunks(
            stacked[offset : offset + count], per_ds_min[i], per_ds_max[i]
        )
        offset += count
    return out


def _compute_budgets_weighted(mixture_cfg: DatasetMixtureConfig, total_chunks: int) -> list[int]:
    """Pure weight-proportional budget per dataset (no clamps).

    The HierarchicalSampler used at training samples each chunk by weight, so
    expected per-dataset chunk count is ``total * w_i / sum(w)``. This function
    materializes that as discrete budgets we can dispatch in parallel.
    """
    n = len(mixture_cfg.datasets)
    weights = mixture_cfg.weights if mixture_cfg.weights is not None else [1.0] * n
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(n, dtype=np.float64)
    budgets = np.round(total_chunks * weights / weights.sum()).astype(int)
    return budgets.tolist()


def _sample_chunks_for_dataset_manual(
    args: tuple[int, Any, int, int, float | None, int],
) -> tuple[int, list[np.ndarray], str | None]:
    """Worker: sample ``n_chunks`` action chunks of shape ``(chunk_size, native_dim)``
    resampled to ``action_freq`` via ``scipy.interpolate.interp1d``.

    Reads raw parquet (action + timestamp) per episode -- no LeRobotDataset
    overhead. The interp matches LeRobot's vector_resample_strategy="nearest"
    closely enough for tokenizer fitting (``kind="linear"`` here is actually
    slightly more accurate; we don't need the strict step-resample behavior).

    When ``action_freq is None`` (mixed-frequency mixture), each dataset is
    sampled at its own native fps -- mirrors ``factory.resolve_delta_timestamps``
    substituting ``ds_meta.fps`` so the chunk is ``chunk_size`` consecutive
    native frames and the interp lands exactly on native frame boundaries.
    """
    idx, cfg, n_chunks, chunk_size, action_freq, seed = args
    if n_chunks <= 0:
        return idx, [], None
    try:
        import pyarrow.parquet as pq
        from scipy.interpolate import interp1d

        from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        native_fps = float(meta.info["fps"])
        if action_freq is None:
            action_freq = native_fps
        target_dt = 1.0 / action_freq
        chunk_duration = (chunk_size - 1) * target_dt  # seconds spanned by a chunk
        # Need at least ceil(chunk_duration * native_fps) + 1 native frames per chunk.
        min_native_frames = int(np.ceil(chunk_duration * native_fps)) + 1

        eligible_eps = (
            list(meta.episodes.keys())
            if cfg.episodes is None
            else [ep for ep in cfg.episodes if ep in meta.episodes]
        )
        ep_max_start: list[tuple[int, int]] = []
        for ep in eligible_eps:
            length = int(meta.episodes[ep].get("length", 0))
            valid = max(0, length - min_native_frames + 1)
            if valid > 0:
                ep_max_start.append((ep, valid))
        if not ep_max_start:
            return idx, [], f"no episode has length >= {min_native_frames} native frames"

        rng = np.random.default_rng(seed)
        ep_ids = np.array([e for e, _ in ep_max_start])
        ep_weights = np.array([v for _, v in ep_max_start], dtype=np.float64)
        probs = ep_weights / ep_weights.sum()
        chosen_eps = rng.choice(ep_ids, size=n_chunks, p=probs)
        valid_by_ep = dict(ep_max_start)
        from collections import defaultdict as _dd

        ep_to_starts: dict[int, list[int]] = _dd(list)
        for ep in chosen_eps:
            ep_int = int(ep)
            start = int(rng.integers(0, valid_by_ep[ep_int]))
            ep_to_starts[ep_int].append(start)

        # Peek at the first episode's parquet schema so we can pass real column
        # names to the resolver -- handles datasets with no registered mapping
        # that use either ``"action"`` or ``"actions"``.
        first_ep = next(iter(ep_to_starts))
        first_rel = meta.get_data_file_path(ep_index=first_ep)
        first_full = meta.root / first_rel
        if not first_full.exists():
            meta.pull_from_repo(allow_patterns=str(first_rel))
        available_cols = set(pq.read_schema(first_full).names)
        action_col = _resolve_native_action_key(
            cfg.repo_id, cfg.data_features_name_mapping, available_keys=available_cols
        )
        chunks: list[np.ndarray] = []
        for ep, starts in ep_to_starts.items():
            rel = meta.get_data_file_path(ep_index=ep)
            full = meta.root / rel
            if not full.exists():
                meta.pull_from_repo(allow_patterns=str(rel))
            try:
                table = pq.read_table(full, columns=[action_col, "timestamp"])
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Failed reading {full}: {e}") from e
            try:
                native_actions = np.asarray(table.column(action_col).to_pylist(), dtype=np.float32)
                native_timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise RuntimeError(f"{full}: non-uniform column shape: {e}") from e
            if native_actions.ndim != 2:
                raise RuntimeError(f"{full}: {action_col!r} ndim={native_actions.ndim}, expected 2")
            # Interpolator built once per parquet -- amortizes across all starts in this episode.
            interp = interp1d(
                native_timestamps,
                native_actions,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=(native_actions[0], native_actions[-1]),
                assume_sorted=True,
            )
            for s in starts:
                t0 = native_timestamps[s]
                target_ts = t0 + np.arange(chunk_size) * target_dt
                # Clamp to native range to avoid extrapolation surprises.
                target_ts = np.clip(target_ts, native_timestamps[0], native_timestamps[-1])
                chunks.append(interp(target_ts).astype(np.float32))
        return idx, chunks, None
    except Exception as e:  # noqa: BLE001
        return idx, [], f"{type(e).__name__}: {e}"


def _sample_via_manual(
    mixture_cfg: DatasetMixtureConfig,
    total_chunks: int,
    chunk_size: int,
    action_freq: float | None,
    seed: int,
    num_workers: int,
) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    """Parallel per-dataset sampling with FPS resampling. Returns
    ``(chunks, per_dataset_counts, errors_by_idx)``."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    budgets = _compute_budgets_weighted(mixture_cfg, total_chunks)
    logger.info(
        "Manual sampler budgets: total=%d (target=%d), min=%d, max=%d",
        sum(budgets),
        total_chunks,
        min(budgets),
        max(budgets),
    )
    work = [
        (i, cfg, budgets[i], chunk_size, action_freq, seed + i) for i, cfg in enumerate(mixture_cfg.datasets)
    ]
    # Collect per-dataset chunks into a fixed-index slot so the corpus we feed
    # into the BPE trainer is in mixture-config order regardless of which
    # workers finish first. BPE tie-breaks on equal pair frequencies depend on
    # first-encounter order in the corpus, so completion-order extends would
    # make two seeded runs diverge (cf. CLAUDE.md rule 3).
    n_datasets = len(mixture_cfg.datasets)
    per_ds_chunks: list[list[np.ndarray]] = [[] for _ in range(n_datasets)]
    per_ds_counts = [0] * n_datasets
    errors: dict[int, str] = {}
    t0 = time.perf_counter()
    last_log = t0
    n_done = 0
    total_so_far = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_sample_chunks_for_dataset_manual, w): w[0] for w in work}
        for fut in as_completed(futs):
            idx, ds_chunks, err = fut.result()
            if err:
                errors[idx] = err
                logger.warning("Dataset %d: %s", idx, err)
            per_ds_chunks[idx] = ds_chunks
            per_ds_counts[idx] = len(ds_chunks)
            n_done += 1
            total_so_far += len(ds_chunks)
            now = time.perf_counter()
            if now - last_log > 10.0 or n_done == len(work):
                logger.info(
                    "Manual sample: %d/%d datasets done, %d chunks so far",
                    n_done,
                    len(work),
                    total_so_far,
                )
                last_log = now
    chunks: list[np.ndarray] = [c for bucket in per_ds_chunks for c in bucket]
    logger.info(
        "Manual sampling done in %.1fs: %d chunks, %d ok, %d errors",
        time.perf_counter() - t0,
        len(chunks),
        len(work) - len(errors),
        len(errors),
    )
    return chunks, per_ds_counts, errors


def _build_train_cfg(
    mixture_cfg: DatasetMixtureConfig,
    chunk_size: int,
    action_dim: int,
    max_state_dim: int,
    dataloader_batch_size: int,
    num_workers: int,
) -> Any:
    """Build a minimal ``TrainPipelineConfig`` good enough for
    ``make_dataset_mixture``.

    We use a ``SimpleNamespace`` policy stub: ``resolve_delta_timestamps`` only
    reads ``policy.action_delta_indices`` (for action resampling) and
    ``policy.history_interval`` via ``getattr`` (for state/camera history).
    ``LeRobotDataset`` itself reads ``cfg.policy`` only inside ``ValueConfig``
    branches and a single assertion that ``cfg.action_chunk == chunk_size``,
    so a stub suffices. ``num_cams=0`` short-circuits the camera-loading path
    in ``LeRobotDataset._standardize_images`` (verified at
    ``lerobot_dataset.py:1868``), which is what makes this script CPU-cheap.
    """
    import dataclasses

    from opentau.configs.train import TrainPipelineConfig

    # DatasetMixtureConfig.val_split_ratio defaults to 0.05, and a per-dataset
    # ``DatasetConfig.val_split_ratio`` overrides it -- if either is non-zero,
    # make_dataset would return a (train, val) tuple per dataset and the assert
    # in ``_build_mixture_parallel`` would (correctly) reject it. Zero out BOTH
    # the mixture default and every per-dataset override so the parallel mixture
    # build always returns single train datasets. Rebuild the dataset configs
    # (``dataclasses.replace`` shares the ``datasets`` list reference) so the
    # caller's parsed config is left untouched.
    datasets_for_fit = [dataclasses.replace(dc, val_split_ratio=0.0) for dc in mixture_cfg.datasets]
    # We only need action chunks for the tokenizer fit. Override mixture-side
    # knobs that would otherwise force per-sample state-history loads and
    # augmentation rolls (none of which affect the action column). Use
    # ``dataclasses.replace`` so the caller's parsed config is unaffected and
    # this function can be called more than once on the same input.
    mixture_cfg_for_fit = dataclasses.replace(
        mixture_cfg,
        n_obs_history=None,
        history_state_drop_prob=0.0,
        subgoal_drop_prob=1.0,  # drop all subgoals -- we don't read them
        subgoal_end_of_segment_prob=0.0,
        response_drop_prob=1.0,  # drop all responses
        metadata_drop_all_prob=1.0,  # drop all metadata
        metadata_drop_each_prob=0.0,
        val_split_ratio=0.0,
        datasets=datasets_for_fit,
    )
    fake_policy = SimpleNamespace(
        action_delta_indices=list(range(chunk_size)),
        history_interval=1,
        n_obs_steps=1,  # matches n_obs_history=None
    )
    cfg = TrainPipelineConfig(
        dataset_mixture=mixture_cfg_for_fit,
        policy=fake_policy,
        batch_size=dataloader_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        gradient_accumulation_steps=1,
        num_workers=num_workers,
        action_chunk=chunk_size,
        num_cams=0,
        max_action_dim=action_dim,
        max_state_dim=max_state_dim,
        steps=1,
        save_checkpoint=False,
        use_policy_training_preset=False,
        # output_dir is auto-set in __post_init__; we never write to it.
    )
    return cfg


def _build_mixture_parallel(train_cfg: Any, num_workers: int) -> Any:
    """Drop-in for ``make_dataset_mixture(cfg)`` with thread-parallel per-dataset construction.

    ``make_dataset_mixture`` constructs each ``LeRobotDataset`` sequentially in a
    Python for-loop. With ~400 datasets and per-dataset I/O of several seconds
    (parquet scan + timestamp check), the sequential build can take 60+ min.
    LeRobotDataset init is I/O-bound (pyarrow + HF datasets cache builds release
    the GIL), so threading wins here even under the GIL. The resulting
    ``WeightedDatasetMixture`` is byte-identical to the upstream one because we
    preserve mixture-config ordering when collecting futures.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from opentau.datasets.dataset_mixture import WeightedDatasetMixture
    from opentau.datasets.factory import _resolve_weights, _validate_metadata_requirements, make_dataset

    # ``_build_train_cfg`` explicitly overrides ``val_split_ratio`` to 0 (the
    # field defaults to 0.05, so we cannot rely on the default). With it forced
    # to 0, ``make_dataset`` only ever returns a single train dataset.
    # Asserting that keeps the parallel-build path simpler than the upstream
    # train/val branching factory.
    assert getattr(train_cfg.dataset_mixture, "val_split_ratio", 0.0) == 0.0, (
        "fit_fast_tokenizer doesn't support val splits"
    )

    dataset_cfgs = train_cfg.dataset_mixture.datasets
    n = len(dataset_cfgs)
    train_datasets: list[Any] = [None] * n
    n_done = 0
    last_log = time.perf_counter()

    def _build_one(idx: int) -> tuple[int, Any]:
        return idx, make_dataset(dataset_cfgs[idx], train_cfg, return_advantage_input=False)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_build_one, i): i for i in range(n)}
        for fut in as_completed(futs):
            idx, res = fut.result()
            if isinstance(res, tuple):
                raise RuntimeError(
                    f"Dataset {idx} ({getattr(dataset_cfgs[idx], 'repo_id', '?')}): "
                    "make_dataset returned a (train, val) tuple but val_split_ratio "
                    "should be 0. Did the train_cfg get mutated?"
                )
            train_datasets[idx] = res
            n_done += 1
            now = time.perf_counter()
            if now - last_log > 10.0 or n_done == n:
                logger.info("Mixture build: %d/%d datasets ready", n_done, n)
                last_log = now

    _validate_metadata_requirements(train_cfg, train_datasets, label="train")
    train_weights = _resolve_weights(train_cfg.dataset_mixture.weights, train_datasets, label="train")
    return WeightedDatasetMixture(
        train_cfg,
        train_datasets,
        train_weights,
        train_cfg.dataset_mixture.action_freq,
    )


def _extract_action_stats(mixture_meta: Any, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Pull a single global ``action`` min/max across the whole mixture.

    Per-dataset normalization (the production training path) keeps separate
    stats per source, but the BPE tokenizer fits one codec over a shared
    action range — call ``mixture_meta.aggregated_action_stats()`` to compute
    that on demand. Output is already padded to ``max_action_dim`` and uses
    NaN-tolerant aggregation under the hood.

    Dims that are ``NaN`` across the whole mixture fall back to ``[-1, 1]`` so
    the normalization step never divides by zero.
    """
    try:
        action_stats = mixture_meta.aggregated_action_stats()
    except (AttributeError, ValueError) as e:
        raise RuntimeError(
            "Mixture meta does not expose `aggregated_action_stats()` or has no "
            "'actions' contributor. Ensure the mixture was built from "
            "`WeightedDatasetMixture` (which produces `DatasetMixtureMetadata`)."
        ) from e
    action_min = np.asarray(action_stats["min"], dtype=np.float64).ravel()
    action_max = np.asarray(action_stats["max"], dtype=np.float64).ravel()
    if action_min.shape[0] < action_dim:
        action_min = np.pad(
            action_min,
            (0, action_dim - action_min.shape[0]),
            constant_values=np.nan,
        )
        action_max = np.pad(
            action_max,
            (0, action_dim - action_max.shape[0]),
            constant_values=np.nan,
        )
    elif action_min.shape[0] > action_dim:
        action_min = action_min[:action_dim]
        action_max = action_max[:action_dim]
    nan_dims = np.where(~np.isfinite(action_min) | ~np.isfinite(action_max))[0]
    if nan_dims.size > 0:
        logger.info(
            "Dims with no stats anywhere in mixture: %s -- using [-1, 1] fallback.",
            nan_dims.tolist(),
        )
        action_min[nan_dims] = -1.0
        action_max[nan_dims] = 1.0
    logger.info("Mixture-aggregated action_min: %s", np.round(action_min, 3).tolist())
    logger.info("Mixture-aggregated action_max: %s", np.round(action_max, 3).tolist())
    return action_min, action_max


_NORMALIZE_EPS = 1e-8  # matches ``opentau.policies.normalize.EPS``.


def _normalize_chunks(chunks: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    """Min-max-normalize a stacked chunk tensor ``(N, T, D)`` to ``[-1, 1]``.

    Mirrors the formula in ``Normalize({"ACTION": NormalizationMode.MIN_MAX})``
    from ``opentau.policies.normalize``:
    ``out = (x - min) / (max - min + EPS) * 2 - 1`` (no clip).
    NOTE: production ``Normalize`` runs the whole expression in float32, while
    we compute in float64 here and cast back at the end -- low float32 bits
    will differ from training. The values agree to ~1e-7 absolute error, which
    is well below the DCT scale used downstream, so the BPE corpus is
    indistinguishable in practice; just don't claim bit-equality.
    The final ``nan_to_num`` is defensive only -- production neither sanitizes
    NaN in ``Normalize`` nor in ``prepare_discrete_actions``, so a NaN input
    would actually crash the upstream DCT. We sanitize so a single bad chunk
    (e.g. a recording with a corrupted action sample) doesn't tank an hour-long
    fit; for non-NaN inputs this is a no-op.
    """
    span_with_eps = (action_max - action_min) + _NORMALIZE_EPS
    norm = (chunks.astype(np.float64) - action_min) / span_with_eps * 2.0 - 1.0
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    return norm.astype(np.float32)


def _drain_mixture(
    mixture_meta: Any,
    dataloader: Any,
    total_chunks: int,
    action_dim: int,
) -> tuple[np.ndarray, int]:
    """Iterate the WeightedDatasetMixture dataloader, normalize, collect chunks.

    Returns ``(stacked_chunks_array (N, T, D), n_batches_consumed)``.

    Note: unlike the manual path's ``_sample_via_manual``, this branch does
    NOT log a warning when datasets have native_action_dim > action_dim --
    the mixture's standardization pipeline already pads to
    ``cfg.max_action_dim`` (and would raise on truncation, since
    ``DatasetMixtureMetadata.pad_vector`` doesn't truncate). So an
    over-dim mismatch surfaces as a build-time failure, not silent
    truncation as in the manual path.
    """
    import torch

    action_min, action_max = _extract_action_stats(mixture_meta, action_dim)

    collected: list[np.ndarray] = []
    n_collected = 0
    n_batches = 0
    t0 = time.perf_counter()
    last_log = t0

    for batch in dataloader:
        actions = batch["actions"]
        # ``.numpy()`` doesn't support bfloat16, so cast to float32 first.
        actions = (
            actions.detach().to(dtype=torch.float32, device="cpu").numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions, dtype=np.float32)
        )
        # Expect shape (B, T, D); some pipelines may yield (B, T, D_native) padded
        # later -- the WeightedDatasetMixture's standardization pipeline pads to
        # max_action_dim, so D == action_dim here.
        if actions.ndim != 3:
            raise RuntimeError(f"Unexpected actions batch ndim={actions.ndim}, shape={actions.shape}")
        normalized = _normalize_chunks(actions, action_min, action_max)
        take = min(normalized.shape[0], total_chunks - n_collected)
        collected.append(normalized[:take])
        n_collected += take
        n_batches += 1
        if n_collected >= total_chunks:
            break
        now = time.perf_counter()
        if now - last_log > 10.0:
            rate = n_collected / max(1e-6, now - t0)
            logger.info(
                "Draining: %d / %d chunks (%.0f chunks/s, %d batches)",
                n_collected,
                total_chunks,
                rate,
                n_batches,
            )
            last_log = now

    elapsed = time.perf_counter() - t0
    stacked = np.concatenate(collected, axis=0) if collected else np.empty((0,))
    logger.info(
        "Drain done in %.1fs: %d chunks across %d batches (rate %.0f chunks/s)",
        elapsed,
        stacked.shape[0],
        n_batches,
        stacked.shape[0] / max(1e-6, elapsed),
    )
    return stacked, n_batches


def _fit_tokenizer(
    chunks: np.ndarray,
    vocab_size: int,
    scale: float,
    chunk_size: int,
    action_dim: int,
    max_token_length: int,
) -> tuple[Any, float]:
    """Fit a fresh ``UniversalActionProcessor``, exposing the BpeTrainer's
    ``max_token_length`` knob so we can cap runaway zero-run merges.

    Mirrors ``UniversalActionProcessor.fit`` from
    ``physical-intelligence/fast/processing_action_tokenizer.py`` pinned at
    commit ``UPSTREAM_PINNED_SHA`` (see module-level constant), except for the
    ``max_token_length`` value -- the upstream hard-codes 10000, which on
    heavily zero-padded action data wastes the BPE's entire merge budget on
    one runaway zero-run token (empirically: 1855 merges, one each at lengths
    2..1320, all zero-pad). When the upstream HF model is updated, bump the
    pinned SHA and re-port the body below.
    """
    from scipy.fft import dct
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.trainers import BpeTrainer
    from transformers import AutoProcessor, PreTrainedTokenizerFast

    # Resolve the UniversalActionProcessor class via trust_remote_code.
    upstream = AutoProcessor.from_pretrained(UPSTREAM_REPO_ID, trust_remote_code=True)
    cls = type(upstream)

    chunk_list = [chunks[i] for i in range(chunks.shape[0])]
    logger.info(
        "Fitting %s on %d chunks (vocab=%d, scale=%g, chunk_size=%d, action_dim=%d, max_token_length=%d)",
        cls.__name__,
        len(chunk_list),
        vocab_size,
        scale,
        chunk_size,
        action_dim,
        max_token_length,
    )
    t0 = time.perf_counter()

    # --- Replicate UniversalActionProcessor.fit body, parametrised. ---
    dct_tokens = [dct(a, axis=0, norm="ortho").flatten() for a in chunk_list]
    quantized = np.around(np.concatenate(dct_tokens) * scale)
    max_token = int(quantized.max())
    min_token = int(quantized.min())
    min_vocab_size = max_token - min_token
    if min_vocab_size > vocab_size:
        raise ValueError(
            f"Vocab size {vocab_size} is too small for the range of tokens "
            f"{min_vocab_size}; bump --vocab-size."
        )
    if min_vocab_size + _MIN_MERGE_HEADROOM > vocab_size:
        logger.warning(
            "Initial alphabet size %d is close to vocab size %d -- "
            "consider increasing --vocab-size for more merge headroom.",
            min_vocab_size,
            vocab_size,
        )

    def _token_iter():
        for tokens in dct_tokens:
            rounded = np.around(tokens * scale) - min_token
            rounded = rounded.astype(int)
            yield "".join(map(chr, rounded))

    alphabet = [chr(i) for i in range(max_token - min_token + 1)]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=alphabet,
        max_token_length=max_token_length,
    )
    bpe = ByteLevelBPETokenizer()
    bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)
    processor = cls(
        PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
        scale=scale,
        vocab_size=vocab_size,
        min_token=min_token,
        time_horizon=chunk_size,
        action_dim=action_dim,
    )
    # --- end replicated body ---

    elapsed = time.perf_counter() - t0
    logger.info(
        "BPE fit done in %.1fs (min_token=%s, vocab_size=%s, max_token_length=%d)",
        elapsed,
        getattr(processor, "min_token", "?"),
        getattr(processor, "vocab_size", "?"),
        max_token_length,
    )
    return processor, elapsed


def _find_upstream_source() -> Path:
    """Snapshot-download the remote-code source file at the pinned revision.

    Passing ``revision=UPSTREAM_PINNED_SHA`` makes the constant load-bearing:
    ``AutoProcessor.from_pretrained(out_dir, trust_remote_code=True)`` later
    executes the copied file, so it must match the body that ``_fit_tokenizer``
    was ported from. Without the pin, ``snapshot_download`` defaults to ``main``
    and the copied source can silently diverge from the ported fit logic.
    """
    from huggingface_hub import snapshot_download

    cache_root = snapshot_download(
        repo_id=UPSTREAM_REPO_ID,
        revision=UPSTREAM_PINNED_SHA,
        allow_patterns=UPSTREAM_SOURCE_FILE,
    )
    src = Path(cache_root) / UPSTREAM_SOURCE_FILE
    if not src.exists():
        raise FileNotFoundError(f"{UPSTREAM_SOURCE_FILE} not in {cache_root}")
    return src


def _save_processor(processor: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(out_dir))
    shutil.copy(_find_upstream_source(), out_dir / UPSTREAM_SOURCE_FILE)
    logger.info("Saved tokenizer + remote-code source to %s", out_dir)


def _verify_roundtrip(out_dir: Path, sample_chunks: np.ndarray) -> dict[str, float]:
    """Reload via ``AutoProcessor.from_pretrained`` and round-trip encode/decode."""
    from transformers import AutoProcessor

    rp = AutoProcessor.from_pretrained(str(out_dir), trust_remote_code=True)
    arr = sample_chunks.astype(np.float32)  # (N, T, D)
    tokens = rp(arr)
    decoded = rp.decode(tokens, time_horizon=arr.shape[1], action_dim=arr.shape[2])
    mse = float(np.mean((arr - decoded) ** 2))
    avg_tok_len = float(np.mean([len(t) for t in tokens]))
    logger.info(
        "Round-trip MSE=%.6f, avg token length=%.1f over %d held-out chunks",
        mse,
        avg_tok_len,
        arr.shape[0],
    )
    return {"roundtrip_mse": mse, "avg_token_length": avg_tok_len}


def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level)

    # The --use-mixture-dataloader path uses ``_extract_action_stats(mixture.meta, ...)``
    # which returns global aggregates; threading per-head normalization through
    # the dataloader-drained chunks would need per-sample dataset_index from
    # the batch and isn't done yet. Warn-and-fall-back so existing invocations
    # that pass `--use-mixture-dataloader` without explicitly setting
    # `--per-head-norm` keep working; the per-head default still applies on
    # the manual sampler path (which is what most fits use).
    if args.use_mixture_dataloader and args.per_head_norm:
        logger.warning(
            "--per-head-norm is not yet implemented on the --use-mixture-dataloader "
            "path; falling back to global aggregated_action_stats() for this run. "
            "Drop --use-mixture-dataloader to use the default manual sampler "
            "(which does support per-head), or pass --no-per-head-norm explicitly "
            "to silence this warning."
        )
        args.per_head_norm = False

    out_dir = args.out_dir
    if args.pilot:
        out_dir = out_dir / "pilot"
        if args.total_chunks > 50_000:
            logger.info(
                "Pilot mode: lowering --total-chunks from %d to 50_000.",
                args.total_chunks,
            )
            args.total_chunks = 50_000
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(args.seed)

    # --- Phase 1: parse mixture ---
    logger.info("Parsing mixture JSON: %s", args.mixture_json)
    mixture_cfg, mixture_hash = _parse_mixture(args.mixture_json)
    logger.info(
        "Mixture has %d datasets (hash=%s); action_freq=%s",
        len(mixture_cfg.datasets),
        mixture_hash[:12],
        "None (mixed-frequency, per-dataset native fps)"
        if mixture_cfg.action_freq is None
        else f"{mixture_cfg.action_freq:g}",
    )

    # --- Phase 2: build mixture & gather chunks ---
    build_time = 0.0
    sample_errors: dict[int, str] = {}
    per_dataset_chunks: list[int] = [0] * len(mixture_cfg.datasets)
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    per_ds_norm_keys: list[str] | None = None
    per_head_stats_report: dict[str, dict[str, list[float]]] | None = None
    if args.use_mixture_dataloader:
        # Slow path: full WeightedDatasetMixture dataloader.
        logger.info(
            "Building WeightedDatasetMixture (parallel-loading %d datasets, workers=%d)...",
            len(mixture_cfg.datasets),
            args.num_workers,
        )
        t_build0 = time.perf_counter()
        train_cfg = _build_train_cfg(
            mixture_cfg,
            chunk_size=args.chunk_size,
            action_dim=args.action_dim,
            max_state_dim=args.max_state_dim,
            dataloader_batch_size=args.dataloader_batch_size,
            num_workers=args.num_workers,
        )
        mixture = _build_mixture_parallel(train_cfg, num_workers=args.num_workers)
        build_time = time.perf_counter() - t_build0
        logger.info("Mixture built in %.1fs", build_time)

        logger.info(
            "Draining %d chunks at batch_size=%d, num_workers=%d...",
            args.total_chunks,
            args.dataloader_batch_size,
            args.num_workers,
        )
        t_drain0 = time.perf_counter()
        dataloader = mixture.get_dataloader()
        all_chunks, n_batches = _drain_mixture(mixture.meta, dataloader, args.total_chunks, args.action_dim)
        drain_time = time.perf_counter() - t_drain0
        logger.info(
            "Collected %d chunks in %.1fs (%d batches)",
            all_chunks.shape[0],
            drain_time,
            n_batches,
        )
        if all_chunks.shape[0] == 0:
            raise RuntimeError("No chunks collected from dataloader. Check warnings above.")
        # Use mixture-computed stats.
        action_min, action_max = _extract_action_stats(mixture.meta, args.action_dim)
    else:
        # Fast path: aggregate stats manually (per-head or global), sample
        # chunks per-dataset with scipy interp for FPS resampling.
        if args.per_head_norm:
            per_ds_min, per_ds_max, per_ds_norm_keys, per_head_stats = _aggregate_stats_per_head(
                mixture_cfg, args.action_dim, args.num_workers
            )
            per_head_stats_report = {
                key: {
                    "min": [round(float(x), 6) for x in s["min"].tolist()],
                    "max": [round(float(x), 6) for x in s["max"].tolist()],
                }
                for key, s in per_head_stats.items()
            }
        else:
            action_min, action_max, _ = _aggregate_stats_manual(
                mixture_cfg, args.action_dim, args.num_workers
            )
            logger.info("Global action_min: %s", np.round(action_min, 3).tolist())
            logger.info("Global action_max: %s", np.round(action_max, 3).tolist())
        t_drain0 = time.perf_counter()
        raw_chunks, per_dataset_chunks, sample_errors = _sample_via_manual(
            mixture_cfg,
            args.total_chunks,
            args.chunk_size,
            mixture_cfg.action_freq,
            args.seed,
            args.num_workers,
        )
        if not raw_chunks:
            raise RuntimeError("No chunks sampled. Check warnings above.")
        # Pad each raw chunk to action_dim and normalize. Datasets with native
        # action dim > action_dim get their extra dims silently dropped -- if
        # the production policy uses max_action_dim < native, those dims are
        # never used at inference, so they shouldn't appear in the BPE corpus.
        max_native = max(c.shape[1] for c in raw_chunks)
        if max_native > args.action_dim:
            logger.warning(
                "Truncating extra action dims: max native dim is %d > --action-dim %d. "
                "Dims %d.. are silently dropped from the BPE corpus.",
                max_native,
                args.action_dim,
                args.action_dim,
            )
        n_steps = args.chunk_size
        dim = args.action_dim
        stacked = np.zeros((len(raw_chunks), n_steps, dim), dtype=np.float32)
        for i, c in enumerate(raw_chunks):
            clip = min(c.shape[1], dim)
            stacked[i, :, :clip] = c[:, :clip]
        if args.per_head_norm:
            all_chunks = _normalize_chunks_per_head(stacked, per_dataset_chunks, per_ds_min, per_ds_max)
        else:
            all_chunks = _normalize_chunks(stacked, action_min, action_max)
        drain_time = time.perf_counter() - t_drain0
        logger.info(
            "Manual path: %d chunks normalized in %.1fs (norm=%s)",
            all_chunks.shape[0],
            drain_time,
            "per_head" if args.per_head_norm else "global",
        )

    fit_time = 0.0
    roundtrip: dict[str, float] | None = None
    if not args.skip_fit:
        # --- Phase 4: fit ---
        processor, fit_time = _fit_tokenizer(
            all_chunks,
            args.vocab_size,
            args.scale,
            args.chunk_size,
            args.action_dim,
            args.max_token_length,
        )
        # --- Phase 5: save ---
        _save_processor(processor, out_dir)
        # --- Phase 6: verify ---
        rng = np.random.default_rng(args.seed + 1)
        n_sample = min(256, all_chunks.shape[0])
        sample_idx = rng.choice(all_chunks.shape[0], size=n_sample, replace=False)
        roundtrip = _verify_roundtrip(out_dir, all_chunks[sample_idx])

    report: dict[str, Any] = {
        "mixture_json": str(args.mixture_json),
        "mixture_hash_sha256": mixture_hash,
        "n_datasets": len(mixture_cfg.datasets),
        "action_freq": mixture_cfg.action_freq,
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "max_token_length": args.max_token_length,
        "seed": args.seed,
        "total_chunks_target": args.total_chunks,
        "total_chunks_actual": int(all_chunks.shape[0]),
        "dataloader_batch_size": args.dataloader_batch_size,
        "num_workers": args.num_workers,
        "sampler": "mixture_dataloader" if args.use_mixture_dataloader else "manual_parquet_interp",
        "normalization": "per_robot_type_control_mode" if args.per_head_norm else "global",
        "per_dataset_chunks": per_dataset_chunks,
        "sample_errors": sample_errors,
        "global_action_min": action_min.tolist() if action_min is not None else None,
        "global_action_max": action_max.tolist() if action_max is not None else None,
        "per_dataset_norm_keys": per_ds_norm_keys,
        "per_head_action_stats": per_head_stats_report,
        "n_norm_heads": (len(per_head_stats_report) if per_head_stats_report is not None else None),
        "timings_seconds": {
            "build_mixture": build_time,
            "drain": drain_time,
            "fit": fit_time,
        },
        "roundtrip": roundtrip,
        "pilot": args.pilot,
    }
    report_path = out_dir / "fit_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote fit report to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
