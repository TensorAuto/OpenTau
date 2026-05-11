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
fitting on the same min-max-normalized action chunks the policy will see
at training time.

Output is a ``processor`` directory loadable via
``AutoProcessor.from_pretrained(out_dir, trust_remote_code=True)``.

Pipeline (all CPU):

    1. Resolve ``$ref`` includes in the mixture JSON and parse it as a
       ``DatasetMixtureConfig``.
    2. Aggregate per-dim ``action`` min/max across the mixture by reading
       each dataset's ``LeRobotDatasetMetadata`` (cheap, only ``meta/``).
    3. Allocate per-dataset chunk budgets weight-proportional to
       ``mixture.weights``, clamped to ``[floor, cap]``.
    4. Sample action chunks from each dataset's parquet files (no video
       decode, no LeRobotDataset machinery).
    5. Min-max-normalize each chunk to ``[-1, 1]`` using the aggregated
       stats; right-pad ``action_dim`` to the policy's ``max_action_dim``.
    6. Call ``UniversalActionProcessor.fit(...)`` (DCT + Rust BpeTrainer)
       and ``save_pretrained`` the result. The upstream remote-code source
       ``processing_action_tokenizer.py`` is copied alongside so the saved
       directory loads via ``trust_remote_code=True``.
    7. Round-trip a held-out sample of chunks for a sanity-check MSE and
       average token length.

Usage:

    python -m opentau.scripts.fit_fast_tokenizer \\
        --mixture-json /path/to/mixture.json \\
        --out-dir /path/to/output_dir/ \\
        [--total-chunks 1000000] [--chunk-size 10] [--action-dim 32] \\
        [--vocab-size 2048] [--scale 10] [--seed 0] [--num-workers 16] \\
        [--cap-per-dataset 30000] [--floor-per-dataset 200] [--pilot]

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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile

logger = logging.getLogger(__name__)

UPSTREAM_REPO_ID = "physical-intelligence/fast"
UPSTREAM_SOURCE_FILE = "processing_action_tokenizer.py"


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
        default=10,
        help="Time horizon per action chunk (matches policy n_action_steps).",
    )
    p.add_argument(
        "--action-dim",
        type=int,
        default=32,
        help="Padded action dimension (matches policy max_action_dim).",
    )
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap-per-dataset", type=int, default=30_000)
    p.add_argument("--floor-per-dataset", type=int, default=200)
    p.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Parallel processes for per-dataset stats + sampling phases.",
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


def _resolve_native_action_key(cfg: DatasetConfig) -> str:
    """Return the native column / stats key for actions in ``cfg``'s dataset.

    Falls back to ``"action"`` (the LeRobot convention) when no mapping is
    found. Side note: ``DatasetConfig.__post_init__`` upserts any per-config
    ``data_features_name_mapping`` into ``DATA_FEATURES_NAME_MAPPING`` at
    parse time, so a single global lookup suffices.
    """
    from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

    mapping = DATA_FEATURES_NAME_MAPPING.get(cfg.repo_id, {}) if cfg.repo_id else {}
    return mapping.get("actions", "action")


def _load_dataset_stats(
    item: tuple[int, DatasetConfig],
) -> tuple[int, str, dict | None, str | None]:
    """Worker: load action min/max from a single dataset's metadata.

    Returns ``(idx, repo_id, {"min": ..., "max": ...} | None, error_str | None)``.
    """
    idx, cfg = item
    try:
        from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        action_key = _resolve_native_action_key(cfg)
        if not meta.stats or action_key not in meta.stats:
            return (
                idx,
                cfg.repo_id,
                None,
                f"key {action_key!r} missing from stats (keys={sorted(meta.stats or [])})",
            )
        action_stats = meta.stats[action_key]
        return (
            idx,
            cfg.repo_id,
            {
                "min": np.asarray(action_stats["min"], dtype=np.float64).ravel(),
                "max": np.asarray(action_stats["max"], dtype=np.float64).ravel(),
            },
            None,
        )
    except Exception as e:  # noqa: BLE001  (worker boundary -- collect all errors)
        return idx, cfg.repo_id or "<no-repo-id>", None, f"{type(e).__name__}: {e}"


def _aggregate_stats(
    mixture_cfg: DatasetMixtureConfig, action_dim: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
    """Element-wise nan-tolerant min / max across the mixture.

    Pads each dataset's stats to ``action_dim`` (NaN sentinel for padded slots).
    Dims that stay NaN across the whole mixture fall back to ``[-1, 1]``.
    """
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
            record = {"repo_id": repo_id, "ok": stats is not None, "error": err}
            per_dataset[idx] = record
            if stats is None:
                n_fail += 1
                logger.warning("Dataset %d (%s): no stats -- %s", idx, repo_id, err)
                continue
            n_ok += 1
            native_dim = int(stats["min"].shape[0])
            record["native_action_dim"] = native_dim
            mn = np.full(action_dim, np.nan, dtype=np.float64)
            mx = np.full(action_dim, np.nan, dtype=np.float64)
            clip = min(action_dim, native_dim)
            mn[:clip] = stats["min"][:clip]
            mx[:clip] = stats["max"][:clip]
            mins_padded.append(mn)
            maxs_padded.append(mx)

    if not mins_padded:
        raise RuntimeError("No dataset stats were loaded successfully -- cannot proceed.")

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

    elapsed = time.perf_counter() - t0
    logger.info("Stats aggregation done in %.1fs (%d ok, %d fail).", elapsed, n_ok, n_fail)
    logger.info("Global action_min: %s", np.round(action_min, 3).tolist())
    logger.info("Global action_max: %s", np.round(action_max, 3).tolist())
    return action_min, action_max, per_dataset


def _compute_budgets(
    mixture_cfg: DatasetMixtureConfig,
    total_chunks: int,
    cap: int,
    floor: int,
) -> list[int]:
    """Per-dataset budget: weight-proportional, clamped to ``[floor, cap]``."""
    n = len(mixture_cfg.datasets)
    weights = mixture_cfg.weights if mixture_cfg.weights is not None else [1.0] * n
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(n, dtype=np.float64)
    budgets = np.round(total_chunks * weights / weights.sum()).astype(int)
    budgets = np.clip(budgets, floor, cap)
    return budgets.tolist()


def _sample_chunks_for_dataset(
    args: tuple[int, DatasetConfig, int, int, int],
) -> tuple[int, list[np.ndarray], str | None]:
    """Worker: sample ``n_chunks`` action chunks of shape ``(chunk_size, native_dim)``.

    Reads raw parquet via ``pyarrow`` directly (no video decode, no
    ``LeRobotDataset`` machinery, no FPS resampling). The fitted tokenizer
    is a function of the DCT-coefficient distribution, which is mostly
    determined by per-step action deltas; the resampling-induced jitter
    is small enough that it does not matter for the BPE.
    """
    idx, cfg, n_chunks, chunk_size, seed = args
    if n_chunks <= 0:
        return idx, [], None
    try:
        import pyarrow.parquet as pq

        from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        # Episode filter (None means "all episodes")
        eligible_eps = (
            list(meta.episodes.keys())
            if cfg.episodes is None
            else [ep for ep in cfg.episodes if ep in meta.episodes]
        )
        ep_starts: list[tuple[int, int]] = []
        for ep in eligible_eps:
            length = int(meta.episodes[ep].get("length", 0))
            valid = max(0, length - chunk_size + 1)
            if valid > 0:
                ep_starts.append((ep, valid))
        if not ep_starts:
            return idx, [], f"no episode has length >= chunk_size={chunk_size}"

        rng = np.random.default_rng(seed)
        ep_ids = np.array([e for e, _ in ep_starts])
        ep_weights = np.array([v for _, v in ep_starts], dtype=np.float64)
        probs = ep_weights / ep_weights.sum()
        chosen_eps = rng.choice(ep_ids, size=n_chunks, p=probs)
        # Group by episode so we read each parquet at most once.
        valid_by_ep = dict(ep_starts)
        ep_to_starts: dict[int, list[int]] = defaultdict(list)
        for ep in chosen_eps:
            ep_int = int(ep)
            start = int(rng.integers(0, valid_by_ep[ep_int]))
            ep_to_starts[ep_int].append(start)

        action_col = _resolve_native_action_key(cfg)
        chunks: list[np.ndarray] = []
        for ep, starts in ep_to_starts.items():
            rel = meta.get_data_file_path(ep_index=ep)
            full = meta.root / rel
            if not full.exists():
                meta.pull_from_repo(allow_patterns=str(rel))
            try:
                table = pq.read_table(full, columns=[action_col])
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Failed reading {action_col!r} column from {full}: {e}") from e
            arr_list = table.column(action_col).to_pylist()
            try:
                actions = np.asarray(arr_list, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise RuntimeError(f"{full}: {action_col!r} column is not a uniform 2-D array: {e}") from e
            if actions.ndim != 2:
                raise RuntimeError(f"{full}: {action_col!r} ndim={actions.ndim}, expected 2")
            for s in starts:
                chunks.append(actions[s : s + chunk_size])
        return idx, chunks, None
    except Exception as e:  # noqa: BLE001
        return idx, [], f"{type(e).__name__}: {e}"


def _normalize_and_pad(
    chunks: list[np.ndarray],
    action_min: np.ndarray,
    action_max: np.ndarray,
    action_dim: int,
) -> list[np.ndarray]:
    """Min-max-normalize each chunk to ``[-1, 1]``; right-pad ``action_dim``.

    Mirrors ``Normalize({"ACTION": NormalizationMode.MIN_MAX})`` in pi0.7's
    ``modeling_pi07_low_level.py``. Padded slots and constant dims map to 0;
    non-finite inputs map to 0 (the policy's ``prepare_discrete_actions``
    does the same via ``torch.nan_to_num``).
    """
    span = action_max - action_min
    safe_span = np.where(span > 0, span, 1.0)
    out: list[np.ndarray] = []
    for chunk in chunks:
        n_steps, native_dim = chunk.shape
        x = np.zeros((n_steps, action_dim), dtype=np.float32)
        clip = min(native_dim, action_dim)
        sub = np.nan_to_num(chunk[:, :clip].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        norm = 2.0 * (sub - action_min[:clip]) / safe_span[:clip] - 1.0
        zero_dims = np.where(span[:clip] <= 0)[0]
        if zero_dims.size > 0:
            norm[:, zero_dims] = 0.0
        x[:, :clip] = np.clip(norm, -1.0, 1.0).astype(np.float32)
        out.append(x)
    return out


def _fit_tokenizer(
    chunks: list[np.ndarray],
    vocab_size: int,
    scale: float,
    chunk_size: int,
    action_dim: int,
) -> tuple[Any, float]:
    """Fit a fresh ``UniversalActionProcessor`` via its classmethod ``fit``."""
    from transformers import AutoProcessor

    # We can't import the class as a Python module without adding the cached
    # snapshot dir to sys.path; using AutoProcessor + type(...) is portable.
    upstream = AutoProcessor.from_pretrained(UPSTREAM_REPO_ID, trust_remote_code=True)
    cls = type(upstream)
    logger.info(
        "Fitting %s on %d chunks (vocab=%d, scale=%g, chunk_size=%d, action_dim=%d)",
        cls.__name__,
        len(chunks),
        vocab_size,
        scale,
        chunk_size,
        action_dim,
    )
    t0 = time.perf_counter()
    processor = cls.fit(
        chunks,
        vocab_size=vocab_size,
        scale=scale,
        time_horizon=chunk_size,
        action_dim=action_dim,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "BPE fit done in %.1fs (min_token=%s, vocab_size=%s)",
        elapsed,
        getattr(processor, "min_token", "?"),
        getattr(processor, "vocab_size", "?"),
    )
    return processor, elapsed


def _find_upstream_source() -> Path:
    """Snapshot-download just the remote-code source file so we can copy it."""
    from huggingface_hub import snapshot_download

    cache_root = snapshot_download(repo_id=UPSTREAM_REPO_ID, allow_patterns=UPSTREAM_SOURCE_FILE)
    src = Path(cache_root) / UPSTREAM_SOURCE_FILE
    if not src.exists():
        raise FileNotFoundError(f"{UPSTREAM_SOURCE_FILE} not in {cache_root}")
    return src


def _save_processor(processor: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(out_dir))
    shutil.copy(_find_upstream_source(), out_dir / UPSTREAM_SOURCE_FILE)
    logger.info("Saved tokenizer + remote-code source to %s", out_dir)


def _verify_roundtrip(out_dir: Path, sample_chunks: list[np.ndarray]) -> dict[str, float]:
    """Reload via ``AutoProcessor.from_pretrained`` and round-trip encode/decode."""
    from transformers import AutoProcessor

    rp = AutoProcessor.from_pretrained(str(out_dir), trust_remote_code=True)
    arr = np.stack(sample_chunks).astype(np.float32)  # (N, T, D)
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

    # --- Phase 1: parse mixture ---
    logger.info("Parsing mixture JSON: %s", args.mixture_json)
    mixture_cfg, mixture_hash = _parse_mixture(args.mixture_json)
    logger.info(
        "Mixture has %d datasets (hash=%s); action_freq=%g",
        len(mixture_cfg.datasets),
        mixture_hash[:12],
        mixture_cfg.action_freq,
    )

    # --- Phase 2: aggregate stats ---
    action_min, action_max, per_dataset_info = _aggregate_stats(
        mixture_cfg, args.action_dim, args.num_workers
    )

    # --- Phase 3: per-dataset budgets ---
    budgets = _compute_budgets(
        mixture_cfg,
        args.total_chunks,
        args.cap_per_dataset,
        args.floor_per_dataset,
    )
    logger.info(
        "Per-dataset budgets: total=%d (target=%d), min=%d, max=%d",
        sum(budgets),
        args.total_chunks,
        min(budgets),
        max(budgets),
    )

    # --- Phase 4: sample chunks ---
    logger.info("Sampling chunks (workers=%d)", args.num_workers)
    t_sample0 = time.perf_counter()
    work = [
        (i, cfg, budgets[i], args.chunk_size, args.seed + i) for i, cfg in enumerate(mixture_cfg.datasets)
    ]
    all_chunks: list[np.ndarray] = []
    n_sampled_per_dataset = [0] * len(mixture_cfg.datasets)
    sample_errors: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_sample_chunks_for_dataset, w): w[0] for w in work}
        for k, fut in enumerate(as_completed(futs)):
            idx, chunks, err = fut.result()
            if err:
                sample_errors[idx] = err
                logger.warning("Dataset %d: %s", idx, err)
            n_sampled_per_dataset[idx] = len(chunks)
            all_chunks.extend(chunks)
            if (k + 1) % 25 == 0 or (k + 1) == len(work):
                logger.info(
                    "Sampling: %d/%d datasets done, %d chunks so far",
                    k + 1,
                    len(work),
                    len(all_chunks),
                )
    sample_time = time.perf_counter() - t_sample0
    logger.info("Sampling done in %.1fs. Total raw chunks: %d", sample_time, len(all_chunks))
    if not all_chunks:
        raise RuntimeError("No chunks were sampled. Check warnings above.")

    # --- Phase 5: normalize + pad ---
    t_norm0 = time.perf_counter()
    all_chunks = _normalize_and_pad(all_chunks, action_min, action_max, args.action_dim)
    norm_time = time.perf_counter() - t_norm0
    logger.info("Normalized + padded %d chunks in %.1fs", len(all_chunks), norm_time)

    fit_time = 0.0
    roundtrip: dict[str, float] | None = None
    if not args.skip_fit:
        # --- Phase 6: fit ---
        processor, fit_time = _fit_tokenizer(
            all_chunks,
            args.vocab_size,
            args.scale,
            args.chunk_size,
            args.action_dim,
        )
        # --- Phase 7: save ---
        _save_processor(processor, out_dir)
        # --- Phase 8: verify ---
        rng = np.random.default_rng(args.seed + 1)
        n_sample = min(256, len(all_chunks))
        sample_idx = rng.choice(len(all_chunks), size=n_sample, replace=False)
        roundtrip = _verify_roundtrip(out_dir, [all_chunks[i] for i in sample_idx])

    # --- Final report ---
    report: dict[str, Any] = {
        "mixture_json": str(args.mixture_json),
        "mixture_hash_sha256": mixture_hash,
        "n_datasets": len(mixture_cfg.datasets),
        "action_freq": mixture_cfg.action_freq,
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "seed": args.seed,
        "total_chunks_target": args.total_chunks,
        "total_chunks_actual": len(all_chunks),
        "cap_per_dataset": args.cap_per_dataset,
        "floor_per_dataset": args.floor_per_dataset,
        "global_action_min": action_min.tolist(),
        "global_action_max": action_max.tolist(),
        "per_dataset_chunks": n_sampled_per_dataset,
        "sample_errors": sample_errors,
        "timings_seconds": {
            "sample": sample_time,
            "normalize": norm_time,
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
