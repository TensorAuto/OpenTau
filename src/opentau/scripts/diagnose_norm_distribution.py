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

"""CPU-only diagnostic: the distribution of *normalized* state/action values,
read from the actual parquet frames, per ``(robot_type, control_mode, dim)``.

Why this exists: every other stats view in the codebase is derived from
``meta/stats.json`` / ``episodes_stats.jsonl``. If outliers were stripped from
the parquet data but the stats metadata was never recomputed, those numbers are
stale. This tool streams the real frames of the configured episode subset,
applies the policy's normalization using the per-``(robot_type, control_mode)``
head stats, and summarizes:

  * the distribution of ``z = (x - mean) / (std + eps)`` per dim (histogram), and
  * the data-derived mean/std/min/max vs the metadata stats (the staleness
    signal: a fresh dim has ``z_std ~= 1``; a dim whose metadata std was inflated
    by a since-removed outlier has ``z_std << 1`` and ``meta_max >> data_max``).

It is streaming (fixed-size histograms, never all values in memory), parallel
(one process per dataset), and reuses the same lightweight metadata + parquet
machinery as ``fit_fast_tokenizer.py``.

Run (where the parquet data lives, e.g. a training node)::

    python src/opentau/scripts/diagnose_norm_distribution.py \
        --config=/path/to/train_mixture_snippet.json \
        --output_dir=outputs/norm_diag --num_workers=16 \
        --max_episodes_per_dataset=50

``--config`` is a ``DatasetMixtureConfig`` JSON (``$ref`` includes are
resolved). ``pyarrow`` is already a (transitive) dependency; plotting also needs
``matplotlib`` (e.g. ``uv sync --extra libero``) — without it the JSON/CSV
reports are still written and only the figures are skipped.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import draccus
import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8  # must match opentau.policies.normalize.EPS

# Histogram bins on the NORMALIZED value z (std units): a dense linear core for
# the shape near 0, symmetric log-spaced tails out to 1e5 std for outliers, and
# two open overflow bins. Shared across every dim so accumulators merge and
# plots align. Exact z_min / z_max are tracked separately, so the overflow bins
# only bound the binned view, not the reported extremes.
_CLIP = 8.0
_N_LIN = 160
_Z_MAX = 1e5
_N_LOG = 30


def make_bin_edges() -> np.ndarray:
    lin = np.linspace(-_CLIP, _CLIP, _N_LIN + 1)
    log_tail = np.logspace(math.log10(_CLIP), math.log10(_Z_MAX), _N_LOG + 1)[1:]
    edges = np.concatenate([[-np.inf], -log_tail[::-1], lin[1:-1], log_tail, [np.inf]])
    return edges.astype(np.float64)


BIN_EDGES = make_bin_edges()
N_BINS = len(BIN_EDGES) - 1
_BEYOND = (3.0, 5.0, 10.0)


@dataclasses.dataclass
class Args:
    """CLI for the normalized-distribution diagnostic."""

    config: str  # DatasetMixtureConfig JSON (with $ref includes)
    output_dir: str = "outputs/norm_diag"
    num_workers: int | None = None  # default: cpu_count()
    frame_stride: int = 1  # subsample every Nth frame per episode (speed)
    max_episodes_per_dataset: int | None = None  # cap episodes/dataset (speed)
    norm_mode: str = "MEAN_STD"  # MEAN_STD | MIN_MAX (must match the policy)
    emit_corrected_stats: bool = True  # write data-derived per-head stats override
    top_n: int = 40  # console: worst-N dims by staleness
    no_plots: bool = False
    hf_home: str | None = None  # else inherit $HF_HOME


class DimStats:
    """Streaming per-dim accumulator for one ``(head, feature)``, vectorized over
    the feature's ``D`` dims. Holds exact raw-x moments (for data-derived stats)
    and a histogram + moments of the normalized ``z``. All fields combine
    associatively, so partial results from workers merge order-independently.
    """

    __slots__ = (
        "D",
        "count",
        "xsum",
        "xsumsq",
        "xmin",
        "xmax",
        "n_nonfinite",
        "hist",
        "zmin",
        "zmax",
        "zsum",
        "zsumsq",
        "n_beyond",
    )

    def __init__(self, n_dims: int):
        d = int(n_dims)
        self.D = d
        self.count = np.zeros(d, np.int64)
        self.xsum = np.zeros(d, np.float64)
        self.xsumsq = np.zeros(d, np.float64)
        self.xmin = np.full(d, np.inf, np.float64)
        self.xmax = np.full(d, -np.inf, np.float64)
        self.n_nonfinite = np.zeros(d, np.int64)
        self.hist = np.zeros((d, N_BINS), np.int64)
        self.zmin = np.full(d, np.inf, np.float64)
        self.zmax = np.full(d, -np.inf, np.float64)
        self.zsum = np.zeros(d, np.float64)
        self.zsumsq = np.zeros(d, np.float64)
        self.n_beyond = np.zeros((d, len(_BEYOND)), np.int64)

    def update(self, x: np.ndarray, z: np.ndarray) -> None:
        """Accumulate a ``(n, D)`` batch of raw values ``x`` and their normalized
        counterparts ``z``. Non-finite raw entries are counted and excluded."""
        if x.ndim != 2 or x.shape[1] != self.D:
            raise ValueError(f"expected (n, {self.D}) batch, got {x.shape}")
        finite = np.isfinite(x) & np.isfinite(z)
        self.n_nonfinite += (~finite).sum(axis=0)
        xf = np.where(finite, x, np.nan)
        zf = np.where(finite, z, np.nan)
        self.count += finite.sum(axis=0)
        self.xsum += np.nansum(xf, axis=0)
        self.xsumsq += np.nansum(xf * xf, axis=0)
        self.zsum += np.nansum(zf, axis=0)
        self.zsumsq += np.nansum(zf * zf, axis=0)
        # Per-dim running extremes (fmin/fmax ignore NaN, so all-NaN cols are no-ops).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column slices
            self.xmin = np.fmin(self.xmin, np.nanmin(xf, axis=0))
            self.xmax = np.fmax(self.xmax, np.nanmax(xf, axis=0))
            self.zmin = np.fmin(self.zmin, np.nanmin(zf, axis=0))
            self.zmax = np.fmax(self.zmax, np.nanmax(zf, axis=0))
        for i, k in enumerate(_BEYOND):
            self.n_beyond[:, i] += (np.abs(zf) > k).sum(axis=0)  # NaN > k is False
        # Histogram of z via one flat bincount over finite entries.
        bins = np.clip(np.searchsorted(BIN_EDGES, zf, side="right") - 1, 0, N_BINS - 1)
        flat = (np.arange(self.D)[None, :] * N_BINS + bins)[finite]
        if flat.size:
            self.hist += np.bincount(flat, minlength=self.D * N_BINS).reshape(self.D, N_BINS)

    def merge(self, other: DimStats) -> DimStats:
        self.count += other.count
        self.xsum += other.xsum
        self.xsumsq += other.xsumsq
        self.zsum += other.zsum
        self.zsumsq += other.zsumsq
        self.n_nonfinite += other.n_nonfinite
        self.n_beyond += other.n_beyond
        self.hist += other.hist
        self.xmin = np.fmin(self.xmin, other.xmin)
        self.xmax = np.fmax(self.xmax, other.xmax)
        self.zmin = np.fmin(self.zmin, other.zmin)
        self.zmax = np.fmax(self.zmax, other.zmax)
        return self

    def data_mean(self) -> np.ndarray:
        return np.divide(self.xsum, self.count, out=np.full(self.D, np.nan), where=self.count > 0)

    def data_std(self) -> np.ndarray:
        mean = self.data_mean()
        var = np.divide(self.xsumsq, self.count, out=np.full(self.D, np.nan), where=self.count > 0) - mean**2
        return np.sqrt(np.clip(var, 0.0, None))

    def z_mean(self) -> np.ndarray:
        return np.divide(self.zsum, self.count, out=np.full(self.D, np.nan), where=self.count > 0)

    def z_std(self) -> np.ndarray:
        zm = self.z_mean()
        var = np.divide(self.zsumsq, self.count, out=np.full(self.D, np.nan), where=self.count > 0) - zm**2
        return np.sqrt(np.clip(var, 0.0, None))


def _normalize(x: np.ndarray, stat: dict, mode: str) -> np.ndarray:
    """Apply the policy's normalization to ``x`` (n, D) — mirrors
    ``opentau.policies.normalize.Normalize.forward``."""
    if mode == "MEAN_STD":
        return (x - stat["mean"]) / (stat["std"] + EPS)
    if mode == "MIN_MAX":
        return (x - stat["min"]) / (stat["max"] - stat["min"] + EPS) * 2.0 - 1.0
    raise ValueError(f"unsupported norm_mode {mode!r}")


def _slice_stat(stat: dict, dim: int) -> dict:
    """Slice a (possibly zero-padded) head stat dict down to a feature's native
    dim, so workers normalize against the real columns only."""
    return {k: np.asarray(stat[k], np.float64)[:dim].copy() for k in ("mean", "std", "min", "max")}


# --------------------------------------------------------------------------- #
# Worker: stream one dataset's parquet, return per-feature DimStats.
# --------------------------------------------------------------------------- #
def _process_dataset(task: dict) -> dict:
    import pyarrow.parquet as pq  # lazy: keep workers torch-free and fast to spawn

    stride = max(1, int(task["frame_stride"]))
    mode = task["norm_mode"]
    out: dict[tuple, DimStats] = {}
    feats = []
    for feat in ("state", "actions"):
        col, dim, stat = task[feat]
        if col is None or dim <= 0:
            continue
        feats.append((feat, col, dim, stat, DimStats(dim)))

    for path in task["parquet_paths"]:
        if not os.path.exists(path):
            continue
        cols = [c for (_f, c, _d, _s, _a) in feats]
        try:
            table = pq.read_table(path, columns=cols)
        except Exception as e:  # noqa: BLE001 -- one bad shard shouldn't sink the dataset
            logger.warning("skip %s: %r", path, e)
            continue
        for _feat, col, dim, stat, acc in feats:
            arr = np.asarray(table.column(col).to_pylist(), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != dim:
                logger.warning("skip %s col %s: shape %s != (n, %d)", path, col, arr.shape, dim)
                continue
            if stride > 1:
                arr = arr[::stride]
            if arr.shape[0] == 0:
                continue
            acc.update(arr, _normalize(arr, stat, mode))

    for feat, _col, _dim, _stat, acc in feats:
        out[(task["head_key"], feat)] = acc
    return out


# --------------------------------------------------------------------------- #
# Parent: build the lightweight mixture metadata + per-dataset tasks.
# --------------------------------------------------------------------------- #
def _build_tasks(args: Args) -> tuple[list[dict], dict, list[str]]:
    """Returns (tasks, head_meta_stats, norm_keys). Reuses the real
    DatasetMixtureMetadata so per-(robot_type, control_mode) head stats and the
    dual-split column resolution match training exactly.

    NOTE: this is intentionally coupled to ``DatasetMixtureMetadata`` internals
    (``per_norm_key_stats``, ``norm_key_to_index``, ``dataset_to_norm_index``,
    ``norm_key_to_dataset_names``) so it mirrors the policy's normalization
    wiring without building full (video-backed) datasets. If those attributes
    are renamed this must be updated. The accumulator/worker math is unit-tested;
    this metadata-wiring half is exercised by running the script end-to-end.
    """
    from types import SimpleNamespace

    from opentau.configs.default import DatasetMixtureConfig
    from opentau.configs.refs import resolve_refs_to_tempfile
    from opentau.datasets.dataset_mixture import DatasetMixtureMetadata, WeightedDatasetMixture
    from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from opentau.datasets.standard_data_format_mapping import resolve_feature_mapping

    tmp = resolve_refs_to_tempfile(args.config)
    mix_cfg = draccus.parse(config_class=DatasetMixtureConfig, config_path=str(tmp), args=[])
    dataset_cfgs = list(mix_cfg.datasets)
    weights = list(mix_cfg.weights) if mix_cfg.weights is not None else [1.0] * len(dataset_cfgs)

    metadatas, kept_cfgs, kept_w = [], [], []
    for dc, w in zip(dataset_cfgs, weights, strict=True):
        try:
            meta = LeRobotDatasetMetadata(dc.repo_id, root=dc.root, revision=dc.revision)
        except Exception as e:  # noqa: BLE001
            logger.warning("metadata load failed for %s: %r", dc.repo_id, e)
            continue
        if dc.robot_type is not None:
            meta.info["robot_type"] = dc.robot_type
        if dc.control_mode is not None:
            meta.info["control_mode"] = dc.control_mode
        metadatas.append(meta)
        kept_cfgs.append(dc)
        kept_w.append(w)
    mix_cfg.datasets = kept_cfgs
    logger.info("loaded %d/%d dataset metadatas", len(metadatas), len(dataset_cfgs))

    # Raw (pre-padding) columns + dims per entry, via the FIXED resolver so a
    # joint entry reads action_joint and an ee entry reads action_ee.
    cols, dims = [], []
    max_s = max_a = 1
    for dc, m in zip(kept_cfgs, metadatas, strict=True):
        nm = resolve_feature_mapping(dc.repo_id, m.info.get("control_mode"))
        scol, acol = nm.get("state"), nm.get("actions")
        sdim = int(np.asarray(m.stats[scol]["mean"]).shape[-1]) if scol in m.stats else 0
        adim = int(np.asarray(m.stats[acol]["mean"]).shape[-1]) if acol in m.stats else 0
        cols.append((scol, acol))
        dims.append((sdim, adim))
        max_s, max_a = max(max_s, sdim), max(max_a, adim)

    cfg_shim = SimpleNamespace(
        max_state_dim=max_s, max_action_dim=max_a, num_cams=0, resolution=(224, 224), dataset_mixture=mix_cfg
    )
    names = WeightedDatasetMixture._make_dataset_names(cfg_shim, metadatas)
    mm = DatasetMixtureMetadata(cfg_shim, metadatas, kept_w, dataset_names=names)
    logger.info("%d (robot_type, control_mode) heads over %d datasets", len(mm.norm_keys), len(names))

    # Per-head meta stats, sliced to each head's native dim.
    head_meta: dict = {}
    for key in mm.norm_keys:
        st = mm.per_norm_key_stats[mm.norm_key_to_index[key]]
        head_meta[key] = st

    tasks: list[dict] = []
    for dc, m, name, (scol, acol), (sdim, adim) in zip(kept_cfgs, metadatas, names, cols, dims, strict=True):
        head_key = mm.norm_keys[mm.dataset_to_norm_index[name]]
        episodes = list(dc.episodes) if dc.episodes is not None else list(m.episodes)
        if args.max_episodes_per_dataset is not None:
            episodes = episodes[: args.max_episodes_per_dataset]
        parquet_paths = [str(m.root / m.get_data_file_path(ep)) for ep in episodes]
        st = head_meta[head_key]
        tasks.append(
            {
                "head_key": head_key,
                "parquet_paths": parquet_paths,
                "frame_stride": args.frame_stride,
                "norm_mode": args.norm_mode,
                "state": (scol, sdim, _slice_stat(st["state"], sdim) if sdim else None),
                "actions": (acol, adim, _slice_stat(st["actions"], adim) if adim else None),
            }
        )

    head_info = {
        key: {
            "robot_type": key.split("::")[0] if "::" in key else key,
            "control_mode": key.split("::", 1)[1] if "::" in key else "",
            "datasets": mm.norm_key_to_dataset_names[key],
            "meta": head_meta[key],
        }
        for key in mm.norm_keys
    }
    return tasks, head_info, list(mm.norm_keys)


# --------------------------------------------------------------------------- #
# Reporting + plotting.
# --------------------------------------------------------------------------- #
def _finalize_report(merged: dict, head_info: dict, norm_keys: list[str]) -> dict:
    rows = []
    for key in norm_keys:
        for feat in ("state", "actions"):
            acc = merged.get((key, feat))
            if acc is None or acc.count.sum() == 0:
                continue
            meta = head_info[key]["meta"][feat]
            dmean, dstd = acc.data_mean(), acc.data_std()
            zmean, zstd = acc.z_mean(), acc.z_std()
            for d in range(acc.D):
                cnt = int(acc.count[d])
                if cnt == 0:
                    continue
                mmax = float(np.asarray(meta["max"])[d])
                dmax = float(acc.xmax[d])
                ratio = math.log10(abs(mmax) / abs(dmax)) if dmax and abs(dmax) > 1e-12 and mmax else 0.0
                rows.append(
                    {
                        "robot_type": head_info[key]["robot_type"],
                        "control_mode": head_info[key]["control_mode"],
                        "feature": feat,
                        "dim": d,
                        "count": cnt,
                        "n_nonfinite": int(acc.n_nonfinite[d]),
                        "data_mean": float(dmean[d]),
                        "data_std": float(dstd[d]),
                        "data_min": float(acc.xmin[d]),
                        "data_max": dmax,
                        "meta_mean": float(np.asarray(meta["mean"])[d]),
                        "meta_std": float(np.asarray(meta["std"])[d]),
                        "meta_min": float(np.asarray(meta["min"])[d]),
                        "meta_max": mmax,
                        "z_mean": float(zmean[d]),
                        "z_std": float(zstd[d]),
                        "z_min": float(acc.zmin[d]),
                        "z_max": float(acc.zmax[d]),
                        "frac_gt3": acc.n_beyond[d, 0] / cnt,
                        "frac_gt5": acc.n_beyond[d, 1] / cnt,
                        "frac_gt10": acc.n_beyond[d, 2] / cnt,
                        "log10_meta_over_data_max": ratio,
                    }
                )
    return {"eps": EPS, "bin_edges": BIN_EDGES.tolist(), "rows": rows}


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _emit_corrected_stats(merged: dict, norm_keys: list[str], path: Path) -> None:
    out: dict = {}
    for key in norm_keys:
        entry = {}
        for feat in ("state", "actions"):
            acc = merged.get((key, feat))
            if acc is None or acc.count.sum() == 0:
                continue
            entry[feat] = {
                "mean": acc.data_mean().tolist(),
                "std": acc.data_std().tolist(),
                "min": np.where(np.isfinite(acc.xmin), acc.xmin, np.nan).tolist(),
                "max": np.where(np.isfinite(acc.xmax), acc.xmax, np.nan).tolist(),
                "count": int(acc.count.max()),
            }
        if entry:
            out[key] = entry
    path.write_text(json.dumps(out, indent=2))


def _plot(merged: dict, head_info: dict, norm_keys: list[str], rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots (e.g. `uv sync --extra libero`).")
        return

    for key in norm_keys:
        feats = [(f, merged.get((key, f))) for f in ("state", "actions")]
        feats = [(f, a) for f, a in feats if a is not None and a.count.sum() > 0]
        if not feats:
            continue
        fig, axes = plt.subplots(1, len(feats), figsize=(7 * len(feats), 5), squeeze=False)
        for ax, (feat, acc) in zip(axes[0], feats, strict=False):
            dens = acc.hist / np.clip(acc.hist.sum(axis=1, keepdims=True), 1, None)
            im = ax.imshow(np.log10(dens + 1e-6), aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"{key}  {feat}  (D={acc.D})")
            ax.set_xlabel("normalized z (std units)")
            ax.set_ylabel("dim")
            zero_bin = int(np.searchsorted(BIN_EDGES, 0.0, side="right") - 1)
            ax.axvline(zero_bin, color="cyan", lw=0.5, alpha=0.6)
            fig.colorbar(im, ax=ax, label="log10 density")
        fig.tight_layout()
        safe = key.replace("/", "_").replace("::", "__")
        fig.savefig(out_dir / f"head__{safe}.png", dpi=120)
        plt.close(fig)

    # Global health heatmap: rows = worst dims, cols = staleness metrics.
    if rows:
        ranked = sorted(
            rows,
            key=lambda r: -max(abs(r["z_std"] - 1.0), r["frac_gt10"], abs(r["log10_meta_over_data_max"])),
        )
        top = ranked[: min(60, len(ranked))]
        labels = [f"{r['robot_type']}::{r['control_mode']}|{r['feature'][:3]}{r['dim']}" for r in top]
        cols = ["z_std", "frac_gt10", "log10(meta/data max)", "z_max"]
        mat = np.array(
            [[r["z_std"], r["frac_gt10"], r["log10_meta_over_data_max"], min(r["z_max"], 50)] for r in top]
        )
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.25)))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
        ax.set_yticks(range(len(top)), labels, fontsize=6)
        ax.set_title("Staleness / health (top dims)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "health_heatmap.png", dpi=120)
        plt.close(fig)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, head_info, norm_keys = _build_tasks(args)
    if not tasks:
        logger.error("no datasets to process")
        return

    n_workers = args.num_workers or cpu_count()
    merged: dict[tuple, DimStats] = {}
    logger.info("streaming %d datasets over %d workers", len(tasks), n_workers)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_dataset, t) for t in tasks]
        for fut in as_completed(futs):
            for k, acc in fut.result().items():
                merged[k] = merged[k].merge(acc) if k in merged else acc
            done += 1
            if done % 25 == 0 or done == len(tasks):
                logger.info("  processed %d/%d datasets", done, len(tasks))

    report = _finalize_report(merged, head_info, norm_keys)
    rows = report["rows"]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_csv(rows, out_dir / "report.csv")
    if args.emit_corrected_stats:
        _emit_corrected_stats(merged, norm_keys, out_dir / "corrected_stats.json")
    if not args.no_plots:
        _plot(merged, head_info, norm_keys, rows, out_dir)

    # Console: worst dims by staleness (z_std far from 1, heavy tails, meta/data max gap).
    ranked = sorted(
        rows, key=lambda r: -max(abs(r["z_std"] - 1.0), r["frac_gt10"], abs(r["log10_meta_over_data_max"]))
    )
    logger.info("wrote %s (%d dim-rows). Worst %d dims by staleness:", out_dir, len(rows), args.top_n)
    logger.info(
        "  %-34s %-7s %3s %9s %9s %9s %10s %10s",
        "robot::mode",
        "feat",
        "dim",
        "z_std",
        "frac>10",
        "z_max",
        "data_max",
        "meta_max",
    )
    for r in ranked[: args.top_n]:
        logger.info(
            "  %-34s %-7s %3d %9.3f %9.4f %9.1f %10.4g %10.4g",
            f"{r['robot_type']}::{r['control_mode']}"[:34],
            r["feature"],
            r["dim"],
            r["z_std"],
            r["frac_gt10"],
            r["z_max"],
            r["data_max"],
            r["meta_max"],
        )


if __name__ == "__main__":
    main(draccus.parse(config_class=Args))
