#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streamlit app to visualize VLA value function with camera images from a LeRobot dataset.

Reads dataset_config.json and values.json, loads a single episode via LeRobotDataset,
and provides a timeline scrubber + value curve. values.json is a dict with keys
(episode_idx, timestamp) and values as the value function outputs.

Usage:
    streamlit run src/opentau/scripts/value_visualizer_app.py -- \\
        --dataset-config path/to/dataset_config.json \\
        --values path/to/values.json \\
        [--train-config path/to/dir_or_train_config.json] \\
        [--episode 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from opentau.configs.default import DatasetConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset

# Hardcoded path to logo image shown in the header (change to your logo file).
LOGO_PATH = Path("assets/logo.png")


def load_dataset_config(path: Path) -> tuple[Path, str, list[int]]:
    """Load dataset_config.json; return (root, repo_id, list of episode indices)."""
    with open(path) as f:
        cfg = json.load(f)
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError(f"No 'datasets' in {path}")
    first = datasets[0]
    root = Path(first["root"]).resolve()
    repo_id = first.get("repo_id", "physical-intelligence/libero")
    episodes = first.get("episodes", [0])
    if isinstance(episodes, list):
        episodes = sorted(episodes)
    else:
        episodes = list(range(episodes))
    return root, repo_id, episodes


def load_values(path: Path) -> dict[tuple[int, float], float]:
    """Load values.json. Dict with keys (episode_idx, timestamp) and values as floats.

    JSON keys are strings like "episode_idx,timestamp"; normalize to (int, round(ts, 6)).
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        k = str(k).strip()
        if k.startswith("(") and k.endswith(")"):
            k = k[1:-1]
        parts = k.replace(" ", "").split(",", 1)
        if len(parts) != 2:
            continue
        ep_idx = int(parts[0])
        ts = round(float(parts[1]), 6)
        out[(ep_idx, ts)] = float(v)
    return out


def _tensor_or_array_to_pil(x) -> Image.Image | None:
    """Convert a tensor (C,H,W) or numpy array to PIL Image for display."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))
        if np.issubdtype(x.dtype, np.floating):
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(x)
    return None


@st.cache_data(show_spinner=True)
def load_frames_lerobot_cached(
    dataset_config_path: Path,
    train_config_path: Path | None,
    values_path: Path,
    episode_index: int,
) -> pd.DataFrame:
    """Load a single episode via LeRobotDataset and attach values. Returns DataFrame with step, value, image."""
    root, repo_id, _ = load_dataset_config(dataset_config_path)
    value_lookup = load_values(values_path)

    if train_config_path is None:
        train_config_path = dataset_config_path.parent / "train_config.json"
        if not train_config_path.is_file():
            train_config_path = dataset_config_path.parent
    path = Path(train_config_path).resolve()
    if path.is_dir():
        train_cfg = TrainPipelineConfig.from_pretrained(path, local_files_only=True)
    else:
        train_cfg = TrainPipelineConfig.from_pretrained(path, local_files_only=True)

    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=str(root),
        episodes=[episode_index],
    )
    res = make_dataset(dataset_cfg, train_cfg, return_advantage_input=False)
    dataset = res[0] if isinstance(res, tuple) else res
    camera_key = "camera0"

    # DataLoader calls __getitem__; each batch contains episode_index and timestamp.
    # With batch_size=1, each iteration visits one datapoint (one frame) exactly once.
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_rows = []
    for batch in dataloader:
        if "episode_index" not in batch or "timestamp" not in batch:
            # Fallback: batch from older dataset; build from per-sample __getitem__
            start_idx = len(all_rows)
            n_in_batch = next(
                v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0
            )
            for i in range(n_in_batch):
                item = dataset[start_idx + i]
                ep = item.get("episode_index", episode_index)
                ep = int(ep.item() if hasattr(ep, "item") else ep) if ep is not None else episode_index
                ts = item.get("timestamp")
                ts = float(ts.item() if hasattr(ts, "item") else ts) if ts is not None else 0.0
                key_round = (ep, round(ts, 6))
                value = value_lookup.get(key_round, value_lookup.get((ep, ts), np.nan))
                pil_img = _tensor_or_array_to_pil(item.get(camera_key))
                all_rows.append({"step": len(all_rows), "episode_index": ep, "timestamp": ts, "value": value, "image": pil_img})
            continue
        ep_b = batch["episode_index"]
        ts_b = batch["timestamp"]
        if ep_b.dim() > 1:
            ep_b = ep_b.squeeze(-1)
        if ts_b.dim() > 1:
            ts_b = ts_b.squeeze(-1)
        imgs_b = batch[camera_key]
        if imgs_b.dim() == 5:
            imgs_b = imgs_b[:, 0]
        for i in range(ep_b.shape[0]):
            ep = int(ep_b[i].item())
            ts = float(ts_b[i].item())
            key_round = (ep, round(ts, 6))
            value = value_lookup.get(key_round, value_lookup.get((ep, ts), np.nan))
            pil_img = _tensor_or_array_to_pil(imgs_b[i])
            all_rows.append({
                "step": len(all_rows),
                "episode_index": ep,
                "timestamp": ts,
                "value": value,
                "image": pil_img,
            })
    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["step"] = np.arange(len(df))
    return df


def run_app(df: pd.DataFrame) -> None:
    """Run the Streamlit UI with the prepared DataFrame (step, value, image)."""
    st.set_page_config(page_title="VLA Value Function Visualizer", layout="wide")
    # Header: logo (if file exists) + title
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=380)
    with title_col:
        st.title("VLA Value Function Analysis")
    st.markdown("Synchronize robot states with predicted value function progress (single episode).")

    st.sidebar.header("Settings")
    smoothing = st.sidebar.slider("Graph Smoothing (Window)", 1, 10, 3)
    if smoothing > 1:
        df = df.copy()
        df["display_value"] = df["value"].rolling(window=smoothing, center=True).mean()
    else:
        df = df.copy()
        df["display_value"] = df["value"]

    df["display_value"] = df["display_value"].ffill().bfill().fillna(0)

    n = len(df)
    if n == 0:
        st.warning("No frames loaded.")
        return

    # Slider is rendered below the value graph; use session state so col1/col2 can use it
    current_step = st.session_state.get("timeline_scrubber", 0)
    current_step = max(0, min(current_step, n - 1))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Camera Feed (Step {current_step})")
        row = df.iloc[current_step]
        if row["image"] is not None:
            st.image(row["image"], width="stretch")
        else:
            st.info("No image for this frame.")
        with st.container(border=True):
            st.metric("Predicted Value", f"{row['value']:.3f}" if np.isfinite(row['value']) else "—")
            st.caption(f"Episode {int(row['episode_index'])}, t = {row['timestamp']:.2f}s")

    with col2:
        st.subheader("Value Function $V(s)$")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df["display_value"],
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name="Value Function",
        ))
        fig.add_vline(x=current_step, line_width=2, line_dash="dash", line_color="red")
        y_cur = df["display_value"].iloc[current_step]
        fig.add_trace(go.Scatter(
            x=[current_step],
            y=[y_cur],
            mode="markers",
            marker=dict(color="red", size=10),
            showlegend=False,
        ))
        y_min = df["display_value"].min()
        y_max = df["display_value"].max()
        y_range = [y_min - 0.05 * (y_max - y_min + 1e-6), y_max + 0.05 * (y_max - y_min + 1e-6)]
        fig.update_layout(
            xaxis_title="Timestep (t)",
            yaxis_title="Value V(s)",
            yaxis=dict(range=y_range),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        # Slider under the value graph
        st.slider("Timeline Scrubber", 0, n - 1, current_step, key="timeline_scrubber")

    st.divider()
    m1, m2, m3 = st.columns(3)
    v_cur = df["value"].iloc[current_step]
    v_prev = df["value"].iloc[max(0, current_step - 1)] if current_step > 0 else v_cur
    m1.metric("Current Value", f"{v_cur:.3f}" if np.isfinite(v_cur) else "—")
    m2.metric("Max Value", f"{df['value'].max():.3f}" if np.isfinite(df["value"]).any() else "—")
    m3.metric("Step Delta", f"{v_cur - v_prev:.4f}" if np.isfinite(v_cur) and np.isfinite(v_prev) else "—")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLA Value Function Visualizer: single episode via LeRobotDataset, values from values.json"
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Path to dataset_config.json (contains root, repo_id, episodes)",
    )
    parser.add_argument(
        "--values",
        type=Path,
        required=True,
        help="Path to values.json (dict: keys (episode_idx, timestamp), values: float)",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Path to train_config.json or directory containing it (default: same dir as dataset-config)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to load (default: first episode from dataset_config)",
    )
    args = parser.parse_args()

    dataset_config_path = args.dataset_config.resolve()
    values_path = args.values.resolve()
    if not dataset_config_path.is_file():
        raise SystemExit(f"Dataset config not found: {dataset_config_path}")
    if not values_path.is_file():
        raise SystemExit(f"Values file not found: {values_path}")

    root, repo_id, episodes = load_dataset_config(dataset_config_path)
    episode_index = args.episode if args.episode is not None else episodes[0]

    df = load_frames_lerobot_cached(
        dataset_config_path,
        args.train_config,
        values_path,
        episode_index,
    )
    run_app(df)


if __name__ == "__main__":
    main()
