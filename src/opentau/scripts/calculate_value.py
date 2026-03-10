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
"""Compute value function outputs for a value policy over a configured dataset.

Loads a value policy from a checkpoint and runs predict_value on each batch from
the dataset mixture in the config. Saves (episode_index, timestamp) -> value to JSON.

Usage:
  # From checkpoint dir (has train_config.json and policy weights)
  python src/opentau/scripts/calculate_value.py \\
    --config_path=/path/to/checkpoints/00520000 \\
    [--batch_size=20] [--output_file=values.json]

  # Use checkpoint but override dataset (config + policy from checkpoint, data from file)
  python src/opentau/scripts/calculate_value.py \\
    --config_path=/path/to/checkpoints/00520000 \\
    --dataset_mixture=examples/my_datasets.json \\
    [--output_file=values.json]

  # From full train config file (policy.pretrained_path must point to checkpoint)
  python src/opentau/scripts/calculate_value.py \\
    --train_config=configs/train/value_config.json \\
    [--output_file=./values.json]

  # Override checkpoint when using train config
  python ... --train_config=configs/train/value_config.json --checkpoint_path=/path/to/checkpoints/00520000

  # Override dataset mixture (works with either --config_path or --train_config)
  python ... --dataset_mixture=examples/advantage_config.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device, init_logging


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.numpy()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.flat[0])
        return x
    return x


_dataset_mixture_path_value = None
_train_config_path_value = None
_output_file_value = None
for arg in sys.argv:
    if (
        arg.startswith("--dataset_mixture_path=")
        or arg.startswith("--dataset_mixture=")
        and "." not in arg.split("=", 1)[0]
    ):
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break
for arg in sys.argv:
    if arg.startswith("--train_config="):
        _train_config_path_value = arg.split("=", 1)[1]
        break
for arg in sys.argv:
    if arg.startswith("--output_file="):
        _output_file_value = arg.split("=", 1)[1]
        break

_checkpoint_path_value = None
for arg in sys.argv:
    if arg.startswith("--checkpoint_path="):
        _checkpoint_path_value = arg.split("=", 1)[1]
        break

_original_wrap = parser.wrap()


def _filter_script_args(fn):
    def wrapped(*args, **kwargs):
        if len(args) > 0:
            return _original_wrap(fn)(*args, **kwargs)
        original_argv = sys.argv.copy()
        try:
            filtered = []
            batch_size_val = None
            has_dataloader_batch_size = False
            for a in sys.argv:
                if a.startswith("--dataset_mixture_path=") or (
                    a.startswith("--dataset_mixture=") and "." not in a.split("=", 1)[0]
                ):
                    continue
                if (
                    a.startswith("--output_file=")
                    or a.startswith("--train_config=")
                    or a.startswith("--checkpoint_path=")
                ):
                    continue
                if a.startswith("--batch_size="):
                    batch_size_val = a.split("=", 1)[1]
                if a.startswith("--dataloader_batch_size="):
                    has_dataloader_batch_size = True
                filtered.append(a)
            if batch_size_val is not None and not has_dataloader_batch_size:
                filtered.append(f"--dataloader_batch_size={batch_size_val}")
            sys.argv = filtered
            return _original_wrap(fn)(*args, **kwargs)
        finally:
            sys.argv = original_argv

    return wrapped


@_filter_script_args
def main(cfg: TrainPipelineConfig):
    output_file = _output_file_value or "values.json"
    dataset_mixture_path = _dataset_mixture_path_value
    if _train_config_path_value:
        logging.info(f"Using full train config: {_train_config_path_value}")

    if dataset_mixture_path:
        logging.info(f"Overriding dataset: loading mixture from {dataset_mixture_path}")
        mixture_cfg = draccus.parse(
            config_class=DatasetMixtureConfig,
            config_path=dataset_mixture_path,
            args=[],
        )
    else:
        logging.info("Using dataset mixture from train config")
        mixture_cfg = cfg.dataset_mixture

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = auto_torch_device()
    checkpoint_path = _checkpoint_path_value or cfg.policy.pretrained_path
    logging.info("Loading value policy from checkpoint: %s", checkpoint_path)
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(checkpoint_path, config=cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    # (episode_index, timestamp) -> value (float)
    all_values = {}

    for dataset_idx, dataset_cfg in enumerate(mixture_cfg.datasets):
        logging.info(f"Creating dataset {dataset_idx}")
        result = make_dataset(dataset_cfg, cfg, return_advantage_input=True)
        dataset = result[0] if isinstance(result, tuple) else result

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=cfg.prefetch_factor,
        )

        with torch.inference_mode():
            for batch in dataloader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                values_tensor = policy.predict_value(batch)
                for ep_idx, ts, val in zip(
                    batch["episode_index"],
                    batch["timestamp"],
                    values_tensor,
                    strict=True,
                ):
                    ep_idx = _to_scalar(ep_idx)
                    ts = _to_scalar(ts)
                    val = _to_scalar(val)
                    if isinstance(ep_idx, np.ndarray):
                        ep_idx = int(ep_idx.flat[0])
                    if isinstance(ts, np.ndarray):
                        ts = int(ts.flat[0])
                    if isinstance(val, np.ndarray):
                        val = float(val.flat[0])
                    key = f"{ep_idx},{ts}"
                    all_values[key] = float(val)

    values_list = list(all_values.values())
    n = len(values_list)
    logging.info(f"Computed {n} values")

    if n == 0:
        logging.warning("No values computed (no LeRobotDataset in mixture or all skipped).")
        return

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_values, f, indent=2)
    logging.info(f"Saved values to {out_path}")

    arr = np.array(values_list)
    logging.info(f"Value stats: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, count={n}")

    # Plot value over timestamp
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping value-over-timestamp plot.")
    else:
        # Parse keys "ep_idx,ts" -> (episode_index, timestamp)
        # Timestamp may be int or float string (e.g. "0.0") depending on dataset format.
        timestamps = []
        values_plot = []
        for key, val in all_values.items():
            ep_idx_str, ts_str = key.split(",", 1)
            timestamps.append(int(float(ts_str)))
            values_plot.append(val)

        timestamps = np.array(timestamps)
        values_plot = np.array(values_plot)
        out_dir = out_path.parent
        plot_path = out_dir / "value_over_timestamp.png"

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        # 1) Scatter: all (timestamp, value) points
        axes[0].scatter(timestamps, values_plot, alpha=0.25, s=4, c="steelblue")
        axes[0].set_xlabel("Timestamp")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Value vs timestamp (all points)")
        axes[0].grid(True, alpha=0.3)

        # 2) Lines: value over timestamp for first N episodes
        by_episode = defaultdict(list)
        for key, val in all_values.items():
            ep_idx_str, ts_str = key.split(",", 1)
            by_episode[int(ep_idx_str)].append((int(float(ts_str)), val))
        max_episodes_plot = 10
        for ep_idx, points in sorted(by_episode.items())[:max_episodes_plot]:
            points.sort(key=lambda p: p[0])
            ts_ep = np.array([p[0] for p in points])
            val_ep = np.array([p[1] for p in points])
            axes[1].plot(ts_ep, val_ep, alpha=0.8, label=f"Episode {ep_idx}")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("Value")
        axes[1].set_title(f"Value vs timestamp (first {min(max_episodes_plot, len(by_episode))} episodes)")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved value-over-timestamp plot to {plot_path}")
    # end plot


if __name__ == "__main__":
    init_logging()
    if _train_config_path_value:
        cli_args = [
            a
            for a in sys.argv[1:]
            if not (
                a.startswith("--train_config=")
                or a.startswith("--output_file=")
                or a.startswith("--checkpoint_path=")
                or a.startswith("--dataset_mixture_path=")
                or (a.startswith("--dataset_mixture=") and "." not in a.split("=", 1)[0])
            )
        ]
        batch_val = next((a.split("=", 1)[1] for a in cli_args if a.startswith("--batch_size=")), None)
        if batch_val and not any(a.startswith("--dataloader_batch_size=") for a in cli_args):
            cli_args.append(f"--dataloader_batch_size={batch_val}")
        cfg = draccus.parse(
            config_class=TrainPipelineConfig,
            config_path=_train_config_path_value,
            args=cli_args,
        )
        main(cfg)
    else:
        main()
