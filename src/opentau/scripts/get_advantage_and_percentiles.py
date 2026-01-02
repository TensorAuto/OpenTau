r"python src/opentau/scripts/get_advantage_and_percentiles.py  \
--config_path=outputs/train/2025-11-29/00-38-59_value/checkpoints/00520000 \
--batch_size=20 \
--dataloader_batch_size=20 \
--dataset_mixture=examples/advantage_config.json"

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
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.utils import ADVANTAGES_PATH
from opentau.policies.factory import get_policy_class
from opentau.policies.value.reward import calculate_n_step_return
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    init_logging,
)


def ensure_primitive(maybe_tensor):
    if isinstance(maybe_tensor, np.ndarray):
        return ensure_primitive(torch.from_numpy(maybe_tensor))
    if isinstance(maybe_tensor, torch.Tensor):
        assert maybe_tensor.numel() == 1, f"Tensor must be a single value, got shape={maybe_tensor.numel()}"
    return maybe_tensor


_default0 = defaultdict(int)

# Store dataset_mixture_path before filtering (needed for parsing inside main)
# Handle both --dataset_mixture_path=<path> and --dataset_mixture=<path> (without nested fields)
_dataset_mixture_path_value = None
for arg in sys.argv:
    if arg.startswith("--dataset_mixture_path="):
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break
    elif arg.startswith("--dataset_mixture=") and "." not in arg.split("=", 1)[0]:
        # --dataset_mixture=<path> without nested fields (e.g., not --dataset_mixture.datasets.0.repo_id=...)
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break

# Create a wrapper that filters dataset_mixture path arguments before draccus parsing
_original_wrap = parser.wrap()


def _filter_dataset_mixture_path(fn):
    """Wrapper that filters --dataset_mixture_path and --dataset_mixture=<path> from sys.argv before draccus sees it."""
    wrapped_fn = _original_wrap(fn)

    def filtered_wrapper(*args, **kwargs):
        # If config is already provided, just call the function
        if len(args) > 0:
            return wrapped_fn(*args, **kwargs)

        # Otherwise, filter dataset_mixture path arguments from sys.argv before draccus parses
        original_argv = sys.argv.copy()
        try:
            filtered_args = []
            for arg in sys.argv:
                # Filter --dataset_mixture_path=<path>
                if (
                    arg.startswith("--dataset_mixture_path=")
                    or arg.startswith("--dataset_mixture=")
                    and "." not in arg.split("=", 1)[0]
                ):
                    continue
                else:
                    filtered_args.append(arg)
            sys.argv = filtered_args
            return wrapped_fn(*args, **kwargs)
        finally:
            sys.argv = original_argv

    return filtered_wrapper


@_filter_dataset_mixture_path
def main(cfg: TrainPipelineConfig):
    dataset_mixture_path = _dataset_mixture_path_value

    if dataset_mixture_path:
        logging.info(f"Loading dataset config from separate file: {dataset_mixture_path}")
        mixture_cfg = draccus.parse(
            config_class=DatasetMixtureConfig, config_path=dataset_mixture_path, args=[]
        )
    else:
        logging.info("Using the dataset mixture config from the TrainPipelineConfig")
        mixture_cfg = cfg.dataset_mixture

    device = auto_torch_device()
    # torch.autograd.set_detect_anomaly(True)

    # TODO(shuheng): Do we need the random seed here?
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    # Advantages across all datasets
    advantages = []

    for dataset_idx, dataset_cfg in enumerate(mixture_cfg.datasets):
        logging.info(f"Creating dataset {dataset_idx}")
        dataset = make_dataset(dataset_cfg, cfg, return_advantage_input=True)
        assert isinstance(dataset, LeRobotDataset), (
            f"Expected instance of LeRobotDataset, got {type(dataset)}"
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=cfg.prefetch_factor,
        )

        values = {}
        ds_advantage = {}  # per-dataset advantages
        with torch.inference_mode():
            # First pass to get the values
            for batch in dataloader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                for success, episode_index, episode_end_idx, current_idx, v0 in zip(
                    batch["success"],
                    batch["episode_index"],
                    batch["episode_end_idx"],
                    batch["current_idx"],
                    policy.predict_value(batch),
                    strict=True,
                ):
                    success, episode_index, episode_end_idx, current_idx, v0 = map(
                        ensure_primitive, (success, episode_index, episode_end_idx, current_idx, v0)
                    )
                    reward = calculate_n_step_return(
                        success=success,
                        n_steps_look_ahead=cfg.policy.reward_config.N_steps_look_ahead,
                        episode_end_idx=episode_end_idx,
                        max_episode_length=cfg.policy.reward_config.reward_normalizer,
                        current_idx=current_idx,
                        c_neg=cfg.policy.reward_config.C_neg,
                    )

                    values[(episode_index, current_idx)] = {"v0": v0, "reward": reward}

            # Second pass to compute the advantages
            for batch in dataloader:
                for episode_index, current_idx, timestamp in zip(
                    batch["episode_index"],
                    batch["current_idx"],
                    batch["timestamp"],
                    strict=True,
                ):
                    episode_index, current_idx, timestamp = map(
                        ensure_primitive, (episode_index, current_idx, timestamp)
                    )
                    # check if the value for the next n_steps_look_ahead steps is available, else set it to 0
                    look_ahead_idx = current_idx + cfg.policy.reward_config.N_steps_look_ahead
                    vn = values.get((episode_index, look_ahead_idx), _default0)["v0"]
                    reward = values.get((episode_index, current_idx), _default0)["reward"]
                    v0 = values.get((episode_index, current_idx), _default0)["v0"]
                    advantage = ensure_primitive(reward + vn - v0)
                    advantages.append(advantage)
                    ds_advantage[(episode_index, timestamp)] = advantage

        # Convert tuple keys to strings for JSON serialization
        advantage_data_json = {f"{ep_idx},{ts}": val for (ep_idx, ts), val in ds_advantage.items()}

        # TODO(shuheng) avoid overwriting existing advantage files.
        with open(Path(dataset.root) / ADVANTAGES_PATH, "w") as f:
            json.dump(advantage_data_json, f, indent=4)

    # Calculate percentiles of advantages: 0th, 5th, 10th, ..., 100th
    percentiles = list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 100]
    advantage_percentiles = np.percentile(np.array(advantages), percentiles)

    print("Advantage percentiles for deciding epsilon threshold:")
    for p, val in zip(percentiles, advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")


if __name__ == "__main__":
    init_logging()
    main()
