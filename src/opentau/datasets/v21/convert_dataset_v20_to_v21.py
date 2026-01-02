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

"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.0 to
2.1. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial
```

"""

import argparse
import logging
from dataclasses import dataclass

from huggingface_hub import HfApi

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from opentau.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from opentau.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    """Context manager to temporarily suppress logging warnings.

    Sets logging level to ERROR on entry and restores previous level on exit.
    """

    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def create_fake_train_config() -> TrainPipelineConfig:
    """Create a fake TrainPipelineConfig for dataset conversion."""

    # Minimal dummy PreTrainedConfig implementation
    @dataclass
    class DummyPolicyConfig(PreTrainedConfig):
        @property
        def observation_delta_indices(self):
            return None

        @property
        def action_delta_indices(self):
            return None

        @property
        def reward_delta_indices(self):
            return None

        def get_optimizer_preset(self):
            return None

        def get_scheduler_preset(self):
            return None

        def validate_features(self):
            pass

    # Create minimal config components
    dataset_cfg = DatasetConfig(repo_id="dummy")  # Will be overridden by LeRobotDataset
    mixture_cfg = DatasetMixtureConfig(datasets=[dataset_cfg], weights=[1.0])
    policy_cfg = DummyPolicyConfig()

    # Create the main config with minimal required parameters
    cfg = TrainPipelineConfig(
        dataset_mixture=mixture_cfg,
        policy=policy_cfg,
        resolution=(224, 224),
        num_cams=2,
        max_state_dim=32,
        max_action_dim=32,
        action_chunk=50,
    )

    return cfg


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
) -> None:
    """Convert a dataset from v2.0 to v2.1 format.

    Converts statistics from global format to per-episode format, updates
    codebase version, and pushes changes to the hub.

    Args:
        repo_id: Repository ID of the dataset to convert.
        branch: Git branch to push changes to. If None, uses default branch.
        num_workers: Number of worker threads for parallel statistics computation.
            Defaults to 4.
    """
    with SuppressWarnings():
        # Create fake config for the dataset
        cfg = create_fake_train_config()
        dataset = LeRobotDataset(cfg, repo_id, revision=V20, force_cache_sync=True)

    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if (dataset.root / STATS_PATH).is_file:
        (dataset.root / STATS_PATH).unlink()

    hub_api = HfApi()
    if hub_api.file_exists(
        repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(
            path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )

    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
