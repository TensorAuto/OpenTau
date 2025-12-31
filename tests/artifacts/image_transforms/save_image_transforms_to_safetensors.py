#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.opentau.datasets.lerobot_dataset import LeRobotDataset
from src.opentau.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    make_transform_from_config,
)
from src.opentau.utils.random_utils import seeded_context

ARTIFACT_DIR = Path("tests/artifacts/image_transforms")
DATASET_REPO_ID = "lerobot/droid_100"


def save_default_config_transform(original_frame: torch.Tensor, output_dir: Path):
    cfg = ImageTransformsConfig(enable=True)
    default_tf = ImageTransforms(cfg)

    with seeded_context(1337):
        img_tf = default_tf(original_frame)

    save_file({"default": img_tf}, output_dir / "default_transforms.safetensors")
    print(f"Saved default transforms to {(output_dir / 'default_transforms.safetensors').resolve()}")


def save_single_transforms(original_frame: torch.Tensor, output_dir: Path):
    transforms = [
        ("ColorJitter", "brightness", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "contrast", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "saturation", [(0.5, 0.5), (2.0, 2.0)]),
        ("ColorJitter", "hue", [(-0.25, -0.25), (0.25, 0.25)]),
        ("SharpnessJitter", "sharpness", [(0.5, 0.5), (2.0, 2.0)]),
    ]

    frames = {"original_frame": original_frame}
    for tf_type, tf_name, min_max_values in transforms:
        for min_max in min_max_values:
            tf_cfg = ImageTransformConfig(type=tf_type, kwargs={tf_name: min_max})
            tf = make_transform_from_config(tf_cfg)
            key = f"{tf_name}_{min_max[0]}_{min_max[1]}"
            frames[key] = tf(original_frame)

    save_file(frames, output_dir / "single_transforms.safetensors")
    print(f"Saved single transforms to {(output_dir / 'single_transforms.safetensors').resolve()}")


def main():
    # Construct a minimal TrainPipelineConfig for the dataset
    from dataclasses import dataclass

    from src.opentau.configs.default import DatasetConfig, DatasetMixtureConfig
    from src.opentau.configs.policies import PreTrainedConfig
    from src.opentau.configs.train import TrainPipelineConfig

    # Minimal dummy PreTrainedConfig implementation for testing
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

    # Minimal DatasetConfig for the mixture
    dataset_cfg = DatasetConfig(repo_id=DATASET_REPO_ID)
    mixture_cfg = DatasetMixtureConfig(datasets=[dataset_cfg], weights=[1.0])
    policy_cfg = DummyPolicyConfig()
    cfg = TrainPipelineConfig(dataset_mixture=mixture_cfg, policy=policy_cfg)
    cfg.num_cams = 2

    dataset = LeRobotDataset(cfg, DATASET_REPO_ID, episodes=[0], image_transforms=None, standardize=False)
    output_dir = Path(ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_frame = dataset[0][dataset.meta.camera_keys[0]]

    save_single_transforms(original_frame, output_dir)
    save_default_config_transform(original_frame, output_dir)


if __name__ == "__main__":
    main()
