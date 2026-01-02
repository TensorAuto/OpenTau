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

import pytest

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.transforms import ImageTransformsConfig
from opentau.policies.pi0.configuration_pi0 import PI0Config


@pytest.fixture
def transforms_config():
    """Return a mock ImageTransformsConfig object with minimal required attributes."""
    return ImageTransformsConfig(
        enable=False,
    )


@pytest.fixture
def policy_config():
    """Return a mock PolicyConfig object with minimal required attributes."""
    return PI0Config()


@pytest.fixture
def dataset_config(transforms_config):
    """Return a mock DatasetConfig object with minimal required attributes."""
    return DatasetConfig(
        repo_id="mock_dataset",
        root="/tmp/mock_dataset",
        image_transforms=transforms_config,
        episodes=[0],
        video_backend=None,
    )


@pytest.fixture
def dataset_mixture_config(dataset_config):
    """Return a mock DatasetMixtureConfig object with minimal required attributes."""
    return DatasetMixtureConfig(
        datasets=[dataset_config],
        weights=[1.0],
        action_freq=30.0,
        image_resample_strategy="nearest",
        vector_resample_strategy="nearest",
    )


@pytest.fixture
def train_pipeline_config(policy_config, dataset_mixture_config):
    """Return a mock TrainPipelineConfig object with minimal required attributes."""
    return TrainPipelineConfig(
        dataset_mixture=dataset_mixture_config,
        resolution=(64, 64),
        num_cams=1,
        policy=policy_config,
        max_state_dim=32,
        max_action_dim=32,
        action_chunk=50,
        batch_size=8,
    )
