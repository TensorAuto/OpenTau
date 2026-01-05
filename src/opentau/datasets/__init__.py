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
"""Dataset management and processing utilities for robot learning and vision-language tasks.

This module provides a comprehensive toolkit for loading, creating, managing, and
processing datasets for training vision-language-action (VLA) models. It supports
both robot learning datasets (with actions and states) and vision-language
grounding datasets (for multimodal understanding tasks).

The module is organized into several key components:

    - **Core Datasets**: LeRobotDataset for robot learning data with support for
      temporal alignment, multi-modal data, and version compatibility.
    - **Grounding Datasets**: Vision-language datasets (CLEVR, COCO-QA, PIXMO, VSR)
      for training visual understanding without robot actions.
    - **Dataset Mixtures**: WeightedDatasetMixture for combining multiple datasets
      with controlled sampling proportions.
    - **Data Processing**: Utilities for statistics computation, image/video
      handling, transforms, and format standardization.
    - **Factory Functions**: High-level functions for creating datasets and mixtures
      from configuration objects.

Key Features:

    - **HuggingFace Integration**: Seamless loading from HuggingFace Hub with
      automatic version checking and backward compatibility.
    - **Temporal Alignment**: Delta timestamps enable sampling features at
      different time offsets with optional Gaussian noise for data augmentation.
    - **Multi-modal Support**: Handles images, videos, state vectors, actions,
      and text prompts with automatic format conversion.
    - **Weighted Sampling**: Combine heterogeneous datasets with configurable
      sampling weights for balanced training.
    - **Standard Data Format**: Unified data format across all datasets for
      consistent model input/output interfaces.
    - **Statistics Management**: Automatic computation and aggregation of dataset
      statistics for normalization.
    - **Video Handling**: Multiple video backends (torchcodec, pyav, video_reader)
      for efficient frame extraction and encoding.
    - **Asynchronous I/O**: High-performance image writing for real-time data
      recording without blocking.

Main Modules:

    - **lerobot_dataset**: Core dataset implementation for robot learning data.
    - **grounding**: Vision-language grounding datasets (CLEVR, COCO-QA, PIXMO, VSR).
    - **dataset_mixture**: Weighted combination of multiple datasets.
    - **factory**: Factory functions for creating datasets from configurations.
    - **utils**: Utility functions for I/O, metadata management, and validation.
    - **compute_stats**: Statistics computation and aggregation utilities.
    - **transforms**: Image transformation pipelines for data augmentation.
    - **video_utils**: Video encoding, decoding, and metadata extraction.
    - **image_writer**: Asynchronous image writing for high-frequency recording.
    - **sampler**: Episode-aware sampling with boundary frame filtering.
    - **standard_data_format_mapping**: Feature name and loss type mappings.

Example:
    Create a dataset mixture from configuration:

        >>> from opentau.datasets.factory import make_dataset_mixture
        >>> mixture = make_dataset_mixture(train_cfg)
        >>> dataloader = mixture.get_dataloader()

    Load a single dataset:

        >>> from opentau.datasets.factory import make_dataset
        >>> dataset = make_dataset(dataset_cfg, train_cfg)

    Access grounding datasets:

        >>> from opentau import available_grounding_datasets
        >>> print(list(available_grounding_datasets.keys()))
        ['clevr', 'cocoqa', 'dummy', 'pixmo', 'vsr']
"""
