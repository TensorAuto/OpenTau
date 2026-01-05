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

"""Vision-language grounding datasets for multimodal learning.

This module provides datasets for training vision-language-action models on
image-text grounding tasks without requiring robot actions. Grounding datasets
are designed to help models learn visual understanding, spatial reasoning,
and language grounding capabilities that can be transferred to robotic tasks.

Grounding datasets differ from standard robot learning datasets in that they:
    - Provide images, prompts, and responses but no robot actions or states
    - Use zero-padding for state and action features to maintain compatibility
    - Focus on visual question answering, spatial reasoning, and object grounding
    - Enable training on large-scale vision-language data without robot hardware

The module uses a registration system where datasets are registered via the
`@register_grounding_dataset` decorator, making them available through the
`available_grounding_datasets` registry.

Available Datasets:
    - CLEVR: Compositional Language and Elementary Visual Reasoning dataset
        for visual question answering with synthetic scenes.
    - COCO-QA: Visual question answering dataset based on COCO images,
        filtered for spatial reasoning tasks.
    - PIXMO: Pixel-level manipulation grounding dataset for object
        localization and manipulation tasks.
    - VSR: Visual Spatial Reasoning dataset for true/false statement
        grounding about spatial relationships in images.
    - dummy: Synthetic test dataset with simple black, white, and gray
        images for testing infrastructure.

Classes:
    GroundingDataset: Base class for all grounding datasets, providing
        common functionality for metadata creation, data format conversion,
        and zero-padding of state/action features.

Modules:
    base: Base class and common functionality for grounding datasets.
    clevr: CLEVR dataset implementation.
    cocoqa: COCO-QA dataset implementation.
    dummy: Dummy test dataset implementation.
    pixmo: PIXMO dataset implementation.
    vsr: VSR dataset implementation.

Example:
    Use a grounding dataset in training configuration:
        >>> from opentau.configs.default import DatasetConfig
        >>> cfg = DatasetConfig(grounding="cocoqa")
        >>> dataset = make_dataset(cfg, train_cfg)

    Access available grounding datasets:
        >>> from opentau import available_grounding_datasets
        >>> print(list(available_grounding_datasets.keys()))
        ['clevr', 'cocoqa', 'dummy', 'pixmo', 'vsr']
"""
