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

"""Provenance fields on a VQA sample.

A VQA item has no episode/frame, so `_to_standard_data_format` must emit the
`-1` sentinel for `episode_index` / `frame_index` while still emitting a real
`source` (the dataset key). This is the branch that keeps a heterogeneous
VQA + LeRobot mixture schema-aligned under default collation, and it's the one
piece of the provenance change that the LeRobot-only tests don't exercise.

``train_pipeline_config`` comes from ``tests/fixtures/config_factory.py``.
"""

import torch

from opentau.datasets.vqa.dummy import DummyVQADataset


def test_vqa_emits_source_and_sentinel_indices(train_pipeline_config):
    """A VQA sample carries a real ``source`` but the ``-1`` sentinel for the
    ``episode_index`` / ``frame_index`` it lacks (plain ``int``, not tensor)."""
    ds = DummyVQADataset(train_pipeline_config)
    item = ds[0]
    assert item["source"] == "dummy"
    assert item["episode_index"] == -1
    assert item["frame_index"] == -1
    assert isinstance(item["episode_index"], int)
    assert isinstance(item["frame_index"], int)


def test_vqa_provenance_default_collates(train_pipeline_config):
    """The VQA provenance fields survive default collation across a batch:
    ``source`` -> ``list[str]``, the int sentinels -> a ``(B,)`` int64 tensor.
    This is exactly the alignment a VQA + LeRobot co-training batch relies on.
    """
    ds = DummyVQADataset(train_pipeline_config)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    assert isinstance(batch["source"], list)
    assert batch["source"] == ["dummy", "dummy"]
    assert batch["episode_index"].dtype == torch.int64
    assert batch["episode_index"].tolist() == [-1, -1]
    assert batch["frame_index"].tolist() == [-1, -1]
