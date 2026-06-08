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

import logging

import pytest
import torch
from datasets import Dataset
from huggingface_hub import DatasetCard

from opentau.datasets import utils as datasets_utils
from opentau.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from opentau.datasets.utils import (
    check_version_compatibility,
    create_lerobot_dataset_card,
    hf_transform_to_torch,
)


def test_default_parameters():
    card = create_lerobot_dataset_card()
    assert isinstance(card, DatasetCard)
    assert card.data.tags == ["OpenTau"]
    assert card.data.task_categories == ["robotics"]
    assert card.data.configs == [
        {
            "config_name": "default",
            "data_files": "data/*/*.parquet",
        }
    ]


def test_with_tags():
    tags = ["tag1", "tag2"]
    card = create_lerobot_dataset_card(tags=tags)
    assert card.data.tags == ["OpenTau", "tag1", "tag2"]


def test_calculate_episode_data_index():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    assert torch.equal(episode_data_index["from"], torch.tensor([0, 2, 3]))
    assert torch.equal(episode_data_index["to"], torch.tensor([2, 3, 6]))


@pytest.fixture
def reset_v21_warning_state():
    """Reset module-global v2.0 warning dedup state around a test so a failed
    assert doesn't leak polluted state to later tests in the same worker."""
    datasets_utils._V21_WARNED_REPOS.clear()
    datasets_utils._V21_FULL_MESSAGE_SHOWN = False
    yield
    datasets_utils._V21_WARNED_REPOS.clear()
    datasets_utils._V21_FULL_MESSAGE_SHOWN = False


def test_v21_warning_dedup(caplog, reset_v21_warning_state):
    with caplog.at_level(logging.WARNING):
        check_version_compatibility("org/first", "2.0", "2.1")
        check_version_compatibility("org/second", "2.0", "2.1")
        check_version_compatibility("org/third", "2.0", "2.1")
        check_version_compatibility("org/first", "2.0", "2.1")  # dup: silent

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_messages) == 3, warning_messages

    # First message is the long-form V21_MESSAGE for the first repo (contains
    # the Discord help link which only appears in the long form).
    assert "org/first" in warning_messages[0]
    assert "discord.com/invite" in warning_messages[0]

    # Subsequent messages are the one-liner naming the repo and the convert script.
    for repo_id, msg in zip(["org/second", "org/third"], warning_messages[1:], strict=True):
        assert repo_id in msg
        assert "convert_dataset_v20_to_v21.py" in msg
        assert "discord.com/invite" not in msg


@pytest.fixture
def reset_dropped_column_warning_state():
    """Reset the warn-once dedup set for dropped non-tensorizable columns so the
    warning assertion is deterministic regardless of test ordering."""
    datasets_utils._DROPPED_NONTENSOR_COLUMNS_WARNED.clear()
    yield
    datasets_utils._DROPPED_NONTENSOR_COLUMNS_WARNED.clear()


def test_hf_transform_drops_nontensorizable_columns(caplog, reset_dropped_column_warning_state):
    """`hf_transform_to_torch` drops dict/struct columns it can't tensorize
    (e.g. chat-style `language_persistent` annotations) and warns once, while
    leaving numeric / string / empty-list columns intact."""
    items_dict = {
        "observation.state": [[1.0, 2.0, 3.0]],
        "task": ["close the fridge"],
        "language_events": [[]],  # null/empty list -> torch.tensor([]) -> kept
        "language_persistent": [[{"role": "user", "content": "hi"}]],  # struct -> dropped
    }
    with caplog.at_level(logging.WARNING):
        out = hf_transform_to_torch(items_dict)

    # The struct column is dropped; model-relevant columns survive unchanged.
    assert "language_persistent" not in out
    assert torch.equal(out["observation.state"][0], torch.tensor([1.0, 2.0, 3.0]))
    assert out["task"][0] == "close the fridge"
    assert isinstance(out["language_events"][0], torch.Tensor)
    assert out["language_events"][0].numel() == 0

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("language_persistent" in m for m in warnings), warnings

    # Warn-once: a second call with the same column does not re-warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hf_transform_to_torch({"language_persistent": [[{"a": 1}]], "observation.state": [[1.0]]})
    repeat = [
        r for r in caplog.records if r.levelno == logging.WARNING and "language_persistent" in r.getMessage()
    ]
    assert not repeat, "dropped-column warning should be emitted only once per column"


def test_hf_transform_raises_on_malformed_model_input():
    """A malformed (non-dict-like) numeric column still raises loudly rather than
    being silently dropped, so genuine data/shape bugs surface at their source."""
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        hf_transform_to_torch({"observation.state": [[[1.0, 2.0], [1.0, 2.0, 3.0]]]})
