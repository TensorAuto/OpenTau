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
from unittest.mock import MagicMock

import torch
from datasets import Dataset

from opentau.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from opentau.datasets.sampler import EpisodeAwareSampler
from opentau.datasets.utils import (
    hf_transform_to_torch,
)
from opentau.utils.random_utils import seeded_context, set_seed


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, episode_indices_to_use=[0, 2])
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index, shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index, shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}


def _episode_data_index():
    """Small multi-episode index used by the sampler RNG tests."""
    return {
        "from": torch.tensor([0, 4, 9]),
        "to": torch.tensor([4, 9, 15]),
    }


def test_seeded_shuffle_is_replayable_and_seed_sensitive():
    """A sampler seed must determine the permutation, not global RNG history."""
    data_index = _episode_data_index()
    first = list(EpisodeAwareSampler(data_index, shuffle=True, seed=1234))
    replay = list(EpisodeAwareSampler(data_index, shuffle=True, seed=1234))
    different = list(EpisodeAwareSampler(data_index, shuffle=True, seed=5678))

    assert first == replay
    assert first != different


def test_shuffle_does_not_read_rank_dependent_global_rng():
    """Every simulated rank must emit the same seeded episode stream.

    ``set_seed`` offsets the process-global torch RNG by rank to decorrelate
    per-sample augmentation draws.  The sampler must use its own generator so
    that this rank-local offset cannot change the shared data order.
    """
    streams = []
    for rank in range(4):
        with seeded_context(2026):
            set_seed(1000, accelerator=MagicMock(process_index=rank))
            streams.append(list(EpisodeAwareSampler(_episode_data_index(), shuffle=True, seed=1000)))

    assert all(stream == streams[0] for stream in streams[1:])


def test_shuffle_keeps_global_rng_state_untouched():
    """Constructing and iterating a sampler must not consume global draws."""
    torch.manual_seed(2026)
    before = torch.get_rng_state()
    sampler = EpisodeAwareSampler(_episode_data_index(), shuffle=True, seed=9)
    list(sampler)
    after = torch.get_rng_state()

    assert torch.equal(after, before)


def test_explicit_generator_and_seed_contract():
    """The explicit generator is supported and ``seed`` takes precedence."""
    generator = torch.Generator().manual_seed(7)
    sampler = EpisodeAwareSampler(_episode_data_index(), shuffle=True, generator=generator, seed=1234)

    assert list(sampler) == list(EpisodeAwareSampler(_episode_data_index(), shuffle=True, seed=1234))
    assert generator.initial_seed() == 1234


def test_unseeded_shuffle_uses_private_generator():
    """Pin the intentional, though implementation-defined, fixed fallback."""
    data_index = _episode_data_index()
    torch.manual_seed(1)
    first_state = torch.get_rng_state()
    first = list(EpisodeAwareSampler(data_index, shuffle=True))
    first_after = torch.get_rng_state()
    torch.manual_seed(999)
    second_state = torch.get_rng_state()
    second = list(EpisodeAwareSampler(data_index, shuffle=True))
    second_after = torch.get_rng_state()

    assert first == second
    assert torch.equal(first_state, first_after)
    assert torch.equal(second_state, second_after)
