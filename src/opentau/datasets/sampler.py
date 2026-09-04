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
"""Episode-aware sampler for PyTorch DataLoader.

This module provides a sampler that respects episode boundaries in robot
learning datasets. It allows filtering specific episodes, dropping frames
from episode boundaries, and optional shuffling while maintaining episode
structure awareness.

The sampler is designed for use with PyTorch DataLoader to ensure proper
sampling behavior when working with sequential episode data, where episode
boundaries are important for maintaining temporal coherence or avoiding
invalid transitions.

Key Features:
    - Episode filtering: Select specific episodes to include in sampling.
    - Boundary frame dropping: Optionally drop frames from the start or end
      of episodes (useful for avoiding invalid transitions or edge cases).
    - Optional shuffling: Shuffle indices while maintaining episode awareness.
    - PyTorch compatible: Implements the standard Sampler interface for use
      with DataLoader.

Classes:
    EpisodeAwareSampler: PyTorch-style sampler that respects episode
        boundaries, supports episode filtering, frame dropping, and shuffling.

Example:
    Create a sampler for specific episodes:
        >>> episode_data_index = {"from": [0, 100, 200], "to": [99, 199, 299]}
        >>> sampler = EpisodeAwareSampler(
        ...     episode_data_index,
        ...     episode_indices_to_use=[0, 2],  # Use episodes 0 and 2
        ...     drop_n_first_frames=5,
        ...     drop_n_last_frames=5,
        ...     shuffle=True
        ... )
        >>> dataloader = DataLoader(dataset, sampler=sampler)
"""

from typing import Iterator, Optional, Union

import torch


class EpisodeAwareSampler:
    """Sampler that optionally incorporates episode boundary information.

    When ``shuffle`` is enabled, permutations are generated from a dedicated
    :class:`torch.Generator`.  Keeping the sampler's random stream separate
    from PyTorch's process-local global generator is important for distributed
    training: ``set_seed`` intentionally offsets the global stream on each
    rank, while ``accelerate`` shards a sampler by having every rank walk the
    same sequence.  The optional ``seed`` therefore always refers to the
    shared, rank-independent stream.

    ``seed`` takes precedence over ``generator`` when both are supplied.  It
    is applied to the supplied generator in place, matching
    :class:`~opentau.datasets.dataset_mixture.HierarchicalSampler`.  When
    neither is supplied, a private default ``torch.Generator`` is used rather
    than the global RNG. PyTorch currently initializes a bare generator with a
    fixed constant, so this fallback produces the same shuffle across launches
    and processes. That is an implementation detail rather than a documented
    PyTorch guarantee; callers that require an explicitly controlled stream
    should pass ``seed`` or ``generator``.
    """

    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        """Build an episode-aware sampler.

        Args:
            episode_data_index: Dictionary with keys ``"from"`` and ``"to"``
                containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None,
                all episodes are used. Assumes that episodes are indexed from
                0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of
                each episode.
            drop_n_last_frames: Number of frames to drop from the end of each
                episode.
            shuffle: Whether to shuffle the indices.
            generator: Optional private PyTorch generator used for shuffling.
                It is never inferred from the process-global RNG.
            seed: Optional seed for the sampler generator. If both ``seed``
                and ``generator`` are supplied, ``seed`` re-seeds the supplied
                generator in place.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle
        self._gen = generator if generator is not None else torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices), generator=self._gen):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
