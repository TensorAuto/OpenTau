#!/usr/bin/env python

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

"""``actions_mse_loss.py`` must compare a prediction and its target in the same space.

A delta-action policy returns absolute joint targets while its dataset stores deltas, so scoring
one against the other measured the joint positions, not the policy. Every test drives a *perfect*
stub policy — one that returns the dataset's own chunk — so any nonzero error is the comparison's
fault rather than a model's.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from opentau.datasets.action_indexing import add_chunk_start_state
from opentau.datasets.dataset_mixture import _TaggedDataset
from opentau.scripts.actions_mse_loss import (
    action_fidelity,
    dataset_delta_action_state_map,
    first_action_pairs,
    inference_main,
)
from tests.datasets.test_action_dim import _fix_dummy_mapping, _make_dataset  # noqa: F401 (autouse fixture)

DOF = 4
DELTA_MAP = {0: 0, 1: 1, 2: 2}  # dim 3 (a gripper, say) stays absolute
CPU = torch.device("cpu")


@pytest.fixture
def dataset(
    lerobot_dataset_factory,
    info_factory,
    hf_dataset_factory,
    tasks_factory,
    episodes_factory,
    stats_factory,
    episodes_stats_factory,
    tmp_path,
):
    return _make_dataset(
        lerobot_dataset_factory,
        info_factory,
        hf_dataset_factory,
        tasks_factory,
        episodes_factory,
        stats_factory,
        episodes_stats_factory,
        tmp_path,
        real_action_dim=DOF,
        suffix="mse",
    )


def _perfect_policy(delta_map):
    """A policy that predicts the dataset's own chunk, returned as absolute targets — what
    ``PI05Policy.sample_actions`` does for a delta checkpoint by adding the batch state back."""

    def sample_actions(batch):
        chunk = batch["actions"].float()
        return add_chunk_start_state(chunk, batch["state"], delta_map) if delta_map else chunk

    return sample_actions


def test_delta_targets_are_compared_in_delta_space(dataset):
    dataset.delta_action_state_map = dict(DELTA_MAP)  # the dataset now emits deltas
    loader = DataLoader(dataset, batch_size=4)
    sample_actions = _perfect_policy(DELTA_MAP)

    pred, truth = first_action_pairs(sample_actions, loader, CPU, dataset_delta_action_state_map(dataset))

    absolute = np.concatenate([sample_actions(b)[:, 0, :DOF].numpy() for b in loader])
    assert np.abs(absolute - truth).max() > 0.1, "precondition: absolute vs delta must differ here"
    assert pred.shape == truth.shape == (len(dataset), DOF)
    # (d + s) - s is exact up to float32 rounding of the sum.
    np.testing.assert_allclose(pred, truth, rtol=0, atol=1e-6)


def test_absolute_targets_compare_the_first_action_of_every_sample(dataset):
    """One row per sample, over the pre-pad width — not sample 0's whole chunk against step 0."""
    loader = DataLoader(dataset, batch_size=4)

    pred, truth = first_action_pairs(
        _perfect_policy(None), loader, CPU, dataset_delta_action_state_map(dataset)
    )

    assert pred.shape == truth.shape == (len(dataset), DOF)
    np.testing.assert_array_equal(pred, truth)


def test_delta_map_is_read_through_the_mixture_wrappers(dataset):
    assert dataset_delta_action_state_map(dataset) is None
    dataset.delta_action_state_map = dict(DELTA_MAP)
    wrapped = _TaggedDataset(Subset(dataset, [0, 1]), "dummy/repo", 0)  # mixture + val split
    assert dataset_delta_action_state_map(wrapped) == DELTA_MAP


def test_r2_is_scored_against_the_ground_truth_variance():
    """``r2_score`` is asymmetric; with the arguments swapped this pair scores +0.3."""
    truth = np.array([[0.0], [1.0], [2.0], [3.0]])
    pred = 2 * truth
    mse, r2 = action_fidelity(pred, truth)
    np.testing.assert_allclose(mse, [3.5])
    np.testing.assert_allclose(r2, [1 - 14 / 5])


def test_entry_point_keeps_its_parser_wrap():
    """Helpers above ``@parser.wrap()`` must not have taken the decorator off the entry point."""
    assert getattr(inference_main, "__wrapped__", None) is not None
