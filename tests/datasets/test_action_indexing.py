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

"""Column selection and delta-action transforms (`opentau.datasets.action_indexing`)."""

import logging

import pytest
import torch

from opentau.configs.default import DatasetConfig
from opentau.datasets.action_indexing import (
    add_chunk_start_state,
    apply_column_index,
    resolve_delta_map,
    subtract_chunk_start_state,
)


class TestApplyColumnIndex:
    def test_none_is_identity(self):
        x = torch.arange(12.0).reshape(3, 4)
        assert apply_column_index(x, None, what="state", who="ds") is x

    def test_subsets_and_reorders(self):
        """The index both selects and permutes — order is respected, not sorted."""
        x = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
        out = apply_column_index(x, [3, 0, 2], what="state", who="ds")
        assert out.tolist() == [[13.0, 10.0, 12.0]]

    def test_applies_to_last_axis_of_a_chunk(self):
        x = torch.arange(2 * 4.0).reshape(2, 4)  # (chunk=2, dim=4)
        out = apply_column_index(x, [1, 0], what="action", who="ds")
        assert out.tolist() == [[1.0, 0.0], [5.0, 4.0]]

    def test_out_of_range_raises_with_offending_columns(self):
        x = torch.zeros(1, 3)
        with pytest.raises(IndexError, match=r"\[7\]"):
            apply_column_index(x, [0, 7], what="state", who="ds")


class TestResolveDeltaMap:
    def test_identity_when_no_indices(self):
        assert resolve_delta_map({0: 0, 1: 1}, None, None, who="ds") == {0: 0, 1: 1}

    def test_translates_both_sides_into_post_index_space(self):
        """Keys become action positions, values become state positions."""
        # action_index=[2,0,1,7] -> parquet col 0 is at position 1, col 1 at 2, col 2 at 0.
        # state_index=[5,3,1]    -> parquet col 5 is at position 0, col 3 at 1, col 1 at 2.
        got = resolve_delta_map({0: 5, 1: 3, 2: 1}, [2, 0, 1, 7], [5, 3, 1], who="ds")
        assert got == {1: 0, 2: 1, 0: 2}

    def test_mapped_action_column_dropped_by_index_raises(self):
        with pytest.raises(ValueError, match="action_index"):
            resolve_delta_map({5: 0}, [0, 1, 2], None, who="ds")

    def test_mapped_state_column_dropped_by_index_raises(self):
        with pytest.raises(ValueError, match="state_index"):
            resolve_delta_map({0: 9}, None, [0, 1, 2], who="ds")

    def test_warns_about_kept_but_unmapped_action_dims(self, caplog):
        """An omitted dim trains absolute; that must be visible, not silent."""
        with caplog.at_level(logging.WARNING):
            resolve_delta_map({0: 0, 1: 1}, None, None, who="ds", action_dim=4)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "ABSOLUTE" in msg and "[2, 3]" in msg

    def test_no_warning_when_every_dim_is_mapped(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_delta_map({0: 0, 1: 1}, None, None, who="ds", action_dim=2)
        assert not any("ABSOLUTE" in r.getMessage() for r in caplog.records)


class TestDeltaTransform:
    def test_subtracts_one_chunk_start_state_across_the_whole_horizon(self):
        """The defining property: every element of the chunk sees the SAME state.

        A per-step delta would instead give a constant column here, since the actions step
        uniformly. Getting this wrong changes the target distribution completely.
        """
        actions = torch.tensor([[0.0], [1.0], [2.0], [3.0]])  # (chunk=4, dim=1)
        state = torch.tensor([10.0])
        out = subtract_chunk_start_state(actions, state, {0: 0})
        assert out.squeeze(-1).tolist() == [-10.0, -9.0, -8.0, -7.0]

    def test_only_mapped_dims_become_relative(self):
        actions = torch.tensor([[5.0, 5.0]])
        state = torch.tensor([1.0, 2.0])
        out = subtract_chunk_start_state(actions, state, {0: 0})
        assert out.tolist() == [[4.0, 5.0]]  # dim 1 untouched (absolute gripper case)

    def test_empty_map_is_identity(self):
        actions = torch.tensor([[5.0, 5.0]])
        assert torch.equal(subtract_chunk_start_state(actions, torch.zeros(2), {}), actions)

    def test_does_not_mutate_input(self):
        actions = torch.tensor([[5.0]])
        original = actions.clone()
        subtract_chunk_start_state(actions, torch.tensor([1.0]), {0: 0})
        assert torch.equal(actions, original)

    @pytest.mark.parametrize(
        "actions_shape,state_shape",
        [
            ((4, 3), (3,)),  # unbatched, no history
            ((4, 3), (2, 3)),  # unbatched, history axis -> last step used
            ((2, 4, 3), (2, 3)),  # batched, no history
            ((2, 4, 3), (2, 5, 3)),  # batched, history axis
        ],
    )
    def test_round_trip_is_exact_for_every_supported_shape(self, actions_shape, state_shape):
        """`add(subtract(x)) == x` — the forward/inverse pair openpi uses at train/serve time."""
        torch.manual_seed(0)
        actions = torch.randn(actions_shape)
        state = torch.randn(state_shape)
        delta_map = {0: 2, 2: 0}
        deltas = subtract_chunk_start_state(actions, state, delta_map)
        assert not torch.equal(deltas, actions)  # the transform actually did something
        torch.testing.assert_close(add_chunk_start_state(deltas, state, delta_map), actions)

    def test_history_axis_uses_the_current_step(self):
        """State history resolves to the LAST step, which is what the chunk is anchored to."""
        actions = torch.tensor([[7.0]])
        state = torch.tensor([[100.0], [3.0]])  # (T=2, dim=1); current step is 3.0
        out = subtract_chunk_start_state(actions, state, {0: 0})
        assert out.tolist() == [[4.0]]


class TestDatasetConfigValidation:
    def _cfg(self, **kw):
        return DatasetConfig(repo_id="org/ds", **kw)

    def test_flag_without_map_raises(self):
        with pytest.raises(ValueError, match="requires a non-empty"):
            self._cfg(use_delta_joint_actions=True)

    def test_map_without_flag_raises(self):
        """Silently ignoring the map would train absolute actions while the config says delta."""
        with pytest.raises(ValueError, match="use_delta_joint_actions"):
            self._cfg(delta_action_state_map={0: 0})

    def test_json_string_keys_are_coerced_to_int(self):
        """A JSON config round-trips object keys as strings; uncoerced they'd match nothing."""
        cfg = self._cfg(use_delta_joint_actions=True, delta_action_state_map={"0": 1, "2": "3"})
        assert cfg.delta_action_state_map == {0: 1, 2: 3}

    def test_duplicate_index_entries_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            self._cfg(state_index=[0, 1, 1])

    def test_empty_index_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self._cfg(action_index=[])

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._cfg(state_index=[0, -1])

    def test_index_rejected_on_vqa_entries(self):
        with pytest.raises(ValueError, match="LeRobot"):
            DatasetConfig(vqa="clevr", state_index=[0])

    def test_defaults_are_inert(self):
        cfg = self._cfg()
        assert cfg.state_index is None
        assert cfg.action_index is None
        assert cfg.delta_action_state_map is None
        assert cfg.use_delta_joint_actions is False
