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

"""CPU unit tests for the shared state/action outlier detector.

``detect_state_action_outliers`` is pure tensor work (no logging, no cross-step
dedup, no collectives — see CLAUDE.md rule 5), so it's exercised directly on
constructed batches without building a policy or touching a GPU. The cross-rank
merge, dedup, and the actual ``logging.warning`` are tested separately in
``tests/scripts/test_outlier_logging.py``.
"""

import pytest
import torch

from opentau.policies.outlier_utils import detect_state_action_outliers


def _recs(records, key):
    return [r for r in records if r["key"] == key]


class TestDetectStateActionOutliers:
    def test_detects_above_threshold_state_only(self):
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][1, 3] = 50.0  # one outlier dim in the state
        records = detect_state_action_outliers(batch, 10.0)
        state_recs = _recs(records, "state")
        assert len(state_recs) == 1
        assert state_recs[0]["dim"] == 3
        assert state_recs[0]["value"] == pytest.approx(50.0)
        # actions are all zero -> never an offender.
        assert not _recs(records, "actions")

    def test_empty_below_threshold(self):
        batch = {"state": torch.full((2, 8), 5.0), "actions": torch.full((2, 4, 8), 9.9)}
        assert detect_state_action_outliers(batch, 10.0) == []

    @pytest.mark.parametrize("threshold", [0.0, -1.0, None])
    def test_disabled_threshold_returns_empty(self, threshold):
        # A huge value must NOT be reported when the check is disabled.
        batch = {"state": torch.full((2, 8), 1e3), "actions": torch.zeros(2, 4, 8)}
        assert detect_state_action_outliers(batch, threshold) == []

    def test_records_source_episode_frame(self):
        batch = {
            "state": torch.zeros(2, 8),
            "actions": torch.zeros(2, 4, 8),
            "source": ["dsA", "dsB"],
            "episode_index": torch.tensor([0, 7]),
            "frame_index": torch.tensor([0, 42]),
        }
        batch["actions"][1, 2, 5] = 99.0  # outlier in sample 1
        records = detect_state_action_outliers(batch, 10.0)
        action_recs = _recs(records, "actions")
        assert len(action_recs) == 1
        rec = action_recs[0]
        assert rec["dim"] == 5
        assert rec["source"] == "dsB"
        assert rec["episode"] == 7
        assert rec["frame"] == 42

    def test_missing_provenance_is_none(self):
        # No source/episode/frame keys: must still report (with None placeholders)
        # and never raise -- protects callers/tests that omit them.
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][0, 0] = 100.0
        records = detect_state_action_outliers(batch, 10.0)
        assert len(records) == 1
        assert records[0]["source"] is None
        assert records[0]["episode"] is None
        assert records[0]["frame"] is None

    @pytest.mark.parametrize("state_shape", [(2, 8), (2, 3, 8)])
    def test_variable_ndim_state(self, state_shape):
        # The `b ... d` reduction must collapse the optional middle (time) axis
        # for both a 2-D state and a 3-D state-history tensor.
        batch = {"state": torch.zeros(*state_shape)}
        if len(state_shape) == 3:
            batch["state"][1, 2, 7] = 30.0  # current (last) slot, dim 7
        else:
            batch["state"][1, 7] = 30.0
        records = detect_state_action_outliers(batch, 10.0)
        assert any(r["key"] == "state" and r["dim"] == 7 for r in records)

    # -- pad-awareness: only timesteps the model attends to can trip ------------

    def test_masked_history_slot_not_detected(self):
        # A huge value in a PADDED history slot must NOT trip (it's masked out of attention).
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.tensor([[True, True, False], [True, True, False]]),
        }
        batch["state"][1, 0, 5] = 1e6  # slot 0 is padded for sample 1
        assert detect_state_action_outliers(batch, 10.0) == []

    def test_current_slot_detected_when_all_padded(self):
        # history_state_drop marks every step padded, but the current (last) frame is always
        # attended (state_mask[:, -1] = True) and must still be scanned.
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.ones(2, 3, dtype=torch.bool),
        }
        batch["state"][1, 2, 4] = 50.0  # current (last) slot for sample 1
        records = detect_state_action_outliers(batch, 10.0)
        assert any(r["key"] == "state" and r["dim"] == 4 for r in records)

    def test_unpadded_history_is_scanned(self):
        # With a mask present but nothing padded, a real (non-padded) history slot still trips.
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        }
        batch["state"][0, 1, 6] = 40.0  # a middle history slot, real (not padded)
        records = detect_state_action_outliers(batch, 10.0)
        assert any(r["key"] == "state" and r["dim"] == 6 for r in records)

    def test_padded_action_step_not_detected(self):
        # A huge value in a PADDED action step (action_is_pad) must NOT trip.
        batch = {
            "actions": torch.zeros(2, 4, 8),
            "action_is_pad": torch.tensor([[False, False, True, True], [False, False, True, True]]),
        }
        batch["actions"][0, 3, 2] = 99.0  # step 3 is padded
        assert detect_state_action_outliers(batch, 10.0) == []

    def test_no_mask_3d_state_scans_current_frame_only(self):
        # With no obs_history_is_pad, the model attends the current (last) frame only, so a value
        # in a non-current history slot is not scanned, while one in the current frame still trips.
        batch = {"state": torch.zeros(2, 3, 8)}
        batch["state"][0, 0, 4] = 1e6  # history slot 0 (not current), no pad mask
        assert detect_state_action_outliers(batch, 10.0) == []
        batch["state"][0, 0, 4] = 0.0
        batch["state"][1, 2, 6] = 40.0  # current (last) slot -> still trips
        records = detect_state_action_outliers(batch, 10.0)
        assert any(r["key"] == "state" and r["dim"] == 6 for r in records)

    def test_worst_value_per_source_key_dim(self):
        # Several samples from the same source hit the same dim: one record carrying the worst.
        batch = {
            "state": torch.zeros(3, 8),
            "source": ["dsA", "dsA", "dsA"],
        }
        batch["state"][0, 2] = 40.0
        batch["state"][1, 2] = 88.0  # worst for (dsA, state, 2)
        batch["state"][2, 2] = 55.0
        records = detect_state_action_outliers(batch, 10.0)
        dim2 = [r for r in records if r["key"] == "state" and r["dim"] == 2]
        assert len(dim2) == 1
        assert dim2[0]["value"] == pytest.approx(88.0)
