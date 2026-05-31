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

"""CPU unit tests for the pi07_paligemma low-level outlier warning.

``_warn_state_action_outliers`` is pure tensor + logging, so it's exercised
directly on constructed batches without building the policy or touching a GPU.
"""

import logging

import pytest
import torch

from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
    _WARNED_OUTLIER_KEYS,
    _warn_state_action_outliers,
)


class TestWarnStateActionOutliers:
    @pytest.fixture(autouse=True)
    def _clear_seen(self):
        # The warn-once dedup set is module-global and persists across the
        # process; clear it around every test so cases are order-independent.
        _WARNED_OUTLIER_KEYS.clear()
        yield
        _WARNED_OUTLIER_KEYS.clear()

    def test_warns_above_threshold_state_only(self, caplog):
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][1, 3] = 50.0  # one outlier dim in the state
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert any("state" in m and "dims=[3]" in m and "50" in m for m in msgs)
        # actions are all zero -> never warned.
        assert not any("actions" in m for m in msgs)

    def test_silent_below_threshold(self, caplog):
        batch = {"state": torch.full((2, 8), 5.0), "actions": torch.full((2, 4, 8), 9.9)}
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, None])
    def test_disabled_threshold_is_silent(self, caplog, threshold):
        # A huge value must NOT warn when the check is disabled.
        batch = {"state": torch.full((2, 8), 1e3), "actions": torch.zeros(2, 4, 8)}
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, threshold)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    def test_names_source_episode_frame(self, caplog):
        batch = {
            "state": torch.zeros(2, 8),
            "actions": torch.zeros(2, 4, 8),
            "source": ["dsA", "dsB"],
            "episode_index": torch.tensor([0, 7]),
            "frame_index": torch.tensor([0, 42]),
        }
        batch["actions"][1, 2, 5] = 99.0  # outlier in sample 1
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msg = "\n".join(r.getMessage() for r in caplog.records)
        assert "actions" in msg
        assert "dims=[5]" in msg
        assert "source=dsB" in msg
        assert "episode=7" in msg
        assert "frame=42" in msg

    def test_missing_provenance_keys_tolerated(self, caplog):
        # No source/episode/frame keys: must still warn (with None placeholders)
        # and never raise -- protects existing/test callers that omit them.
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][0, 0] = 100.0
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msg = "\n".join(r.getMessage() for r in caplog.records)
        assert "source=None" in msg
        assert "episode=None" in msg
        assert "frame=None" in msg

    @pytest.mark.parametrize("state_shape", [(2, 8), (2, 3, 8)])
    def test_variable_ndim_state(self, caplog, state_shape):
        # The `b ... d` reduction must collapse the optional middle (time) axis
        # for both a 2-D state and a 3-D state-history tensor.
        batch = {"state": torch.zeros(*state_shape)}
        if len(state_shape) == 3:
            batch["state"][1, 2, 7] = 30.0  # outlier in a middle-T slot
        else:
            batch["state"][1, 7] = 30.0
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msg = "\n".join(r.getMessage() for r in caplog.records if "Outlier" in r.getMessage())
        assert "state" in msg
        assert "dims=[7]" in msg

    def test_warns_once_per_source_key_dim(self, caplog):
        # The same (source, key, dim) offender on a later step is suppressed.
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][1, 3] = 50.0
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert sum("Outlier" in r.getMessage() for r in caplog.records) == 1
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    def test_fresh_offender_still_warns(self, caplog):
        # A previously-unseen dim warns even after another dim was already seen.
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][1, 3] = 50.0
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        caplog.clear()
        batch["state"][1, 3] = 0.0  # old offender resolved
        batch["state"][0, 5] = 77.0  # new offender appears
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert len(msgs) == 1
        assert "dims=[5]" in msgs[0]
        assert "dims=[3]" not in msgs[0]

    # -- pad-awareness: only timesteps the model attends to can trip ------------

    def test_masked_history_slot_does_not_warn(self, caplog):
        # A huge value in a PADDED history slot must NOT warn (it's masked out of attention).
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.tensor([[True, True, False], [True, True, False]]),
        }
        batch["state"][1, 0, 5] = 1e6  # slot 0 is padded for sample 1
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    def test_current_slot_warns_even_when_all_padded(self, caplog):
        # history_state_drop marks every step padded, but the current (last) frame is always
        # attended (state_mask[:, -1] = True) and must still be scanned.
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.ones(2, 3, dtype=torch.bool),
        }
        batch["state"][1, 2, 4] = 50.0  # current (last) slot for sample 1
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert any("state" in m and "dims=[4]" in m for m in msgs)

    def test_unpadded_history_is_still_scanned(self, caplog):
        # With a mask present but nothing padded, a real (non-padded) history slot still trips.
        batch = {
            "state": torch.zeros(2, 3, 8),
            "obs_history_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        }
        batch["state"][0, 1, 6] = 40.0  # a middle history slot, real (not padded)
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert any("state" in m and "dims=[6]" in m for m in msgs)

    def test_padded_action_step_does_not_warn(self, caplog):
        # A huge value in a PADDED action step (action_is_pad) must NOT warn.
        batch = {
            "actions": torch.zeros(2, 4, 8),
            "action_is_pad": torch.tensor([[False, False, True, True], [False, False, True, True]]),
        }
        batch["actions"][0, 3, 2] = 99.0  # step 3 is padded
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    # -- re-warn on a larger magnitude, not just the first occurrence ------------

    def test_rewarns_only_on_larger_magnitude(self, caplog):
        batch = {"state": torch.zeros(2, 8), "actions": torch.zeros(2, 4, 8)}
        batch["state"][1, 3] = 50.0
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert sum("Outlier" in r.getMessage() for r in caplog.records) == 1
        caplog.clear()
        batch["state"][1, 3] = 120.0  # larger than the last warned (50) -> re-warns
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert len(msgs) == 1 and "max=120.00" in msgs[0]
        caplog.clear()
        batch["state"][1, 3] = 60.0  # smaller than the last warned (120) -> suppressed
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)

    def test_no_mask_3d_state_scans_current_frame_only(self, caplog):
        # With no obs_history_is_pad, the model attends the current (last) frame only, so a value
        # in a non-current history slot is not scanned, while one in the current frame still warns.
        batch = {"state": torch.zeros(2, 3, 8)}
        batch["state"][0, 0, 4] = 1e6  # history slot 0 (not current), no pad mask
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        assert not any("Outlier" in r.getMessage() for r in caplog.records)
        caplog.clear()
        batch["state"][0, 0, 4] = 0.0
        batch["state"][1, 2, 6] = 40.0  # current (last) slot -> still warns
        with caplog.at_level(logging.WARNING):
            _warn_state_action_outliers(batch, 10.0)
        msgs = [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]
        assert any("state" in m and "dims=[6]" in m for m in msgs)
