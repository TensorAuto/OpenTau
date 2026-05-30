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

import opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level as pg_ll_modeling
from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
    _warn_state_action_outliers,
)


class TestWarnStateActionOutliers:
    @pytest.fixture(autouse=True)
    def _clear_seen(self):
        # The warn-once dedup set is module-global and persists across the
        # process; clear it around every test so cases are order-independent.
        pg_ll_modeling._WARNED_OUTLIER_KEYS.clear()
        yield
        pg_ll_modeling._WARNED_OUTLIER_KEYS.clear()

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
