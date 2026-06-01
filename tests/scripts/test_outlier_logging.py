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

"""CPU unit tests for the rank-0 outlier-record logger.

``log_outlier_records_distributed`` merges per-rank records, dedups across steps,
and emits the warning on rank 0. Driven here single-process with a real CPU
``Accelerator`` (the accelerate gathers are identity when not distributed), so the
merge / dedup / message-format logic is exercised without a GPU or torchrun.
"""

import logging

import accelerate
import pytest

from opentau.policies.outlier_utils import OutlierRecord
from opentau.scripts import train as train_mod
from opentau.scripts.train import log_outlier_records_distributed


@pytest.fixture(scope="module")
def accelerator():
    # Single-process CPU; gather_for_metrics / gather_object run their non-distributed (identity)
    # paths, so we exercise the real logger end-to-end without torchrun.
    return accelerate.Accelerator(cpu=True)


@pytest.fixture(autouse=True)
def _clear_seen():
    # The cross-step dedup map is module-global; clear it around every test so cases are
    # order-independent.
    train_mod._WARNED_OUTLIER_KEYS.clear()
    yield
    train_mod._WARNED_OUTLIER_KEYS.clear()


def _record(key="state", source=None, dim=3, value=50.0, episode=None, frame=None):
    return OutlierRecord(key=key, source=source, dim=dim, value=value, episode=episode, frame=frame)


def _outlier_msgs(caplog):
    return [r.getMessage() for r in caplog.records if "Outlier" in r.getMessage()]


class TestLogOutlierRecordsDistributed:
    def test_logs_message_verbatim(self, accelerator, caplog):
        records = [_record(key="state", source="dsB", dim=3, value=50.0, episode=7, frame=42)]
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, records, 10.0)
        msgs = _outlier_msgs(caplog)
        assert len(msgs) == 1
        m = msgs[0]
        assert "Outlier normalized state" in m
        assert "exceed |10.0|" in m
        assert "dims=[3]" in m
        assert "worst dim=3" in m
        assert "max=50.00" in m
        assert "source=dsB" in m
        assert "episode=7" in m
        assert "frame=42" in m
        assert "Re-warned only when" in m

    def test_empty_records_no_warning(self, accelerator, caplog):
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [], 10.0)
        assert not _outlier_msgs(caplog)

    def test_global_merge_reports_max(self, accelerator, caplog):
        # Same (source, key, dim) from two "ranks": the merge keeps the worst |value|.
        records = [
            _record(source="dsA", dim=2, value=40.0),
            _record(source="dsA", dim=2, value=88.0),
        ]
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, records, 10.0)
        msgs = _outlier_msgs(caplog)
        assert len(msgs) == 1
        assert "max=88.00" in msgs[0]
        assert "dims=[2]" in msgs[0]

    def test_state_and_action_warned_separately(self, accelerator, caplog):
        records = [
            _record(key="state", source="dsA", dim=1, value=33.0),
            _record(key="actions", source="dsA", dim=2, value=44.0),
        ]
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, records, 10.0)
        msgs = _outlier_msgs(caplog)
        assert any("Outlier normalized state" in m for m in msgs)
        assert any("Outlier normalized actions" in m for m in msgs)

    def test_warns_once_per_source_key_dim(self, accelerator, caplog):
        # The same (source, key, dim) offender on a later step is suppressed.
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=50.0)], 10.0)
        assert len(_outlier_msgs(caplog)) == 1
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=50.0)], 10.0)
        assert not _outlier_msgs(caplog)

    def test_fresh_offender_still_warns(self, accelerator, caplog):
        # A previously-unseen dim warns even after another dim was already seen.
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=50.0)], 10.0)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=5, value=77.0)], 10.0)
        msgs = _outlier_msgs(caplog)
        assert len(msgs) == 1
        assert "dims=[5]" in msgs[0]
        assert "dims=[3]" not in msgs[0]

    def test_rewarns_only_on_larger_magnitude(self, accelerator, caplog):
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=50.0)], 10.0)
        assert len(_outlier_msgs(caplog)) == 1
        caplog.clear()
        with caplog.at_level(logging.WARNING):  # larger than last warned (50) -> re-warns
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=120.0)], 10.0)
        msgs = _outlier_msgs(caplog)
        assert len(msgs) == 1 and "max=120.00" in msgs[0]
        caplog.clear()
        with caplog.at_level(logging.WARNING):  # smaller than last warned (120) -> suppressed
            log_outlier_records_distributed(accelerator, [_record(source="dsA", dim=3, value=60.0)], 10.0)
        assert not _outlier_msgs(caplog)

    def test_distinct_sources_each_warn(self, accelerator, caplog):
        # Same key/dim but different source are distinct offenders (global dedup is keyed on source).
        records = [
            _record(source="dsA", dim=3, value=50.0),
            _record(source="dsB", dim=3, value=51.0),
        ]
        with caplog.at_level(logging.WARNING):
            log_outlier_records_distributed(accelerator, records, 10.0)
        msgs = _outlier_msgs(caplog)
        # one state line, but it reports 2 new offenders for dim 3
        assert len(msgs) == 1
        assert "2 new/worsened" in msgs[0]
