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

"""Unit tests for pure helpers in profile_step.

The actual benchmark loop (``profile()``) is an integration test that
requires a GPU, a real policy, and ``accelerate launch``; it is
exercised end-to-end via the training pipeline that CI already covers
in ``.github/workflows/regression_test.yml``.
"""

from __future__ import annotations

import pytest

from opentau.scripts.profile_step import _fmt_ms


class TestFmtMs:
    def test_empty_list_returns_na(self):
        assert _fmt_ms([]) == "n/a"

    def test_single_value_gives_matching_stats(self):
        out = _fmt_ms([0.001])  # 1 ms
        # All three stats should be exactly 1 ms for a single-element list.
        assert "mean=   1.00ms" in out
        assert "median=   1.00ms" in out
        assert "p95=   1.00ms" in out

    def test_multiple_values_have_correct_mean_and_median(self):
        # Seconds → should render as ms with three summary stats.
        samples = [0.001, 0.002, 0.003, 0.004, 0.005]  # 1..5 ms
        out = _fmt_ms(samples)
        # mean((1,2,3,4,5)) = 3.0 ms
        assert "mean=   3.00ms" in out
        # median of 5 elements = 3.0 ms
        assert "median=   3.00ms" in out

    def test_p95_uses_index_based_formula(self):
        # p95 is computed as sorted[int(0.95 * (len - 1))]
        # For 20 elements (indices 0..19), int(0.95 * 19) = 18 → 19th smallest.
        samples = [i / 1000.0 for i in range(1, 21)]  # 1..20 ms
        out = _fmt_ms(samples)
        assert "p95=  19.00ms" in out

    def test_unsorted_input_is_handled(self):
        # _fmt_ms must sort before picking p95; give intentionally unsorted data.
        samples = [0.005, 0.001, 0.003, 0.004, 0.002]
        out = _fmt_ms(samples)
        # median should be middle of sorted: 3 ms
        assert "median=   3.00ms" in out

    def test_large_values_render_in_ms(self):
        # 2.5 seconds → 2500 ms
        out = _fmt_ms([2.5])
        assert "mean=2500.00ms" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
