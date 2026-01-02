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

import time
from unittest.mock import patch

import pytest

from opentau.utils.benchmark import TimeBenchmark


def test_context_manager_usage():
    """
    Tests the basic functionality as a context manager.
    """
    benchmark = TimeBenchmark()
    sleep_duration = 0.05

    with benchmark:
        time.sleep(sleep_duration)

    assert benchmark.result is not None
    assert isinstance(benchmark.result, float)
    # Check that the result is approximately the sleep duration
    assert benchmark.result == pytest.approx(sleep_duration, abs=0.01)


def test_decorator_usage():
    """
    Tests the functionality as a decorator.
    """
    benchmark = TimeBenchmark()
    sleep_duration = 0.5

    @benchmark
    def my_function():
        time.sleep(sleep_duration)

    my_function()

    assert benchmark.result is not None
    assert benchmark.result == pytest.approx(sleep_duration, abs=0.1)


def test_print_option():
    """
    Tests that the print functionality works when enabled.
    """

    with patch("builtins.print") as mock_print:
        benchmark = TimeBenchmark(print=True)

        with benchmark:
            time.sleep(0.01)

        mock_print.assert_called_once()
        # Check that the print call contains the expected text
        call_args, _ = mock_print.call_args
        assert "Elapsed time:" in call_args[0]
        assert "seconds" in call_args[0]


def test_exception_handling():
    """
    Tests that time is still measured even if an exception occurs.
    """
    benchmark = TimeBenchmark()
    sleep_duration = 0.05

    with pytest.raises(ValueError, match="Test exception"), benchmark:
        time.sleep(sleep_duration)
        raise ValueError("Test exception")

    # The exception should not prevent the result from being calculated
    assert benchmark.result is not None
    assert benchmark.result == pytest.approx(sleep_duration, abs=0.01)


def test_result_is_none_before_run():
    """
    Tests that results are None before the context is entered.
    """
    benchmark = TimeBenchmark()
    assert benchmark.result is None
    # assert benchmark.result_ms is None
