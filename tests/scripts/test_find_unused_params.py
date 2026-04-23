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

"""Unit tests for pure helpers in find_unused_params.

The actual audit loop (``find()``) requires a GPU and a real policy
build; it is exercised end-to-end via manual runs documented in
``docs/source/tutorials/benchmarking.rst`` and in issue #177.
"""

from __future__ import annotations

import pytest

from opentau.scripts.find_unused_params import _module_root


class TestModuleRoot:
    """Groups dotted parameter names by their leading components."""

    def test_default_depth_two(self):
        # Current implementation defaults depth=2.
        assert (
            _module_root("model.paligemma_with_expert.gemma_expert.lm_head.weight")
            == "model.paligemma_with_expert"
        )

    def test_depth_three(self):
        assert (
            _module_root(
                "model.paligemma_with_expert.gemma_expert.lm_head.weight",
                depth=3,
            )
            == "model.paligemma_with_expert.gemma_expert"
        )

    def test_depth_one(self):
        assert _module_root("model.paligemma_with_expert.gemma_expert.lm_head.weight", depth=1) == "model"

    def test_path_shorter_than_depth_returns_full_path(self):
        # Fewer components than requested depth → return the name as-is.
        assert _module_root("weight", depth=2) == "weight"
        assert _module_root("module.weight", depth=3) == "module.weight"

    def test_path_equal_to_depth_returns_full_path(self):
        # Exactly `depth` components → return all of them joined.
        assert _module_root("a.b.c", depth=3) == "a.b.c"

    def test_depth_zero_returns_empty_string(self):
        # Edge case: depth=0 means "take no components".
        # ".".join([]) is "", which is the expected behavior even if unusual.
        assert _module_root("a.b.c", depth=0) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
