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

"""Unit tests for pure helpers in profile_dataloader.

The actual benchmark loop (``profile()``) is an integration test that
requires a real LeRobot dataset on disk and ``accelerate launch``;
it is exercised end-to-end via the training pipeline CI in
``.github/workflows/regression_test.yml``.
"""

from __future__ import annotations

import pytest
import torch

from opentau.scripts.profile_dataloader import _batch_size_of, _to_device


class TestToDeviceCpuPassthrough:
    """``_to_device`` should be a no-op for CPU tensors.

    The implementation calls ``.to(device, non_blocking=True)`` only
    when ``device.type == 'cuda'``; CPU path returns the tensor as-is.
    """

    @property
    def cpu(self) -> torch.device:
        return torch.device("cpu")

    def test_single_tensor_is_unchanged(self):
        x = torch.arange(6).reshape(2, 3)
        out = _to_device(x, self.cpu)
        assert out is x  # same object (no copy)

    def test_dict_of_tensors_is_walked(self):
        batch = {
            "images": torch.zeros(2, 3),
            "labels": torch.arange(4),
        }
        out = _to_device(batch, self.cpu)
        assert set(out.keys()) == {"images", "labels"}
        assert out["images"] is batch["images"]
        assert out["labels"] is batch["labels"]

    def test_nested_dict_of_tensors(self):
        batch = {
            "obs": {"img0": torch.zeros(2, 3), "img1": torch.ones(2, 3)},
            "actions": torch.arange(4),
        }
        out = _to_device(batch, self.cpu)
        assert torch.equal(out["obs"]["img0"], batch["obs"]["img0"])
        assert torch.equal(out["obs"]["img1"], batch["obs"]["img1"])

    def test_list_of_tensors(self):
        batch = [torch.zeros(1), torch.ones(1)]
        out = _to_device(batch, self.cpu)
        assert isinstance(out, list)
        assert len(out) == 2
        assert torch.equal(out[0], batch[0])

    def test_tuple_of_tensors_preserves_type(self):
        batch = (torch.zeros(1), torch.ones(1))
        out = _to_device(batch, self.cpu)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_string_inside_collection_is_not_iterated(self):
        # Strings are Sequences too; _to_device must not recurse into them.
        batch = {"task": "pick_up_the_cube", "actions": torch.zeros(3)}
        out = _to_device(batch, self.cpu)
        assert out["task"] == "pick_up_the_cube"

    def test_scalar_passthrough(self):
        # Non-tensor, non-collection values must pass through unchanged.
        assert _to_device(42, self.cpu) == 42
        assert _to_device(None, self.cpu) is None


class TestBatchSizeOf:
    """``_batch_size_of`` returns the first dim of the first tensor found."""

    def test_tensor_returns_batch_dim(self):
        assert _batch_size_of(torch.zeros(5, 3, 4)) == 5

    def test_scalar_tensor_returns_none(self):
        # 0-D tensor has no batch dim.
        assert _batch_size_of(torch.tensor(3.14)) is None

    def test_dict_of_tensors_uses_first_matching(self):
        batch = {"images": torch.zeros(7, 3, 224, 224), "actions": torch.zeros(7, 10)}
        assert _batch_size_of(batch) == 7

    def test_nested_dict_recurses(self):
        batch = {"obs": {"img0": torch.zeros(4, 3, 224, 224)}, "actions": torch.zeros(4, 10)}
        assert _batch_size_of(batch) == 4

    def test_list_of_tensors(self):
        batch = [torch.zeros(3, 10), torch.zeros(3, 20)]
        assert _batch_size_of(batch) == 3

    def test_empty_dict_returns_none(self):
        assert _batch_size_of({}) is None

    def test_empty_list_returns_none(self):
        assert _batch_size_of([]) is None

    def test_dict_with_only_non_tensors_returns_none(self):
        assert _batch_size_of({"task": "foo", "scalar": 42}) is None

    def test_scalar_returns_none(self):
        assert _batch_size_of(42) is None
        assert _batch_size_of(None) is None
        assert _batch_size_of("a_string") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
