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

"""Tests for zero-weight loss-term skipping.

When a ``loss_weighting`` entry is exactly 0 the corresponding term is dropped
from the backward pass entirely (``_assemble_weighted_loss``), so autograd never
traverses that subgraph and the parameters that feed only that term receive no
gradient. This differs from the old ``0 * term`` behavior, which still
backpropagated zeros into those parameters. These CPU tests pin both the
gradient-flow contract and the config validation.
"""

import pytest
import torch

from opentau.configs.train import TrainPipelineConfig
from opentau.scripts.train import _assemble_weighted_loss


def _leaf(value: float) -> torch.Tensor:
    return torch.tensor(value, requires_grad=True)


class TestAssembleWeightedLoss:
    def test_both_terms_included_matches_manual_sum(self):
        mse, ce = _leaf(2.0), _leaf(3.0)
        loss = _assemble_weighted_loss({"MSE": mse, "CE": ce}, {"MSE": 1.0, "CE": 0.5})
        assert loss.item() == pytest.approx(1.0 * 2.0 + 0.5 * 3.0)
        loss.backward()
        # Both terms are in the graph -> both leaves get gradients.
        assert mse.grad is not None
        assert ce.grad is not None

    def test_zero_ce_weight_drops_ce_from_backward(self):
        mse, ce = _leaf(2.0), _leaf(3.0)
        loss = _assemble_weighted_loss({"MSE": mse, "CE": ce}, {"MSE": 1.0, "CE": 0.0})
        assert loss.item() == pytest.approx(2.0)
        loss.backward()
        assert mse.grad is not None  # kept term trains normally
        # Dropped term: backward never traversed it, so the leaf gets *no* grad
        # (None), not a zero grad. This is the whole point of the feature.
        assert ce.grad is None

    def test_zero_mse_weight_drops_mse_from_backward(self):
        mse, ce = _leaf(2.0), _leaf(3.0)
        loss = _assemble_weighted_loss({"MSE": mse, "CE": ce}, {"MSE": 0.0, "CE": 1.0})
        assert loss.item() == pytest.approx(3.0)
        loss.backward()
        assert ce.grad is not None
        assert mse.grad is None

    def test_all_zero_raises(self):
        mse, ce = _leaf(2.0), _leaf(3.0)
        with pytest.raises(ValueError, match="No active loss term"):
            _assemble_weighted_loss({"MSE": mse, "CE": ce}, {"MSE": 0.0, "CE": 0.0})

    def test_weighting_key_absent_from_losses_is_skipped(self):
        # A weighting key that the policy did not return is ignored rather than
        # raising a KeyError mid-step.
        mse = _leaf(2.0)
        loss = _assemble_weighted_loss({"MSE": mse}, {"MSE": 1.0, "CE": 1.0})
        assert loss.item() == pytest.approx(2.0)


class TestLossWeightingValidation:
    """``TrainPipelineConfig._validate_loss_weighting`` guards startup configs."""

    @staticmethod
    def _validate(weighting: dict):
        cfg = object.__new__(TrainPipelineConfig)
        cfg.loss_weighting = weighting
        TrainPipelineConfig._validate_loss_weighting(cfg)

    def test_zero_weight_is_allowed(self):
        # A single zero weight is the supported drop-a-term case.
        self._validate({"MSE": 1, "CE": 0})
        self._validate({"MSE": 0, "CE": 1})

    def test_all_zero_rejected(self):
        with pytest.raises(ValueError, match="all-zero"):
            self._validate({"MSE": 0, "CE": 0})

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            self._validate({"MSE": 1, "CE": -0.5})

    def test_missing_required_key_rejected(self):
        with pytest.raises(ValueError, match="must define weights"):
            self._validate({"MSE": 1})
