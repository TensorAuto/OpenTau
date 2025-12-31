#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

from opentau.policies.tau0.local_visual_encoder import SmallCNN


@pytest.mark.slow  # ~ 7 sec
def test_smallcnn_output_shape_default():
    model = SmallCNN()  # default output_size=1024
    x = torch.rand(2, 3, 224, 224) * 2 - 1.0  # in [-1, 1]
    y = model(x)
    assert y.shape == (2, 100, 1024)


@pytest.mark.slow  # ~ 7 sec
def test_smallcnn_output_shape_custom_output_size():
    model = SmallCNN(output_size=256)
    x = torch.rand(3, 3, 224, 224) * 2 - 1.0
    y = model(x)
    assert y.shape == (3, 100, 256)


@pytest.mark.slow  # ~ 7 sec
def test_smallcnn_backward_creates_grads():
    model = SmallCNN()
    x = torch.rand(1, 3, 224, 224) * 2 - 1.0
    y = model(x).sum()
    y.backward()
    # Check a couple of parameters have gradients
    assert model.conv1.weight.grad is not None
    assert model.final_linear.weight.grad is not None
