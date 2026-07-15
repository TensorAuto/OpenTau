#!/usr/bin/env python

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

"""cosmos3_nano: the cosmos3 flow-matching VLA policy on a frozen Qwen3-VL-8B reasoner.

Architecturally identical to ``policies/cosmos3/modeling_cosmos3.py`` — the model,
prefix/suffix pipeline, flow-matching loss, and sampler are all inherited unchanged.
The Cosmos3-Nano reasoner keeps the exact KV interface the expert cross-attention
consumes (8 KV heads x head_dim 128), so the only difference is the backbone
checkpoint (``TensorAuto/cosmos3-reason-8b``) and depth (36 layers), both carried by
:class:`Cosmos3NanoConfig`.
"""

from opentau.policies.cosmos3.modeling_cosmos3 import Cosmos3Policy
from opentau.policies.cosmos3_nano.configuration_cosmos3_nano import Cosmos3NanoConfig


class Cosmos3NanoPolicy(Cosmos3Policy):
    """OpenTau wrapper for cosmos3_nano — ``Cosmos3Policy`` with the Nano backbone defaults."""

    config_class = Cosmos3NanoConfig
    name = "cosmos3_nano"
