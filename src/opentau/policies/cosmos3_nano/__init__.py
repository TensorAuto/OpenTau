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

"""cosmos3_nano: the cosmos3 flow-matching VLA policy on the smaller Cosmos3-Nano reasoner.

cosmos3_nano is architecturally identical to ``policies/cosmos3`` — a frozen Qwen3-VL
backbone encodes images + language once (prefix) and a trainable sub-1B Qwen3-style
flow-matching action expert cross-attends to its per-layer key/value cache to denoise a
continuous action chunk — but swaps the 32B backbone (the Cosmos3-Super reasoning tower)
for the **Qwen3-VL-8B reasoning tower of NVIDIA Cosmos3-Nano**, extracted into a
standalone checkpoint by ``opentau.scripts.extract_cosmos3_reasoner``.

The Nano reasoner keeps the exact KV interface the expert consumes (8 key/value heads x
head_dim 128 — identical to the 32B tower), so the only geometry change is the backbone
depth: 36 layers instead of 64. All modeling code is imported from
``policies/cosmos3``; this package only carries the nano defaults.
"""
