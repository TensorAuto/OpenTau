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

"""cosmos3: a flow-matching Vision-Language-Action policy.

cosmos3 pairs a **frozen Qwen3-VL-32B backbone** (loaded with NVIDIA
``Cosmos-Reason2-32B`` "reasoner" weights) with a custom, sub-1B **Qwen3-style
flow-matching action expert** in the π0.5 dual-stream style: the backbone
encodes images + the language prompt once (prefix), and the trainable expert
cross-attends to the backbone's per-layer key/value cache to denoise a
continuous action chunk. Continuous actions only (MSE flow matching) -- no FAST
discrete-action tokens and no subtask/response head.
"""
