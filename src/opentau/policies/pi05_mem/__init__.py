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
"""
PI05 Mem Policy Module.

This module implements a variant of the π05 (Pi05) Vision-Language-Action Flow
Model with the low-level memory architecture from Torne, Pertsch, Walke et al.
"MEM: Multi-Scale Embodied Memory for Vision Language Action Models". The
SigLIP image encoder is extended with space-time separable attention every
N-th layer so past frames can inform the current-frame tokens without
introducing new learnable parameters, and temporal state sequences are
projected into one continuous token per timestep for the Gemma backbone.
"""
