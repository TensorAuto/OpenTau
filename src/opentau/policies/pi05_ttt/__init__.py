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
PI05 TTT Policy Module.

This module implements a variant of the π₀.₅ Vision-Language-Action Flow Model
with the Test-Time-Training memory architecture from Jiang, Chebotar, Zheng
et al. "RoboTTT: Context Scaling for Robot Policies". A TTT layer is added
after the attention block of each action-expert layer, so attention keeps
operating within a single timestep while the TTT layer's fast weights — a small
MLP updated by gradient descent at every timestep, in training and at inference
alike — carry the rollout history across timesteps.
"""
