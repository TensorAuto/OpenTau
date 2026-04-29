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
π07 Low-Level Planner Module.

This module implements the low-level planner of the π07 hierarchical
architecture. It uses V-JEPA2 as a video encoder, processes temporal state
sequences (one continuous token per timestep), and supports optional subtask
response, subgoal image, and metadata conditioning.  Action generation
combines flow matching (continuous actions via an action expert) with FAST
discrete token prediction (through the VLM backbone with Knowledge
Insulation).
"""
