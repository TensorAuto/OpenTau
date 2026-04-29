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
PI06 Policy Module.

This module implements the π06 (Pi06) Vision-Language-Action model from
Physical Intelligence. Relative to π05, π06 swaps the PaliGemma-3B vision-language
backbone for a Gemma-3 4B multimodal backbone, enlarges the action expert to
~860M parameters (matching the backbone depth), raises the default image
resolution from 224×224 to 448×448, and cuts the flow-matching denoising
schedule from 10 to 5 steps.

References:
    - π0.6 Model Card, Physical Intelligence, November 17, 2025.
    - π*0.6: a VLA That Learns From Experience, arXiv:2511.14759.
"""
