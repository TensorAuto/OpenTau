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

"""Utilities for encoding spatial outputs as PaliGemma-style location tokens.

PaliGemma reserves 1024 single-token IDs `<loc0000>`..`<loc1023>` that
quantize a coordinate axis into 1024 bins. Bounding boxes and points are
emitted as plain strings (`<locYMIN><locXMIN><locYMAX><locXMAX> label`),
which the standard tokenizer turns into a single integer per `<locNNNN>`.

Two helpers live here:

- ``loc_codec``: pure functions to convert pixel coordinates to/from
  the loc-token string format. No torch dependency.
- ``tokenizer_utils.ensure_loc_tokens``: makes the loc strings available
  on any HuggingFace tokenizer. A no-op for PaliGemma (already shipped).
  For Gemma 3 (and any other base tokenizer) it appends them as special
  tokens and, when given a model handle, resizes the embedding table to
  match.

Concrete grounding datasets (PixMo-points, RefCOCO, …) are NOT yet shipped
under this package — see the follow-up tracking the configurable response
formatter that will make them config-driven rather than one class per source.
"""
