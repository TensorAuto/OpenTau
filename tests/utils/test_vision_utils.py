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

"""CPU unit tests for `opentau.utils.vision_utils`."""

from __future__ import annotations

import pytest
import torch

from opentau.utils.vision_utils import bilinear_resample_pos_embed


class TestBilinearResamplePosEmbed:
    def test_identity_when_same_size(self):
        # No-op (and no copy) — caller relies on this to skip the resample
        # when the published model already matches the policy's patch count.
        pos = torch.randn(1024, 16)
        out = bilinear_resample_pos_embed(pos, target_num_patches=1024)
        assert out is pos

    def test_downsamples_to_target_shape(self):
        pos = torch.randn(4096, 8)  # 64×64
        out = bilinear_resample_pos_embed(pos, target_num_patches=1024)  # 32×32
        assert out.shape == (1024, 8)

    def test_upsamples_to_target_shape(self):
        pos = torch.randn(16, 4)  # 4×4
        out = bilinear_resample_pos_embed(pos, target_num_patches=64)  # 8×8
        assert out.shape == (64, 4)

    def test_preserves_dtype(self):
        # bf16 in, bf16 out — the published Gemma 3 weights are bf16, and the
        # policy expects bf16 in its state_dict. F.interpolate computes in fp32
        # internally and the helper casts back.
        pos = torch.randn(64, 8, dtype=torch.bfloat16)
        out = bilinear_resample_pos_embed(pos, target_num_patches=16)
        assert out.dtype == torch.bfloat16

    def test_deterministic(self):
        # Two calls with the same input must return bit-identical output.
        # Required for byte-equality tests against an independently-resampled
        # reference (see test_pi06_untrained_siglip_matches_gemma3_4b_pt).
        pos = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
        out1 = bilinear_resample_pos_embed(pos, target_num_patches=16)
        out2 = bilinear_resample_pos_embed(pos, target_num_patches=16)
        assert torch.equal(out1, out2)

    def test_rejects_non_square_source(self):
        pos = torch.randn(7, 4)  # 7 is not a square
        with pytest.raises(AssertionError, match="non-square source grid"):
            bilinear_resample_pos_embed(pos, target_num_patches=4)

    def test_rejects_non_square_target(self):
        pos = torch.randn(4, 4)
        with pytest.raises(AssertionError, match="non-square target grid"):
            bilinear_resample_pos_embed(pos, target_num_patches=7)
