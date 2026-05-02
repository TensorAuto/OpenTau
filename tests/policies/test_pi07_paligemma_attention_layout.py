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

"""CPU-only locking tests for the π0.7 PaliGemma attention layout.

These tests pin the post-PR-#235-style fix: the π0.7 paper §VI.B says
*"The following text tokens use causal attention"*, so every text span in
both planners' prefixes must open one causal block per token instead of being
lumped into a bidirectional image / prefix-LM block.

The tests only exercise the shared :func:`make_att_2d_masks` utility, so
they run on CPU without instantiating any model. ``make_att_2d_masks`` is
imported from the high-level planner module to sidestep the unrelated
``VJEPA2VideoEncoder`` pre-existing import bug in the low-level planner
module (tracked separately in #232 / #234); the implementation is byte-
identical in both planners.
"""

import torch

from opentau.policies.pi07_paligemma.high_level_planner.modeling_pi07_high_level import (
    make_att_2d_masks,
)


class TestPI07HighLevelPlannerAttentionLayout:
    """Locks the high-level planner's post-fix attention layout. Mirrors the
    test added in PR #235 for π0.6. The pre-fix code emitted ``[0] * N`` for
    language tokens, lumping them into the bidirectional image block."""

    def test_embed_prefix_layout_has_causal_language_block(self):
        """Image prefix is bidirectional, language tokens are strictly causal,
        language can attend back to the image prefix, images cannot peek at
        later language tokens.
        """
        num_img_embs, num_lang_embs = 4, 5
        att = torch.tensor([[0] * num_img_embs + [1] * num_lang_embs])
        pad = torch.ones_like(att, dtype=torch.bool)
        mask = make_att_2d_masks(pad, att)

        img_slice = slice(0, num_img_embs)
        lang_slice = slice(num_img_embs, num_img_embs + num_lang_embs)

        # Image prefix is fully bidirectional within itself.
        assert torch.all(mask[0, img_slice, img_slice])

        # Language-vs-language is strictly causal.
        lang_block = mask[0, lang_slice, lang_slice]
        expected_causal = torch.tril(torch.ones(num_lang_embs, num_lang_embs, dtype=torch.bool))
        assert torch.equal(lang_block, expected_causal)

        # Language still sees the entire image prefix (prefix-LM cross attention).
        assert torch.all(mask[0, lang_slice, img_slice])
        # Image tokens do NOT see later language tokens.
        assert not torch.any(mask[0, img_slice, lang_slice])


class TestPI07LowLevelPlannerAttentionLayout:
    """Locks the low-level planner's post-fix attention layout. The pre-fix
    code violated paper §VI.B in two distinct ways:
      1. ``[0] * N`` for language tokens (same bug as PR #235).
      2. ``[1] + [0] * (N - 1)`` for the response (Subtask) span: only the
         first token was causal; the remaining N-1 were bidirectional within
         the span, silently leaking future-token information into the
         response loss.
    """

    def test_embed_prefix_layout_has_causal_language_block(self):
        """Video prefix is bidirectional, language tokens are strictly causal,
        language can attend back to the video prefix, video cannot peek at
        later language tokens.
        """
        num_vid_embs, num_lang_embs = 4, 5
        att = torch.tensor([[0] * num_vid_embs + [1] * num_lang_embs])
        pad = torch.ones_like(att, dtype=torch.bool)
        mask = make_att_2d_masks(pad, att)

        vid_slice = slice(0, num_vid_embs)
        lang_slice = slice(num_vid_embs, num_vid_embs + num_lang_embs)

        assert torch.all(mask[0, vid_slice, vid_slice])
        lang_block = mask[0, lang_slice, lang_slice]
        expected_causal = torch.tril(torch.ones(num_lang_embs, num_lang_embs, dtype=torch.bool))
        assert torch.equal(lang_block, expected_causal)
        assert torch.all(mask[0, lang_slice, vid_slice])
        assert not torch.any(mask[0, vid_slice, lang_slice])

    def test_embed_prefix_layout_has_causal_response_block(self):
        """The response (Subtask) span is text: every token must open its
        own causal block (``[1] * N``). This test guards against regression
        to the prefix-LM ``[1] + [0] * (N - 1)`` pattern that allowed bytes
        within the response span to attend to one another bidirectionally.
        """
        num_response_embs = 6
        # Three preceding causal text tokens stand in for the rest of the
        # text prompt; the response span follows immediately.
        prefix_text_n = 3
        att = torch.tensor([[1] * prefix_text_n + [1] * num_response_embs])
        pad = torch.ones_like(att, dtype=torch.bool)
        mask = make_att_2d_masks(pad, att)

        resp_slice = slice(prefix_text_n, prefix_text_n + num_response_embs)
        resp_block = mask[0, resp_slice, resp_slice]
        expected_causal = torch.tril(torch.ones(num_response_embs, num_response_embs, dtype=torch.bool))
        assert torch.equal(resp_block, expected_causal), (
            f"Response sub-block must be strictly causal, got:\n{resp_block.int()}"
        )
