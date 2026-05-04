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

"""Tests for `<loc>` token handling on the PaliGemma tokenizer.

PaliGemma reserves the 1024 loc tokens at IDs 256000..257023 in its base
vocab, but the bare HF tokenizer **does not** register them as added/
special tokens, so a string like ``"<loc0000>"`` BPE-fragments into seven
pieces during ``encode``. Calling ``ensure_loc_tokens`` on the tokenizer
promotes the existing entries to single-token match mode without
reassigning their IDs (no vocab growth, no model-side resize). Every test
below exercises the post-promotion behavior — the public contract that
`PI05Policy.__init__` relies on.

Marked `slow` because the tokenizer is fetched from the HF Hub on first
run. The model itself is NOT downloaded — only `AutoTokenizer` files.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from opentau.datasets.grounding.loc_codec import xyxy_to_loc_tokens
from opentau.datasets.grounding.tokenizer_utils import LOC_TOKENS, ensure_loc_tokens


@pytest.fixture(scope="module")
def paligemma_tokenizer():
    """Module-scoped tokenizer with `<loc>` tokens already promoted."""
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    ensure_loc_tokens(tok)
    return tok


@pytest.mark.slow
def test_ensure_loc_tokens_does_not_grow_vocab() -> None:
    """No new IDs assigned: the strings already live at 256000..257023."""
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    assert ensure_loc_tokens(tok) == 0


@pytest.mark.slow
def test_loc0000_is_single_token_after_promotion(paligemma_tokenizer) -> None:
    ids = paligemma_tokenizer.encode("<loc0000>", add_special_tokens=False)
    assert len(ids) == 1


@pytest.mark.slow
def test_loc_token_ids_match_reserved_block(paligemma_tokenizer) -> None:
    """Promotion preserves the reserved IDs at 256000..257023."""
    assert paligemma_tokenizer.get_vocab()["<loc0000>"] == 256000
    assert paligemma_tokenizer.get_vocab()["<loc1023>"] == 257023


@pytest.mark.slow
def test_loc_token_ids_are_contiguous(paligemma_tokenizer) -> None:
    """`<loc0001>` sits one ID after `<loc0000>` — sanity on the block."""
    id0 = paligemma_tokenizer.encode("<loc0000>", add_special_tokens=False)[0]
    id1 = paligemma_tokenizer.encode("<loc0001>", add_special_tokens=False)[0]
    assert id1 == id0 + 1


@pytest.mark.slow
def test_bbox_postfix_round_trips(paligemma_tokenizer) -> None:
    """A 4-loc bbox postfix encodes + decodes without losing any loc token."""
    img_w, img_h = 1024, 1024
    postfix = xyxy_to_loc_tokens((10.0, 20.0, 30.0, 40.0), img_w, img_h) + " dog"

    ids = paligemma_tokenizer.encode(postfix, add_special_tokens=False)
    decoded = paligemma_tokenizer.decode(ids)

    for tok in LOC_TOKENS:
        if tok in postfix:
            assert tok in decoded


@pytest.mark.slow
def test_ensure_loc_tokens_is_idempotent(paligemma_tokenizer) -> None:
    """Calling promotion a second time on an already-promoted tokenizer
    does not grow the vocab or change behavior."""
    before_size = len(paligemma_tokenizer)
    assert ensure_loc_tokens(paligemma_tokenizer) == 0
    assert len(paligemma_tokenizer) == before_size


@pytest.mark.slow
def test_all_1024_loc_tokens_are_present(paligemma_tokenizer) -> None:
    vocab = paligemma_tokenizer.get_vocab()
    for tok in LOC_TOKENS:
        assert tok in vocab, f"{tok} missing from PaliGemma vocab"
