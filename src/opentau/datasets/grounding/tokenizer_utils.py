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

"""Tokenizer-side support for PaliGemma `<loc0000>`..`<loc1023>` tokens.

Two cases must be handled, both via the same call:

1. **PaliGemma (`google/paligemma-3b-pt-224`).** The 1024 loc strings are
   already in the SentencePiece vocab at IDs ``256000``..``257023``. They
   are NOT, however, registered as **added tokens**, so the bare HF tokenizer
   BPE-fragments any `<loc0000>`-shaped string into seven pieces
   (``['<', 'loc', '0', '0', '0', '0', '>']``) instead of matching it as one
   unit at ID 256000. Calling ``add_tokens`` with an ``AddedToken`` whose
   string already exists in the vocab is the documented HF mechanism to
   *promote* an existing entry to single-token-match status without
   reassigning its ID. No new vocab slots are created and no embedding
   resize is needed.
2. **Gemma 3 (`google/gemma-3-4b-pt`).** The strings are absent. The same
   ``add_tokens`` call appends 1024 new IDs at the end of the vocab; the
   model's embedding table and tied LM head must be resized to match. The
   new rows are random-init — they learn from the grounding data on first
   use. There is no PaliGemma loc-embedding transfer.

The utility below covers both cases idempotently, so it can be wired into
every policy ``__init__`` defensively.
"""

from __future__ import annotations

import logging

from transformers.tokenization_utils_base import AddedToken

LOC_TOKENS: tuple[str, ...] = tuple(f"<loc{i:04d}>" for i in range(1024))

_logger = logging.getLogger(__name__)


def ensure_loc_tokens(tokenizer, model=None) -> int:
    """Idempotently make `<loc0000>`..`<loc1023>` available as single tokens.

    Always promotes the 1024 loc strings to added/special tokens via
    ``tokenizer.add_tokens``. For PaliGemma this is a no-op vocab-size-wise:
    the strings already live at the reserved IDs ``256000``..``257023``, and
    the call only flips them into single-token match mode. For Gemma 3 the
    1024 strings are appended as new IDs, and the model's embedding table
    and tied LM head are resized via ``model.resize_token_embeddings`` when
    a model handle is supplied.

    Safe to call multiple times — once the strings are registered as added
    tokens, subsequent calls neither grow the vocab nor resize.

    Args:
        tokenizer: A HuggingFace `PreTrainedTokenizer` / `PreTrainedTokenizerFast`.
        model: Optional `PreTrainedModel` whose embeddings should be resized
            when new IDs are assigned. Pass the top-level VLM (e.g. the
            ``Gemma3ForConditionalGeneration`` /
            ``PaliGemmaForConditionalGeneration`` instance) — HF's
            ``resize_token_embeddings`` dispatches through
            ``get_input_embeddings`` / ``set_input_embeddings`` to the
            language model and updates the tied LM head as well.

    Returns:
        The number of NEW IDs appended to the tokenizer vocab. Always 0 for
        PaliGemma; 1024 on the first call against a fresh Gemma 3 tokenizer;
        0 for any subsequent call.
    """
    initial_len = len(tokenizer)
    added_tokens = [AddedToken(t, special=True, normalized=False) for t in LOC_TOKENS]
    tokenizer.add_tokens(added_tokens, special_tokens=True)
    n_new_ids = len(tokenizer) - initial_len

    if n_new_ids > 0 and model is not None:
        model.resize_token_embeddings(len(tokenizer))

    if n_new_ids > 0:
        _logger.info(
            "ensure_loc_tokens: appended %d <locNNNN> token IDs (new vocab size %d); embeddings %sresized.",
            n_new_ids,
            len(tokenizer),
            "" if model is not None else "NOT ",
        )
    return n_new_ids
