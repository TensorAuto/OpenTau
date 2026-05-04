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

"""Tests for `<loc>` token extension on the Gemma 3 4B tokenizer.

Gemma 3 (used by π0.6) does NOT ship `<loc0000>`..`<loc1023>`. The
`ensure_loc_tokens` utility appends them as 1024 special tokens and,
when given a model handle, resizes the model's embedding/LM head to
match.

Marked `slow` because the tokenizer is fetched from the HF Hub on first
run. The Gemma 3 *model* is NOT loaded — instead a fake mini-model is
used to verify that `resize_token_embeddings` is invoked when tokens are
added.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

from opentau.datasets.grounding.loc_codec import xyxy_to_loc_tokens
from opentau.datasets.grounding.tokenizer_utils import LOC_TOKENS, ensure_loc_tokens


def _fresh_gemma3_tokenizer():
    """Always return a freshly loaded tokenizer so tests don't share state."""
    return AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")


class _FakeResizableModel:
    """Minimal stand-in for an HF model: tracks `resize_token_embeddings` calls.

    The real `Gemma3ForConditionalGeneration` is not loaded in CI because
    its weights are several GB. We only need to confirm that the resize
    call fires with the right vocab size.
    """

    def __init__(self, initial_vocab: int, hidden: int = 8) -> None:
        self.embed = nn.Embedding(initial_vocab, hidden)
        self.resize_calls: list[int] = []

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        self.resize_calls.append(new_size)
        old = self.embed
        self.embed = nn.Embedding(new_size, old.embedding_dim)
        with torch.no_grad():
            n = min(old.num_embeddings, new_size)
            self.embed.weight[:n] = old.weight[:n]
        return self.embed


@pytest.mark.slow
def test_gemma3_lacks_loc_tokens_initially() -> None:
    tok = _fresh_gemma3_tokenizer()
    vocab = tok.get_vocab()
    assert "<loc0000>" not in vocab
    assert "<loc1023>" not in vocab


@pytest.mark.slow
def test_ensure_loc_tokens_adds_1024() -> None:
    tok = _fresh_gemma3_tokenizer()
    n_added = ensure_loc_tokens(tok)
    assert n_added == 1024


@pytest.mark.slow
def test_ensure_loc_tokens_is_idempotent() -> None:
    tok = _fresh_gemma3_tokenizer()
    assert ensure_loc_tokens(tok) == 1024
    # Second call must be a no-op now that the strings are in the vocab.
    assert ensure_loc_tokens(tok) == 0


@pytest.mark.slow
def test_loc_tokens_become_single_token_after_extension() -> None:
    tok = _fresh_gemma3_tokenizer()
    ensure_loc_tokens(tok)
    for sample in ("<loc0000>", "<loc0042>", "<loc1023>"):
        ids = tok.encode(sample, add_special_tokens=False)
        assert len(ids) == 1, f"{sample} did not tokenize to a single id"


@pytest.mark.slow
def test_bbox_postfix_round_trips_after_extension() -> None:
    tok = _fresh_gemma3_tokenizer()
    ensure_loc_tokens(tok)

    img_w, img_h = 1024, 1024
    postfix = xyxy_to_loc_tokens((10.0, 20.0, 30.0, 40.0), img_w, img_h) + " dog"
    decoded = tok.decode(tok.encode(postfix, add_special_tokens=False))

    # Each loc string from the input must survive the round-trip.
    for tok_str in LOC_TOKENS:
        if tok_str in postfix:
            assert tok_str in decoded


@pytest.mark.slow
def test_resize_fires_when_tokens_added() -> None:
    """When tokens are added AND a model is provided, embeddings must resize."""
    tok = _fresh_gemma3_tokenizer()
    initial_vocab = len(tok)
    fake = _FakeResizableModel(initial_vocab=initial_vocab)

    n_added = ensure_loc_tokens(tok, model=fake)

    assert n_added == 1024
    assert fake.resize_calls == [initial_vocab + 1024]
    assert fake.embed.num_embeddings == initial_vocab + 1024


@pytest.mark.slow
def test_resize_does_not_fire_on_idempotent_call() -> None:
    """Second call (tokens already present) must not call resize."""
    tok = _fresh_gemma3_tokenizer()
    ensure_loc_tokens(tok)  # first call adds tokens

    fake = _FakeResizableModel(initial_vocab=len(tok))
    n_added = ensure_loc_tokens(tok, model=fake)

    assert n_added == 0
    assert fake.resize_calls == []


@pytest.mark.slow
def test_ensure_loc_tokens_does_not_perturb_caller_rng() -> None:
    """The resize must not consume entropy from the caller's RNG stream.

    Two `torch.randn` draws bracketing `ensure_loc_tokens(..., model=...)`
    must match what the same outer RNG produces without the helper running
    in between. Otherwise the `model.resize_token_embeddings` random init
    would couple construction order to RNG state and silently violate
    CLAUDE.md hard rule #3 (deterministic seeded reruns).
    """
    tok = _fresh_gemma3_tokenizer()
    fake = _FakeResizableModel(initial_vocab=len(tok))

    torch.manual_seed(123)
    expected = torch.randn(8)

    torch.manual_seed(123)
    ensure_loc_tokens(tok, model=fake)
    actual = torch.randn(8)

    assert torch.equal(expected, actual), (
        "ensure_loc_tokens leaked the resize's RNG draws into the caller's stream"
    )


@pytest.mark.slow
def test_ensure_loc_tokens_resize_is_seed_independent() -> None:
    """The resize seeds deterministically, so two calls under different
    outer RNG states must produce bit-identical new embedding rows.
    """
    initial_vocab = len(_fresh_gemma3_tokenizer())

    tok_a = _fresh_gemma3_tokenizer()
    fake_a = _FakeResizableModel(initial_vocab=initial_vocab, hidden=16)
    torch.manual_seed(7)
    ensure_loc_tokens(tok_a, model=fake_a)

    tok_b = _fresh_gemma3_tokenizer()
    fake_b = _FakeResizableModel(initial_vocab=initial_vocab, hidden=16)
    torch.manual_seed(99)
    ensure_loc_tokens(tok_b, model=fake_b)

    new_rows_a = fake_a.embed.weight[initial_vocab:]
    new_rows_b = fake_b.embed.weight[initial_vocab:]
    assert torch.equal(new_rows_a, new_rows_b), (
        "new <locNNNN> embedding rows differ between runs — RNG snapshot/restore failed"
    )
