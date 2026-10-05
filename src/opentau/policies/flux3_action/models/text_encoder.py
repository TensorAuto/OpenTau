# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Qwen3-VL text encoder: stacks hidden states from selected layers along the
channel dim to form the DiT text context. Uses Qwen3-VL-4B: `ctx` = selected hidden layers stacked along the channel dim
(width = len(output_layer) * qwen_hidden = 8 * 2560 = 20480). Needs `transformers`.

Prompts are encoded one at a time, right-padded to the next multiple of
``TEXT_PAD_MULTIPLE`` (capped at ``TEXT_PAD_MAX_LENGTH``). Prompt and negative
bucket to independent lengths, so CFG runs as two bs=1 forward passes (see
``flux_action.inference.sampling``).
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

TEXT_PAD_MULTIPLE = 80
TEXT_PAD_MAX_LENGTH = 8192


def padded_text_token_count(
    real_len: int,
    pad_multiple: int,
    pad_max_length: int,
) -> int:
    return min(math.ceil(real_len / pad_multiple) * pad_multiple, pad_max_length)


@dataclass
class Qwen3VLEmbedderParams:
    output_layer: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32])
    torch_dtype: str = "bfloat16"
    stack_channels: bool = True
    use_compile: bool = False
    compile_config: dict | None = field(default_factory=lambda: {"dynamic_compile": True})


class Qwen3VLEmbedder(nn.Module):
    is_mock = False  # marker for tools and tests; the policy calls ``forward_bucketed`` on either encoder

    def __init__(self, model_spec: str, params: Qwen3VLEmbedderParams = Qwen3VLEmbedderParams()):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.params = params
        self.dtype = getattr(torch, params.torch_dtype)
        self.output_layer = list(params.output_layer)
        self.stack_channels = params.stack_channels

        # ``repo_id[:subfolder][@revision]`` also supports shared frozen encoders.
        hub_kwargs = {}
        if not os.path.exists(model_spec):
            from .runtime import parse_weight_spec

            model_spec, subfolder, revision = parse_weight_spec(model_spec)
            hub_kwargs = {"revision": revision, "subfolder": subfolder or ""}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_spec, torch_dtype=self.dtype, **hub_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_spec, **hub_kwargs)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the text encoder requires a right-padding tokenizer")

        if params.use_compile:
            cc = params.compile_config or {}
            for layer in self.model.model.language_model.layers:
                layer.forward = torch.compile(
                    layer.forward,
                    mode=cc.get("compile_mode", "default"),
                    fullgraph=cc.get("fullgraph", False),
                    dynamic=cc.get("dynamic_compile", False),
                    backend=cc.get("backend", "inductor"),
                )

    @torch.no_grad()
    def forward_bucketed(self, text: str, *, fixed_length: int | None = None) -> Tensor:
        """Single-string encode -> ``(1, L, len(output_layer) * qwen_hidden)``,
        right-padded to the bucketed length. The DiT's ``vector`` and
        ``timesteps_ctx`` are zeros, F3 does not condition on a pooled text
        vector."""
        return self.forward_bucketed_batch([text], fixed_length=fixed_length)[0]

    @torch.no_grad()
    def forward_bucketed_batch(self, texts: list[str], *, fixed_length: int | None = None) -> list[Tensor]:
        """Encode many strings: one forward per padded length bucket, ``[(1, L_i, ...)]`` in input order.

        Every string is right-padded to its own bucket (the next multiple of ``TEXT_PAD_MULTIPLE``), as
        in :meth:`forward_bucketed`; strings of one bucket share a forward. The attention mask hides the
        padding, so a string's context does not depend on what it is batched with.
        """
        if fixed_length is not None and not 1 <= fixed_length <= TEXT_PAD_MAX_LENGTH:
            raise ValueError(f"fixed_length must be between 1 and {TEXT_PAD_MAX_LENGTH}")
        tokenizer = self.processor.tokenizer
        formatted = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            for text in texts
        ]
        buckets: dict[int, list[int]] = {}
        for i, prompt in enumerate(formatted):
            if fixed_length is not None:
                buckets.setdefault(fixed_length, []).append(i)
                continue
            real_len = tokenizer(
                prompt, return_tensors="pt", padding=False, truncation=True, max_length=TEXT_PAD_MAX_LENGTH
            )["input_ids"].shape[1]
            target = padded_text_token_count(real_len, TEXT_PAD_MULTIPLE, TEXT_PAD_MAX_LENGTH)
            buckets.setdefault(target, []).append(i)
        results: list[Tensor | None] = [None] * len(texts)
        for target_length, members in sorted(buckets.items()):
            toks = tokenizer(
                [formatted[i] for i in members],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=target_length,
                padding_side="right",
            )
            out = self.model.model(
                input_ids=toks["input_ids"].to(self.model.device, non_blocking=True),
                attention_mask=toks["attention_mask"].to(self.model.device, non_blocking=True),
                output_hidden_states=True,
                use_cache=False,
            )
            if len(self.output_layer) == 1:
                result = out.hidden_states[self.output_layer[0]]
            else:
                stacked = torch.stack([out.hidden_states[k] for k in self.output_layer], dim=1)
                result = rearrange(stacked, "b c l d -> b l (c d)") if self.stack_channels else stacked
            result = result.to(self.dtype)
            for j, i in enumerate(members):
                results[i] = result[j : j + 1]
        return results


# ---------------------------------------------------------------------------------------------
# Policy-side stand-in and loader.
# ---------------------------------------------------------------------------------------------
DEFAULT_TEXT_TOKENS = TEXT_PAD_MULTIPLE
CTX_DIM = 20480  # 8 Qwen3-VL-4B layers x 2560 hidden
VEC_DIM = 768  # pooled vector width; FLUX Action feeds zeros


class MockTextEncoder(nn.Module):
    """Shape-only stand-in: deterministic pseudo-random context per caption, no weights needed."""

    is_mock = True

    def __init__(self, context_in_dim: int = CTX_DIM, device: str | torch.device = "cpu"):
        super().__init__()
        self.context_in_dim = context_in_dim
        del device  # contexts are produced on the CPU; the caller moves them (see ``text_context``)

    @torch.inference_mode()
    def encode(self, batch: int, n_tokens: int = DEFAULT_TEXT_TOKENS, seed: int = 0) -> tuple[Tensor, Tensor]:
        gen = torch.Generator().manual_seed(seed)
        ctx = torch.randn(batch, n_tokens, self.context_in_dim, generator=gen).to(torch.bfloat16)
        vector = torch.zeros(batch, VEC_DIM, dtype=torch.bfloat16)
        return ctx, vector

    @torch.inference_mode()
    def forward_bucketed(self, text: str, *, fixed_length: int | None = None) -> Tensor:
        # different captions -> different (but reproducible) contexts, so CFG has something to guide on
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**31)
        return self.encode(1, n_tokens=fixed_length or DEFAULT_TEXT_TOKENS, seed=seed)[0]

    def forward_bucketed_batch(self, texts: list[str], *, fixed_length: int | None = None) -> list[Tensor]:
        kwargs = {} if fixed_length is None else {"fixed_length": fixed_length}
        return [self.forward_bucketed(text, **kwargs) for text in texts]


def load_text_encoder(
    spec: str | None,
    device: str | torch.device = "cpu",
    *,
    compile_model: bool = False,
    context_in_dim: int = CTX_DIM,
) -> nn.Module:
    """``spec`` = Hub id or local path of Qwen3-VL-4B (needs ``transformers``); ``None`` -> mock."""
    if spec:
        from ..hub import looks_like_local_path

        if looks_like_local_path(str(spec)) and not Path(spec).expanduser().exists():
            raise FileNotFoundError(
                f"no such text encoder directory: {spec}. Paths are relative to the working directory; "
                f"download the shared encoders first (docs/setup.md), or pass a Hub id to download them."
            )
        enc = Qwen3VLEmbedder(spec, Qwen3VLEmbedderParams(use_compile=compile_model))
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc.to(device)
    return MockTextEncoder(context_in_dim, device)


@torch.no_grad()
def text_context(text_encoder: nn.Module, caption: str, device: str | torch.device) -> Tensor:
    """Caption -> ctx ``(1, L, 20480)`` bf16, ``L`` a multiple of 80. The empty caption is the CFG null."""
    return text_contexts(text_encoder, [caption], device)[0]


@torch.no_grad()
def text_contexts(
    text_encoder: nn.Module,
    captions: list[str],
    device: str | torch.device,
    *,
    fixed_length: int | None = None,
) -> list[Tensor]:
    """Captions -> contexts ``(1, L_i, 20480)`` bf16 in order; one encoder call per length bucket when the
    encoder offers ``forward_bucketed_batch``, otherwise one call per caption."""
    encode = getattr(text_encoder, "forward_bucketed_batch", None)
    kwargs = {} if fixed_length is None else {"fixed_length": fixed_length}
    ctxs = (
        encode(captions, **kwargs)
        if encode is not None
        else [text_encoder.forward_bucketed(c, **kwargs) for c in captions]
    )
    # produced under inference_mode: clone so autograd may consume it downstream
    return [ctx.to(device=device, dtype=torch.bfloat16).clone() for ctx in ctxs]
