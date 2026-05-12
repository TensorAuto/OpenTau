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

"""Shared utilities for the ring-attention verification experiments.

Builds a tiny ``Gemma3WithExpertModel`` (no real pretrained weights, no
vision tower) and helper functions to construct synthetic prefix inputs +
the pi07 block-causal attention mask. Everything is fp32-on-CPU by
default and gets moved to the active CUDA device by the caller — keeps
the construction logic identical between sdpa and ring runs.
"""

from __future__ import annotations

import contextlib
import os
import socket

import torch
import torch.distributed as dist


def _free_port() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialise NCCL. Returns (rank, world_size, device).

    Idempotent — safe to call twice within the same process.
    """
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size, torch.device(f"cuda:{rank}")


def build_tiny_gemma3_config(
    num_hidden_layers: int = 2,
    hidden_size: int = 256,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 64,
    intermediate_size: int = 512,
):
    """A Gemma3WithExpertConfig small enough that a 64K-token prefix fits in 24 GB.

    Importing inside the function keeps the file usable as a stand-alone
    helper even before opentau is installed (e.g. early imports during the
    distributed launch).
    """
    from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig

    return Gemma3WithExpertConfig(
        gemma3_config={
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "query_pre_attn_scalar": head_dim,
                "sliding_window": 1024,
                "rope_theta": 1_000_000.0,
                "rope_local_base_freq": 10_000.0,
                "rms_norm_eps": 1e-6,
                "vocab_size": 1024,
                "max_position_embeddings": 131_072,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window_pattern": 6,
                "torch_dtype": "float32",
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "image_size": 28,
                "projection_dim": hidden_size,
                "projector_hidden_act": "gelu_fast",
                "vision_use_head": False,
                "torch_dtype": "float32",
                "layer_norm_eps": 1e-6,
            },
            "image_token_index": 1,
            "mm_tokens_per_image": 4,
            "boi_token_index": 2,
            "eoi_token_index": 3,
            "initializer_range": 0.02,
        },
        gemma_expert_config={
            "model_type": "gemma",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": head_dim,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": 8192,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "torch_dtype": "float32",
            "use_adarms": True,
            "adarms_cond_dim": hidden_size,
            "use_cache": True,
            "vocab_size": 1024,
        },
        freeze_vision_encoder=True,
        train_expert_only=False,
        load_pretrained_gemma3=False,
        discrete_action_vocab_size=256,
        dropout=0.0,
        gradient_checkpointing=False,
        disable_action_expert=True,
        disable_internal_bf16_cast=False,
    )


def build_model(config, device: torch.device):
    """Construct the model on CPU then move to ``device``, leaving params in fp32."""
    from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertModel

    model = Gemma3WithExpertModel(config=config)
    return model.to(device)


def synthesise_prefix_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Produce inputs_embeds / attention_mask / position_ids for a prefix forward.

    The attention mask is the pi07 block-causal mask with two blocks (a
    bidirectional image+language prefix and a bidirectional state postfix);
    that's the same structure ``make_att_2d_masks`` produces for typical
    pi07 training inputs and exercises the non-causal branch ring attention
    needs to support.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    embs = torch.randn(batch_size, seq_len, hidden_size, generator=gen, dtype=dtype).to(device)
    # Block-causal mask: split sequence into 2 bidirectional blocks.
    split = seq_len // 2
    pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    att_masks = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=device)
    att_masks[:, 0] = 1
    att_masks[:, split] = 1
    from opentau.policies.pi07.low_level.modeling_pi07_low_level import make_att_2d_masks

    attn_mask = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    return embs, attn_mask, position_ids


def reset_peak_memory(device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def peak_memory_gb(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**3)


@contextlib.contextmanager
def attention_implementation(model, impl: str):
    """Swap the model's ``attention_implementation`` for the duration of a block."""
    original = model.config.attention_implementation
    model.config.attention_implementation = impl
    try:
        yield
    finally:
        model.config.attention_implementation = original


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
