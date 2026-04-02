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

"""V-JEPA2 video encoder with Perceiver token reduction.

Wraps a frozen V-JEPA2 ViT encoder with a learnable Perceiver cross-attention
reducer and a linear projection to the VLM embedding dimension.
"""

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from transformers import VJEPA2Config, VJEPA2Model


class PerceiverReducer(nn.Module):
    """Learned cross-attention bottleneck that compresses V-JEPA2 encoder tokens
    into a fixed number of output tokens."""

    def __init__(self, hidden_size: int, num_queries: int = 256, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, N, D) -> (B, num_queries, D)."""
        b = tokens.shape[0]
        q = self.norm_q(self.queries.expand(b, -1, -1))
        kv = self.norm_kv(tokens)
        out, _ = self.cross_attn(q, kv, kv)
        return out


class VJEPA2VideoEncoder(nn.Module):
    """Frozen V-JEPA2 encoder + learnable Perceiver reducer + projection to VLM dim.

    Takes video tensors (B, T, C, H, W) and produces (B, num_video_tokens, vlm_hidden_size).
    """

    def __init__(
        self,
        vjepa2_model_name: str,
        num_frames: int,
        crop_size: int,
        num_video_tokens: int,
        vlm_hidden_size: int,
        perceiver_heads: int = 8,
        freeze_encoder: bool = True,
        encoder_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if encoder_dtype is None:
            encoder_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        vjepa2_config = VJEPA2Config.from_pretrained(vjepa2_model_name)
        vjepa2_config.crop_size = crop_size
        vjepa2_config.frames_per_clip = num_frames

        self.encoder = VJEPA2Model.from_pretrained(
            vjepa2_model_name,
            config=vjepa2_config,
            torch_dtype=encoder_dtype,
            attn_implementation="sdpa",
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        vjepa2_hidden = vjepa2_config.hidden_size  # 1024 for ViT-L
        self.reducer = PerceiverReducer(
            vjepa2_hidden,
            num_queries=num_video_tokens,
            num_heads=perceiver_heads,  # gitleaks:allow
        )
        self.proj = nn.Linear(vjepa2_hidden, vlm_hidden_size)

    def forward(self, video: Tensor) -> Tensor:
        """Encode a video clip into a fixed-length token sequence.

        Args:
            video: (B, T, C, H, W) pixel values, ImageNet-normalized.

        Returns:
            (B, num_video_tokens, vlm_hidden_size)
        """
        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with ctx:
            encoder_out = self.encoder(video, skip_predictor=True)
        tokens = encoder_out.last_hidden_state  # (B, N_patches, vjepa2_hidden)
        tokens = self.reducer(tokens)  # (B, num_video_tokens, vjepa2_hidden)
        tokens = self.proj(tokens)  # (B, num_video_tokens, vlm_hidden_size)
        return tokens
