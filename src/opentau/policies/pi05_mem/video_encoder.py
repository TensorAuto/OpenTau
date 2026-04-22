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

"""SigLIP video encoder with space-time separable attention (MEM paper).

Implements the low-level memory video encoder from Torne, Pertsch, Walke et al.
"MEM: Multi-Scale Embodied Memory for Vision Language Action Models"
(Section III-C + Appendix C): a standard SigLIP ViT extended with
space-time separable attention at every ``spacetime_layer_stride``-th layer
and a fixed sinusoidal temporal position encoding whose current-frame row is
zero. Past-timestep tokens are dropped after the encoder so the output shape
matches a single-image VLA.

Key properties:
  - Introduces no new learnable parameters on top of the pretrained SigLIP
    weights (temporal attention re-uses each layer's own Q/K/V/O projections).
  - Single-frame invariance: with ``T=1`` the output is byte-identical to
    ``PaliGemmaModel.get_image_features`` (see the single-frame invariance
    test in ``tests/policies/test_pi05_mem_gpu.py``).
  - Convention: the current frame lives at the **last** time index
    (``t = T-1``). This matches
    ``src/opentau/datasets/factory.py:136`` (delta_timestamps) and
    ``PI05MemPolicy._build_history_batch``. ``obs_history_is_pad[:, -1]`` is
    always ``False`` by construction.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaMultiModalProjector,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
)

# Loaded for its side-effect of applying ``transformers_patch`` and re-exports
# ``PaliGemmaForConditionalGeneration`` with the patched ``get_image_features``.
from opentau.utils.transformers_patch import PaliGemmaForConditionalGeneration

# PaliGemma's SigLIP vision config (google/paligemma-3b-pt-224). Hardcoded here
# so an empty encoder can be built without downloading pretrained weights.
_PALIGEMMA_VISION_CONFIG_DICT: dict = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_image_tokens": 256,
    "patch_size": 14,
    "image_size": 224,
    "projection_dim": 2048,
    "projector_hidden_act": "gelu_fast",
    "vision_use_head": False,
}
_PALIGEMMA_TEXT_HIDDEN_SIZE = 2048


def _build_temporal_sinusoidal_pe(
    num_frames: int,
    embed_dim: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Fixed sinusoidal temporal positional embedding, ``(T, embed_dim)``.

    Row ``T-1`` (the current frame) is all zeros; earlier rows encode the
    temporal offset into the past via sin/cos on a geometric period schedule
    (matching ``create_sinusoidal_pos_embedding`` in ``modeling_pi05.py``).

    The zero-current-row condition lets a ``T=1`` forward pass match an
    un-modified SigLIP ViT exactly, which is required for single-frame
    invariance against ``PaliGemmaModel.get_image_features``.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2")
    if num_frames < 1:
        raise ValueError(f"num_frames ({num_frames}) must be >= 1")

    # time[i] = i - (T-1) in {-(T-1), ..., -1, 0}; row T-1 has time = 0.
    time = torch.arange(num_frames, dtype=torch.float64, device=device) - (num_frames - 1)
    fraction = torch.linspace(0.0, 1.0, embed_dim // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi  # (embed_dim/2,)
    phase = time.unsqueeze(-1) * scaling.unsqueeze(0)  # (T, embed_dim/2)
    pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (T, embed_dim)
    # Shift so row T-1 is exactly zero (preserves relative sinusoidal structure,
    # enforces boundary condition e(current) = 0 from MEM Appendix C).
    pe = pe - pe[-1:]
    return pe.to(dtype=dtype)


class _TemporalSelfAttention(nn.Module):
    """Parameter-free causal temporal self-attention.

    Re-uses an existing ``SiglipAttention`` instance's
    ``q_proj``/``k_proj``/``v_proj``/``out_proj`` linear layers, but applies
    them over the ``T`` axis (for each fixed patch position) with a
    standard lower-triangular causal mask (position ``i`` attends to
    positions ``j <= i``; since ``t = T-1`` is the current frame, the current
    frame attends to all past frames).
    """

    def __init__(self, attn: SiglipAttention):
        super().__init__()
        self.attn = attn  # reference only; parameters remain owned by attn

    def forward(self, hidden_states: Tensor) -> Tensor:
        """hidden_states: (B*N, T, D) -> (B*N, T, D)."""
        attn = self.attn
        bn, t, d = hidden_states.shape
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        q = attn.q_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bn, t, num_heads, head_dim).transpose(1, 2)

        # is_causal=True -> lower-triangular mask, each position attends to
        # itself and earlier positions (our convention: t=T-1 is current).
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0, scale=attn.scale
        )
        out = out.transpose(1, 2).reshape(bn, t, d)
        return attn.out_proj(out)


class SpaceTimeEncoderLayerWrapper(nn.Module):
    """Wraps a ``SiglipEncoderLayer`` to add a pre-attention temporal sublayer.

    The wrapped block's forward computes:

        h_pe  = h + e(t)                                   # broadcast over (B, N)
        h     = h + temporal_attn( LN1(h_pe) )             # new; causal over T
        return base_layer(h)                               # verbatim SigLIP:
                                                           # h + spatial_attn(LN1(h))
                                                           # h + MLP(LN2(h))

    The residual for the temporal sublayer adds to the pre-PE hidden state so
    that ``T=1`` (where ``e(T-1) = 0``) produces a true no-op (temporal attn
    output at ``T=1`` is a linear function of the value tokens, which still
    matters; that's why we keep the residual on ``h`` rather than ``h_pe``).

    Reusing ``layer_norm1`` for both the temporal and spatial sublayers keeps
    the paper's "no new learnable parameters" guarantee. It is an intentional
    design choice: the two attentions operate on different axes and the
    LayerNorm is applied to different input tensors each time.
    """

    def __init__(
        self,
        base_layer: SiglipEncoderLayer,
        num_frames: int,
        num_tokens_per_frame: int,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_frames = num_frames
        self.num_tokens_per_frame = num_tokens_per_frame
        self._temporal_attn = _TemporalSelfAttention(base_layer.self_attn)

        embed_dim = base_layer.embed_dim
        pe = _build_temporal_sinusoidal_pe(num_frames, embed_dim)
        # Non-persistent: not saved in state_dict but moves with .to(device).
        self.register_buffer("_temporal_pe", pe, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        """hidden_states: (B*T, N, D) -> tuple starting with (B*T, N, D).

        Signature matches ``SiglipEncoderLayer.forward`` so ``SiglipEncoder``
        can dispatch unchanged.
        """
        t = self.num_frames
        bt, n, d = hidden_states.shape
        if bt % t != 0:
            raise ValueError(
                f"hidden_states.shape[0] ({bt}) must be divisible by num_frames ({t}); "
                "video encoder expects inputs flattened as (B*T, N, D)."
            )
        b = bt // t
        if n != self.num_tokens_per_frame:
            raise ValueError(
                f"hidden_states.shape[1] ({n}) != num_tokens_per_frame ({self.num_tokens_per_frame})."
            )

        # Temporal sublayer.
        x = rearrange(hidden_states, "(b t) n d -> b t n d", b=b, t=t)
        # Cast PE to match tensor dtype each call (supports mixed-precision).
        pe = self._temporal_pe.to(dtype=x.dtype).view(1, t, 1, d)
        x_pe = x + pe

        t_in = rearrange(x_pe, "b t n d -> (b n) t d")
        t_norm = self.base_layer.layer_norm1(t_in)
        t_out = self._temporal_attn(t_norm)
        # Residual on the pre-PE hidden (not on x_pe): PE is a transient
        # positional signal, not a feature perturbation to carry forward.
        t_res = rearrange(x, "b t n d -> (b n) t d") + t_out
        h_after_t = rearrange(t_res, "(b n) t d -> (b t) n d", n=n)

        # Spatial + MLP sublayers: delegate to the unmodified SiglipEncoderLayer.
        return self.base_layer(
            h_after_t,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class SpaceTimeSiglipVideoEncoder(nn.Module):
    """SigLIP-based video encoder with space-time separable attention.

    Takes video tensors of shape ``(B, T, 3, H, W)`` in the ``[0, 1]`` range
    and produces ``(B, num_video_tokens, vlm_hidden_size)``. Rescales pixels
    to ``[-1, 1]`` internally (SigLIP's expected range).

    Past-timestep tokens are dropped after the encoder; only the current
    frame's ``num_video_tokens = 256`` tokens are returned, so the output
    shape is identical to a single-frame VLA's vision-token prefix.

    The ``multi_modal_projector`` (``nn.Linear(1152, 2048)``) is applied to
    match ``PaliGemmaModel.get_image_features`` output space. We intentionally
    **do not** apply the ``/ sqrt(text_hidden_size)`` scaling, matching
    ``opentau.utils.transformers_patch.patched_paligemma_model_get_image_features``
    which removes it from stock HuggingFace.
    """

    def __init__(
        self,
        paligemma_model_name: str = "google/paligemma-3b-pt-224",
        num_frames: int = 8,
        num_video_tokens: int = 256,
        vlm_hidden_size: int = 2048,
        spacetime_layer_stride: int = 4,
        freeze_encoder: bool = True,
        encoder_dtype: Optional[torch.dtype] = None,
        load_pretrained: bool = True,
    ):
        super().__init__()
        if num_video_tokens != 256:
            raise ValueError(
                f"num_video_tokens must equal 256 (SigLIP at patch_size=14, "
                f"image_size=224 produces exactly 256 patches), got {num_video_tokens}."
            )
        if vlm_hidden_size != _PALIGEMMA_TEXT_HIDDEN_SIZE:
            raise ValueError(
                f"vlm_hidden_size must equal {_PALIGEMMA_TEXT_HIDDEN_SIZE} to match "
                f"PaliGemma's multi_modal_projector output, got {vlm_hidden_size}."
            )
        if spacetime_layer_stride < 1:
            raise ValueError(f"spacetime_layer_stride ({spacetime_layer_stride}) must be >= 1.")

        self.num_frames = num_frames
        self.num_video_tokens = num_video_tokens
        self.vlm_hidden_size = vlm_hidden_size
        self.spacetime_layer_stride = spacetime_layer_stride

        if encoder_dtype is None:
            encoder_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if load_pretrained:
            # Load the full PaliGemma so the SigLIP + projector weights come
            # from the same checkpoint used by pi05_continuous_state. Keep
            # only the pieces we need; release the rest.
            full = PaliGemmaForConditionalGeneration.from_pretrained(
                paligemma_model_name, torch_dtype=encoder_dtype
            )
            self.vision_tower: SiglipVisionModel = full.vision_tower
            self.multi_modal_projector: PaliGemmaMultiModalProjector = full.multi_modal_projector
            del full
        else:
            # Empty initialization; a checkpoint loader is expected to
            # overwrite these weights.
            paligemma_cfg = PaliGemmaConfig(
                vision_config=_PALIGEMMA_VISION_CONFIG_DICT,
                text_config={
                    "hidden_size": _PALIGEMMA_TEXT_HIDDEN_SIZE,
                    "model_type": "gemma",
                    "vocab_size": 257152,
                },
            )
            vision_cfg = SiglipVisionConfig(**_PALIGEMMA_VISION_CONFIG_DICT)
            self.vision_tower = SiglipVisionModel(vision_cfg).to(dtype=encoder_dtype)
            self.multi_modal_projector = PaliGemmaMultiModalProjector(paligemma_cfg).to(dtype=encoder_dtype)

        # Wrap every stride-th layer with space-time attention. We wrap AFTER
        # loading pretrained weights so the wrapped layers retain their
        # original parameters (the wrapper holds a reference, not a copy).
        # State-dict keys for wrapped layers will carry a ``.base_layer.``
        # prefix; reload within this project is symmetric and needs no
        # remapping.
        layers = self.vision_tower.vision_model.encoder.layers
        n_layers = len(layers)
        for i in range(spacetime_layer_stride - 1, n_layers, spacetime_layer_stride):
            layers[i] = SpaceTimeEncoderLayerWrapper(
                base_layer=layers[i],
                num_frames=num_frames,
                num_tokens_per_frame=num_video_tokens,
            )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.vision_tower.eval()
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            # NOTE: multi_modal_projector is intentionally left trainable,
            # matching the semantics of ``freeze_vision_encoder`` in
            # ``pi05_continuous_state`` (only the vision tower is frozen).

    def forward(self, video: Tensor) -> Tensor:
        """Encode a video clip and return the current-frame tokens.

        Args:
            video: ``(B, T, C, H, W)`` pixel values in ``[0, 1]``, with
                ``T == num_frames``, ``C == 3``, and spatial size matching
                the SigLIP config (224x224 by default).

        Returns:
            ``(B, num_video_tokens, vlm_hidden_size)`` current-frame tokens,
            ready to concatenate into the VLA prefix.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W); got {tuple(video.shape)}.")
        b, t, c, h, w = video.shape
        if t != self.num_frames:
            raise ValueError(
                f"Expected T={self.num_frames} frames; got {t}. "
                "Reinstantiate the encoder with a matching num_frames."
            )

        # SigLIP expects pixel values in [-1, 1]. The dataset loader yields
        # [0, 1]; rescale here (keeps prepare_videos producer-agnostic).
        video = video * 2.0 - 1.0

        # Flatten time into batch for the SigLIP pipeline.
        flat = rearrange(video, "b t c h w -> (b t) c h w")

        # Patch embedding + learned spatial position embedding.
        hidden = self.vision_tower.vision_model.embeddings(flat)

        # Encoder stack: standard spatial layers + wrapped every-Nth layer
        # with temporal attention. SpaceTimeEncoderLayerWrapper matches the
        # SiglipEncoderLayer signature, so SiglipEncoder.forward is unchanged.
        hidden = self.vision_tower.vision_model.encoder(inputs_embeds=hidden).last_hidden_state

        hidden = self.vision_tower.vision_model.post_layernorm(hidden)

        # Drop past-timestep tokens: keep only the current frame (t = T-1).
        # This matches the MEM paper's "we only pass the representation
        # computed for the current timestep onwards" and makes the encoder
        # a drop-in replacement for a single-frame vision tower.
        hidden = rearrange(hidden, "(b t) n d -> b t n d", b=b, t=t)
        current = hidden[:, -1]

        # multi_modal_projector: SigLIP hidden (1152) -> VLA hidden (2048).
        # We deliberately omit the `/ sqrt(hidden_size)` division to match
        # the patched ``PaliGemmaModel.get_image_features`` (see
        # ``opentau.utils.transformers_patch``).
        return self.multi_modal_projector(current)
