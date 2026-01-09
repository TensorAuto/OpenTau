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

"""Module for patching transformers

Most patches come from the branch fix/lerobot-openpi
"""

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel

# Monkey patch __init__ of GemmaConfig to fix or modify its behavior as needed.

_original_gemma_config_init = GemmaConfig.__init__


def patched_gemma_config_init(
    self, *args, use_adarms: bool = False, adarms_cond_dim: Optional[int] = None, **kwargs
):
    """Initializes the GemmaConfig with added ADARMS support.

    Args:
        self: The GemmaConfig instance.
        *args: Variable length argument list.
        use_adarms: Whether to use Adaptive RMS normalization.
        adarms_cond_dim: The dimension of the conditioning vector for ADARMS.
        **kwargs: Arbitrary keyword arguments.
    """
    # Call the original init with all other arguments
    _original_gemma_config_init(self, *args, **kwargs)

    # Initialize custom attributes
    self.use_adarms = use_adarms
    self.adarms_cond_dim = adarms_cond_dim

    # Set default for adarms_cond_dim if use_adarms is True
    if self.use_adarms and self.adarms_cond_dim is None:
        # hidden_size is set by _original_gemma_config_init
        self.adarms_cond_dim = self.hidden_size


GemmaConfig.__init__ = patched_gemma_config_init


# --- Modeling Patches ---


def _gated_residual(x, y, gate):
    """
    Applies gated residual connection with optional gate parameter.

    Args:
        x: Input tensor (residual)
        y: Output tensor to be added
        gate: Optional gate tensor to modulate the addition

    Returns:
        x + y if gate is None, otherwise x + y * gate
    """
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


modeling_gemma._gated_residual = _gated_residual


class PatchedGemmaRMSNorm(nn.Module):
    """RMS normalization with optional adaptive support (ADARMS)."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        """Initializes the PatchedGemmaRMSNorm.

        Args:
            dim: The dimension of the input tensor.
            eps: The epsilon value for numerical stability.
            cond_dim: The dimension of the conditioning vector (if using ADARMS).
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the normalization layer.

        Args:
            x: The input tensor.
            cond: The conditioning tensor for adaptive normalization.

        Returns:
            A tuple containing the normalized tensor and the gate tensor (if applicable).
            If cond is None, the gate tensor will be None.

        Raises:
            ValueError: If cond dimension does not match the configured cond_dim.
        """
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        """Returns the extra representation of the module."""
        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str


# Apply patches
modeling_gemma.GemmaRMSNorm = PatchedGemmaRMSNorm


def patched_gemma_decoder_layer_init(self, config: GemmaConfig, layer_idx: int):
    """Initializes a GemmaDecoderLayer with potential ADARMS support.

    Args:
        self: The GemmaDecoderLayer instance.
        config: The configuration object.
        layer_idx: The index of the layer.
    """
    modeling_gemma.GradientCheckpointingLayer.__init__(self)
    self.hidden_size = config.hidden_size

    self.self_attn = modeling_gemma.GemmaAttention(config=config, layer_idx=layer_idx)

    self.mlp = modeling_gemma.GemmaMLP(config)

    cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
    self.input_layernorm = modeling_gemma.GemmaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
    )
    self.post_attention_layernorm = modeling_gemma.GemmaRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
    )


modeling_gemma.GemmaDecoderLayer.__init__ = patched_gemma_decoder_layer_init


def patched_gemma_model_init(self, config: GemmaConfig):
    """Initializes the GemmaModel with potential ADARMS support.

    Args:
        self: The GemmaModel instance.
        config: The configuration object.
    """
    modeling_gemma.GemmaPreTrainedModel.__init__(self, config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [modeling_gemma.GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )

    cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
    self.norm = modeling_gemma.GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
    self.rotary_emb = modeling_gemma.GemmaRotaryEmbedding(config=config)
    self.gradient_checkpointing = False

    # Initialize weights and apply final processing
    self.post_init()


modeling_gemma.GemmaModel.__init__ = patched_gemma_model_init


def patched_gemma_pretrained_model_init_weights(self, module: nn.Module):
    """Initializes the weights of the GemmaPreTrainedModel.

    Args:
        self: The GemmaPreTrainedModel instance.
        module: The module to initialize.
    """
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, modeling_gemma.GemmaRMSNorm):
        if hasattr(module, "weight"):
            module.weight.data.fill_(1.0)


modeling_gemma.GemmaPreTrainedModel._init_weights = patched_gemma_pretrained_model_init_weights


def patched_paligemma_model_get_image_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    """Obtains image last hidden states from the vision tower and apply multimodal projection.

    Args:
        self: The PaliGemmaModel instance.
        pixel_values: The tensors corresponding to the input images.
            Shape: (batch_size, channels, height, width).

    Returns:
        Image feature tensor of shape (num_images, image_length, embed_dim).
    """
    image_outputs = self.vision_tower(pixel_values)
    selected_image_feature = image_outputs.last_hidden_state
    image_features = self.multi_modal_projector(selected_image_feature)
    return image_features


PaliGemmaModel.get_image_features = patched_paligemma_model_get_image_features
