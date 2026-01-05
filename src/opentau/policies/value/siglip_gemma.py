# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""SigLip + Gemma Model for Value Function Estimation.

This module defines the configuration and model classes for a value function estimator
that combines a SigLIP vision encoder and a Gemma language model.
"""

import torch
from einops import rearrange
from torch import nn
from transformers import (
    AutoConfig,
    Gemma3ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    SiglipVisionModel,
)
from transformers.models.auto import CONFIG_MAPPING


class SiglipGemmaValueConfig(PretrainedConfig):
    """Configuration class for SiglipGemmaValueModel.

    Args:
        siglip_config: Configuration for the SigLIP vision model.
        gemma_config: Configuration for the Gemma language model.
        num_value_bins: Number of bins for value discretization. Defaults to 201.
        **kwargs: Additional keyword arguments passed to `PretrainedConfig`.
    """

    model_type = "SiglipGemmaValueModel"
    sub_configs = {"siglip_config": AutoConfig, "gemma_config": AutoConfig}

    def __init__(
        self,
        siglip_config: dict | None = None,
        gemma_config: dict | None = None,
        num_value_bins: int = 201,
        **kwargs,
    ):
        self.num_value_bins = num_value_bins

        if siglip_config is None:
            # Default SIGLIP config similar to PaliGemma vision config
            self.siglip_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=4304,
                model_type="siglip_vision_model",
                num_attention_heads=16,
                num_hidden_layers=27,
                num_image_tokens=256,
                patch_size=14,
                projector_hidden_act="gelu_fast",
                torch_dtype="float32",
                vision_use_head=False,
            )
        elif isinstance(siglip_config, dict):
            if "model_type" not in siglip_config:
                siglip_config["model_type"] = "siglip_vision_model"

            cfg_cls = CONFIG_MAPPING[siglip_config["model_type"]]
            self.siglip_config = cfg_cls(**siglip_config)
        else:
            self.siglip_config = siglip_config

        if gemma_config is None:
            # Default config for Gemma 3 270M
            # Based on typical scaling: smaller than 1B model
            self.gemma_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=128,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=640,
                initializer_range=0.02,
                intermediate_size=2048,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(gemma_config, dict):
            if "model_type" not in gemma_config:
                gemma_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[gemma_config["model_type"]]
            self.gemma_config = cfg_cls(**gemma_config)
        else:
            self.gemma_config = gemma_config

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()

        if self.attention_implementation not in ["eager", "fa2"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager' or 'fa2'."
            )


class SiglipGemmaValueModel(PreTrainedModel):
    """SigLIP + Gemma Model for Value Function Estimation.

    This model combines a SigLIP vision encoder and a Gemma language model to estimate
    state values. It projects the final hidden state to a set of discretized value bins.

    Args:
        config: Configuration object of type `SiglipGemmaValueConfig`.
    """

    config_class = SiglipGemmaValueConfig

    def __init__(self, config: SiglipGemmaValueConfig):
        """Initializes the SiglipGemmaValueModel.

        Args:
            config: Configuration object of type `SiglipGemmaValueConfig`.
        """
        super().__init__(config=config)

        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch14-224")

        # Initialize language model (Gemma 3 270M)
        self.gemma = Gemma3ForCausalLM.from_pretrained("google/gemma-3-270m")
        self.gemma = self.gemma.model  # we do not want the LM head

        # Value head: projects final hidden state to discretized value bins
        self.value_head = nn.Linear(640, config.num_value_bins)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embeds images using the SIGLIP vision encoder.

        Args:
            image: Tensor containing image pixel values.

        Returns:
            torch.Tensor: The embedded image features.
        """
        # Handle different transformers versions
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(image)
        else:
            outputs = self.vision_encoder(pixel_values=image)
            return outputs.last_hidden_state

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds language tokens using the Gemma embedding layer.

        Args:
            tokens: Tensor containing language token IDs.

        Returns:
            torch.Tensor: The embedded language tokens.
        """
        return self.gemma.embed_tokens(tokens)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass that processes vision and language inputs and outputs a value.

        Args:
            inputs_embeds: Tensor of shape [batch_size, sequence_length, embedding_dim]
                containing the combined embeddings of images and text.
            attention_mask: Attention mask for the sequence.
            position_ids: Position IDs for RoPE.

        Returns:
            torch.Tensor: Logits for discretized values of shape [batch_size, num_value_bins].
        """

        attention_mask = rearrange(attention_mask, "b n1 n2 -> b 1 n1 n2")  # support multihead attention
        # HACK: use full attention for sliding attention as well since our context length is almost the same size as the sliding window
        mask_mapping = {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }
        outputs = self.gemma(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=mask_mapping,
        )
        hidden_states = outputs.last_hidden_state

        # Extract the last token's hidden state for value prediction
        # Use the last token (which should be the last language token)
        final_hidden = hidden_states[:, -1]

        # Project to logits for discretized values
        logits = self.value_head(final_hidden)

        return logits
