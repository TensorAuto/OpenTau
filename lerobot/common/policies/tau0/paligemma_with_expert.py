# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.common.policies.tau0.local_visual_encoder import SmallCNN


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        load_pretrained_paligemma: bool = False,
        use_cache_layer: list[int] = [17]
        * 18,  # which kv cache to use for each layer of the action expert (defaults to using the last layer of the VLM for all action expert layers)
        dropout: float = 0.1,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.load_pretrained_paligemma = load_pretrained_paligemma
        self.use_cache_layer = use_cache_layer
        self.dropout = dropout

        if paligemma_config is None:
            # Default config from tau0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override tau0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from tau0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
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
        elif isinstance(self.gemma_expert_config, dict):
            # Override tau0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager' or 'fa2'."
            )

        if len(self.use_cache_layer) != self.gemma_expert_config.num_hidden_layers:
            raise ValueError(
                f"use_cache_layer must be a list of length {self.gemma_expert_config.num_hidden_layers}. Got {len(self.use_cache_layer)}."
            )
        for layer_idx in self.use_cache_layer:
            if layer_idx < 0 or layer_idx >= self.paligemma_config.text_config.num_hidden_layers:
                raise ValueError(
                    f"use_cache_layer must be between 0 and {self.paligemma_config.text_config.num_hidden_layers - 1}. Got {layer_idx}."
                )

        super().__init__(**kwargs)


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig, execution_target: Optional[str] = None):
        super().__init__(config=config)
        self.config = config
        self.execution_target = execution_target

        # Initialize PaliGemma and Gemma Expert models based on the execution target
        # If execution_target is None, both models are initialized for unified training
        if execution_target is None or execution_target == "cloud":
            if config.load_pretrained_paligemma:
                self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
                    "google/paligemma-3b-pt-224"
                )
            else:
                self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        if execution_target is None or execution_target == "robot":
            self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
            # Remove unused embed_tokens
            self.gemma_expert.model.embed_tokens = None
            self.onboard_vision_encoder = SmallCNN(output_size=config.gemma_expert_config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.set_requires_grad()

    def set_requires_grad(self):
        if self.execution_target != "robot":
            if self.config.freeze_vision_encoder:
                self.paligemma.model.vision_tower.eval()
                for params in self.paligemma.vision_tower.parameters():
                    params.requires_grad = False

            if self.config.train_expert_only:
                self.paligemma.eval()
                for params in self.paligemma.parameters():
                    params.requires_grad = False

    def set_execution_target(self, execution_target: str):
        """Used for setting the execution target of pretrained models"""
        if execution_target not in [None, "robot", "cloud"]:
            raise KeyError(f"{execution_target} must be one of the following: {[None, 'robot', 'cloud']}")
        self.execution_target = execution_target
        if execution_target == "robot":
            del self.paligemma
        if execution_target == "cloud":
            del self.gemma_expert
            del self.onboard_vision_encoder

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder and self.execution_target != "robot":
            self.paligemma.model.vision_tower.eval()

        if self.config.train_expert_only and self.execution_target != "robot":
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.paligemma, "get_image_features"):
            return self.paligemma.get_image_features(image)
        else:
            return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward_through_model(
        self,
        model: PreTrainedModel,
        num_layers: int,
        head_dim: int,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embed: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        n_cross_att_tokens: Optional[int] = None,
        cached_layer: Optional[int] = None,
    ):
        """
        Forward through a Gemma model (either PaliGemma or Gemma Expert) with the given inputs.
        returns the outputs of the model and the past key values if `use_cache` is True.

        Args:
            model (PreTrainedModel): The model to forward through.
            num_layers (int): The number of layers in the model.
            head_dim (int): The dimension of the attention heads.
            batch_size (int): The batch size of the inputs.
            attention_mask (Optional[torch.Tensor], optional): The attention mask to use. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): The position ids to use. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional): The past key values to use.
            inputs_embed (List[torch.FloatTensor], optional): The inputs to embed. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use the cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional): Whether to fill the key value cache. Defaults to None.
            cached_layer (Optional[int], optional): The layer to use for caching. If None, all layers are cached. Defaults to None.
        """
        for layer_idx in range(num_layers):
            hidden_states = inputs_embed
            query_states = []
            key_states = []
            value_states = []

            layer = model.layers[layer_idx]
            # normalizer = torch.tensor(model.config.hidden_size**0.5, dtype=hidden_states.dtype)
            # hidden_states = hidden_states * normalizer
            hidden_states, _ = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    if layer_idx in self.config.use_cache_layer:
                        past_key_values[layer_idx] = {
                            # save the first n_cross_att_tokens for action expert cross attention
                            "key_states": key_states[:, :n_cross_att_tokens, :, :],
                            "value_states": value_states[:, :n_cross_att_tokens, :, :],
                        }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    cached_layer = self.config.use_cache_layer[layer_idx]
                    key_states = torch.cat([key_states, past_key_values[cached_layer]["key_states"]], dim=1)
                    value_states = torch.cat(
                        [value_states, past_key_values[cached_layer]["value_states"]], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            out_emb = self.dropout(out_emb)

            # first residual
            out_emb += inputs_embed
            after_first_residual = out_emb.clone()

            out_emb, _ = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)

            out_emb = self.dropout(out_emb)

            # second residual
            out_emb += after_first_residual

            inputs_embed = out_emb

        # final norm
        out_emb, _ = model.norm(inputs_embed)

        return out_emb, past_key_values

    def forward(
        self,
        inputs_embeds: List[torch.FloatTensor],
        n_cross_att_tokens: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        vlm_attention_mask: Optional[torch.Tensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        action_expert_attention_mask: Optional[torch.Tensor] = None,
        action_expert_position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass through the PaliGemma with Gemma Expert model.

        Args:
            inputs_embeds (List[torch.FloatTensor], optional): List of input embeddings for PaliGemma and Gemma Expert.
                The first element is for PaliGemma and the second for Gemma Expert. If the list element is None, then
                the respective model is not run.
            n_cross_att_tokens (int): Number of cross attention tokens to pass from PaliGemma to Gemma Expert.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional): Past key values used for running
                Gemma Expert only. If None, then the past key values are not used.
            vlm_attention_mask (Optional[torch.Tensor], optional): Attention mask for PaliGemma.
            vlm_position_ids (Optional[torch.LongTensor], optional): Position IDs for PaliGemma.
            action_expert_attention_mask (Optional[torch.Tensor], optional): Attention mask for Gemma Expert.
            action_expert_position_ids (Optional[torch.LongTensor], optional): Position IDs for Gemma Expert.
        """

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # initialize output embeds to None
        paligemma_output_embeds = None
        gemma_output_embeds = None

        # run paligemma first
        if inputs_embeds[0] is not None:
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            head_dim = self.paligemma.config.text_config.head_dim
            paligemma_output_embeds, past_key_values = self.forward_through_model(
                model=self.paligemma.model.language_model,
                num_layers=num_layers,
                head_dim=head_dim,
                batch_size=batch_size,
                attention_mask=vlm_attention_mask,
                position_ids=vlm_position_ids,
                past_key_values=None,
                inputs_embed=inputs_embeds[0],
                use_cache=True,
                fill_kv_cache=True,
                n_cross_att_tokens=n_cross_att_tokens,
                cached_layer=self.config.use_cache_layer,
            )

        # run gemma expert with cross attention on paligemma outputs
        if inputs_embeds[1] is not None:
            num_layers = self.gemma_expert.config.num_hidden_layers
            head_dim = self.gemma_expert.config.head_dim
            gemma_output_embeds, past_key_values = self.forward_through_model(
                model=self.gemma_expert.model,
                num_layers=num_layers,
                head_dim=head_dim,
                batch_size=batch_size,
                attention_mask=action_expert_attention_mask,
                position_ids=action_expert_position_ids,
                past_key_values=past_key_values,
                inputs_embed=inputs_embeds[1],
                use_cache=True,
                fill_kv_cache=False,
                cached_layer=self.config.use_cache_layer,
            )

        # concatenate paligemma and gemma expert outputs
        outputs_embeds = [paligemma_output_embeds, gemma_output_embeds]

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
