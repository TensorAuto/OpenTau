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

"""Gemma 3 backbone with Gemma-v1 action expert, for the PI06 policy.

Mirrors `paligemma_with_expert.py` but:
  * the vision-language backbone is `Gemma3ForConditionalGeneration` (Gemma 3 4B,
    SigLIP-400m/14 + Gemma 3 text, 34 interleaved sliding-window/global layers);
  * the action expert is a Gemma-v1 `GemmaForCausalLM` with the AdaRMS and
    gated-residual patches applied by `opentau.utils.transformers_patch`.

The per-layer attention loop below concatenates backbone and expert queries/
keys/values along the sequence dimension at every layer (the MoE-like pattern
introduced in π0), so the expert can cross-attend to the backbone's activations
at every depth. Gemma 3 specifics (q_norm/k_norm, pre/post feedforward RMSNorms,
per-layer local vs global RoPE, sliding-window attention) are all honored.

`transformers_patch` is imported at module load so the expert path picks up
adaptive RMSNorm and `_gated_residual`. The Gemma 3 backbone remains stock —
its layer-norms return a plain tensor and are used without a `cond=` argument.
"""

import torch
from torch import nn
from transformers import (
    AutoConfig,
    Cache,
    Gemma3ForConditionalGeneration,
    GemmaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

# Ensure the Gemma-v1 AdaRMS / gated-residual patches are live before we
# construct an action expert. Import for side effects only.
from opentau.utils import transformers_patch  # noqa: F401


def _preferred_dtype():
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Applies RoPE to `x` with the given positions and base wavelength.

    Args:
        x: Tensor of shape `(B, L, H, D)`.
        positions: Tensor of shape `(B, L)`.
        max_wavelength: RoPE base frequency. Gemma 3 uses 10_000 for sliding
            (local) layers and 1_000_000 for full (global) layers; Gemma-v1
            expert uses 10_000.

    Returns:
        RoPE-transformed tensor, same shape / dtype as the input.
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


# NOTE: π0.6 deliberately does NOT enforce Gemma 3's sliding-window mask.
# The model card describes "bidirectional attention among ALL of the image
# tokens" and "block-wise causal" prefix attention — wording that's
# incompatible with a 1024-token window once you have 4 cameras × 256 image
# tokens = 1024 image tokens. The local layers' pretrained weights still
# rotate at θ=10_000 (we honour that), but the per-layer attention pattern
# is the global block-causal mask everywhere.


class Gemma3WithExpertConfig(PretrainedConfig):
    """Configuration wrapper bundling a Gemma 3 VLM config and a Gemma-v1 expert config."""

    model_type = "Gemma3WithExpertModel"
    sub_configs = {"gemma3_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        gemma3_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        load_pretrained_gemma3: bool = False,
        discrete_action_vocab_size: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initializes the configuration.

        Args:
            gemma3_config: Optional Gemma 3 config dict. Defaults to the
                `google/gemma-3-4b-pt` topology.
            gemma_expert_config: Optional Gemma-v1 action-expert config dict.
                Defaults to a ~860M-parameter Gemma with AdaRMS enabled.
            freeze_vision_encoder: Freeze the SigLIP tower during training.
            train_expert_only: Only update the expert and its heads.
            attention_implementation: "eager" or "fa2" (fa2 not yet supported).
            load_pretrained_gemma3: Whether to pull pretrained Gemma 3 weights
                from the Hub (only recommended when He-initializing the expert).
            discrete_action_vocab_size: FAST tokenizer vocab size.
            dropout: Dropout probability applied in the per-layer loop.
            **kwargs: Passed to `PretrainedConfig`.
        """
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.load_pretrained_gemma3 = load_pretrained_gemma3
        self.discrete_action_vocab_size = discrete_action_vocab_size
        self.dropout = dropout

        # Gemma 3 backbone defaults (match google/gemma-3-4b-pt).
        if gemma3_config is None:
            self.gemma3_config = CONFIG_MAPPING["gemma3"](
                text_config={
                    "model_type": "gemma3_text",
                    "hidden_size": 2560,
                    "intermediate_size": 10240,
                    "num_hidden_layers": 34,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "query_pre_attn_scalar": 256,
                    "sliding_window": 1024,
                    "rope_theta": 1_000_000.0,
                    "rope_local_base_freq": 10_000.0,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262_208,
                    "max_position_embeddings": 131_072,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "sliding_window_pattern": 6,
                    "torch_dtype": "float32",
                },
                vision_config={
                    "model_type": "siglip_vision_model",
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "patch_size": 14,
                    # π0.6 feeds 448×448 images. `Gemma3MultiModalProjector`
                    # hardcodes `patches_per_image = image_size // patch_size`,
                    # so this MUST match the actual input resolution or the
                    # projector's reshape crashes (see test_pi06.py::
                    # TestGemma3WithExpertConfig::test_vision_image_size_matches_input).
                    "image_size": 448,
                    "projection_dim": 2560,
                    "projector_hidden_act": "gelu_fast",
                    "vision_use_head": False,
                    "torch_dtype": "float32",
                    "layer_norm_eps": 1e-6,
                },
                image_token_index=262144,
                mm_tokens_per_image=256,
                boi_token_index=255999,
                eoi_token_index=256000,
                initializer_range=0.02,
            )
        elif isinstance(gemma3_config, dict):
            if "model_type" not in gemma3_config:
                gemma3_config["model_type"] = "gemma3"
            cfg_cls = CONFIG_MAPPING[gemma3_config["model_type"]]
            self.gemma3_config = cfg_cls(**gemma3_config)
        else:
            self.gemma3_config = gemma3_config

        # Gemma-v1 action-expert defaults (~860M params).
        if gemma_expert_config is None:
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1280,
                initializer_range=0.02,
                intermediate_size=5120,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=34,
                # GQA to match the backbone so per-layer KV concatenation works.
                num_key_value_heads=4,
                pad_token_id=0,
                rms_norm_eps=1e-6,
                rope_theta=10_000.0,
                torch_dtype="float32",
                use_adarms=True,
                adarms_cond_dim=1280,
                use_cache=True,
                vocab_size=262_208,
            )
        elif isinstance(gemma_expert_config, dict):
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"
            cfg_cls = CONFIG_MAPPING[gemma_expert_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)
        else:
            self.gemma_expert_config = gemma_expert_config

        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )
        if self.attention_implementation not in ["eager", "fa2"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). "
                "Expected 'eager' or 'fa2'."
            )

        super().__init__(**kwargs)


class Gemma3WithExpertModel(PreTrainedModel):
    """Gemma 3 VLM interleaved layer-wise with a Gemma-v1 action expert."""

    config_class = Gemma3WithExpertConfig

    def __init__(self, config: Gemma3WithExpertConfig):
        super().__init__(config=config)
        self.config = config

        if config.load_pretrained_gemma3:
            self.gemma3 = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-pt")
        else:
            self.gemma3 = Gemma3ForConditionalGeneration(config=config.gemma3_config)

        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # The expert shares embeddings nowhere — drop the unused token table.
        self.gemma_expert.model.embed_tokens = None

        text_hidden = config.gemma3_config.text_config.hidden_size

        self.discrete_action_embedding = nn.Embedding(
            num_embeddings=config.discrete_action_vocab_size,
            embedding_dim=text_hidden,
            padding_idx=0,
        )
        self.da_head = nn.Linear(
            in_features=text_hidden,
            out_features=config.discrete_action_vocab_size,
        )

        self.dropout = nn.Dropout(config.dropout)

        if not torch.compiler.is_compiling():
            self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

        # Cache commonly accessed config scalars.
        self._text_config = config.gemma3_config.text_config
        self._expert_config = config.gemma_expert_config
        self._num_layers = self._text_config.num_hidden_layers
        self._head_dim = self._text_config.head_dim
        self._rope_global = float(self._text_config.rope_theta)
        self._rope_local = float(getattr(self._text_config, "rope_local_base_freq", 10_000.0))
        self._layer_types: list[str] = list(self._text_config.layer_types)
        # Notes:
        #   * the expert's own `rope_theta` is deliberately ignored at runtime
        #     — the shared attention requires the backbone's per-layer θ for
        #     both streams (see `forward()`).
        #   * `text_config.sliding_window` is also deliberately unused — see
        #     the comment near `apply_rope` for why π0.6 doesn't enforce it.
        self._query_pre_attn_scaling = float(self._text_config.query_pre_attn_scalar) ** -0.5

    # Trainable / dtype plumbing

    def set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder:
            vision_tower = self._vision_tower()
            if vision_tower is not None:
                vision_tower.eval()
                for params in vision_tower.parameters():
                    params.requires_grad = False

        if self.config.train_expert_only:
            self.gemma3.eval()
            for params in self.gemma3.parameters():
                params.requires_grad = False
            for param in self.da_head.parameters():
                param.requires_grad = False
            for param in self.discrete_action_embedding.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            vision_tower = self._vision_tower()
            if vision_tower is not None:
                vision_tower.eval()
        if self.config.train_expert_only:
            self.gemma3.eval()
        return self

    def to_bfloat16_like_physical_intelligence(self) -> None:
        self.gemma3 = self.gemma3.to(dtype=torch.bfloat16)
        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal_projector",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    # Embedding helpers

    def _vision_tower(self):
        # Gemma 3's vision tower lives at `gemma3.model.vision_tower` depending on
        # the transformers version; fall back gracefully.
        for path in ("vision_tower", "model.vision_tower"):
            obj = self.gemma3
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                return obj
        return None

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Runs the SigLIP tower + multimodal projector to obtain image tokens."""
        if hasattr(self.gemma3, "get_image_features"):
            return self.gemma3.get_image_features(image)
        return self.gemma3.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token ids through Gemma 3's shared text embedding table."""
        lm = getattr(self.gemma3, "language_model", None)
        if lm is None:
            lm = self.gemma3.model.language_model
        return lm.embed_tokens(tokens)

    def embed_discrete_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dtype != torch.long:
            actions = actions.long()
        return self.discrete_action_embedding(actions)

    # Attention core

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError(
                "fa2 attention is not supported for pi06 yet because of the interleaved "
                "local/global mask pattern — use 'eager' instead."
            )
        return self.eager_attention_forward

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        scaling: float | None = None,
    ) -> torch.Tensor:
        """Standard eager scaled-dot-product attention. `attention_mask` is a
        boolean 2D mask of shape `(B, Q, K)` (True = attend)."""
        num_att_heads = self._text_config.num_attention_heads
        num_key_value_heads = self._text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

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

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= scaling if scaling is not None else head_dim**-0.5
        big_neg = -2.3819763e38

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return att_output

    # Per-layer interleaved forward

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        n_cross_att_tokens: int | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.FloatTensor | None], list[torch.FloatTensor] | Cache | None]:
        """Interleaved per-layer forward for the Gemma 3 backbone and Gemma-v1 expert.

        The two streams (index 0 = backbone, index 1 = expert) share each layer's
        attention — queries and KVs are concatenated along the sequence axis. When
        one stream's embeddings are None the other runs alone, pulling KVs for the
        missing stream from `past_key_values` when `use_cache=True`.

        Args:
            attention_mask: 2D boolean mask of shape `(B, Q, K_total)`. See
                `opentau.policies.pi05.modeling_pi05.make_att_2d_masks`.
            position_ids: `(B, L_total)` token positions, used for RoPE.
            past_key_values: Per-layer KV cache populated on a previous call.
            inputs_embeds: `[backbone_embeds, expert_embeds]`. Either may be None.
            n_cross_att_tokens: Number of prefix tokens to retain in the cache
                (must be provided when `fill_kv_cache=True`).
            use_cache: Read KVs from `past_key_values` (prefix cross-attention).
            fill_kv_cache: Write this call's KVs into `past_key_values`.
            adarms_cond: Per-stream AdaRMS conditioning tensors `[None, cond]`.

        Returns:
            A pair `(outputs_embeds, past_key_values)` where `outputs_embeds` is a
            two-element list mirroring the `inputs_embeds` layout.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        backbone_layers = self._backbone_layers()
        expert_layers = self.gemma_expert.model.layers
        backbone_norm = self._backbone_final_norm()
        expert_norm = self.gemma_expert.model.norm

        # Infer batch size from whichever stream is present.
        batch_size = None
        for h in inputs_embeds:
            if h is not None:
                batch_size = h.shape[0]
                break
        if batch_size is None:
            raise ValueError("`inputs_embeds` must contain at least one non-None entry.")

        head_dim = self._head_dim

        for layer_idx in range(self._num_layers):
            layer_type = self._layer_types[layer_idx]
            is_sliding = layer_type == "sliding_attention"
            layer_rope_theta = self._rope_local if is_sliding else self._rope_global

            layers_this_step = [backbone_layers[layer_idx], expert_layers[layer_idx]]
            # Both streams MUST use the same RoPE base at this layer. Shared
            # attention concatenates Q/K along the sequence axis; the dot-product
            # invariant `R(q,p)·R(k,q) = q·R(q-p)k` only holds when the same θ
            # produced both rotations. For global Gemma-3 layers (θ=1M) this
            # means the expert also rotates at 1M even though the config carries
            # a single fallback `rope_theta=10k`.
            rope_thetas = [layer_rope_theta, layer_rope_theta]

            query_states = []
            key_states = []
            value_states = []
            gates = []
            # Track the pre-attention residual + post-attn layernorm output for the
            # Gemma-3 backbone side, since it needs a second residual around the MLP
            # using `pre_feedforward_layernorm` / `post_feedforward_layernorm`.
            backbone_preattn_residual = None

            for stream_idx, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    gates.append(None)
                    query_states.append(None)
                    key_states.append(None)
                    value_states.append(None)
                    continue

                layer = layers_this_step[stream_idx]

                if stream_idx == 0:
                    # Gemma 3 backbone.
                    backbone_preattn_residual = hidden_states
                    h = layer.input_layernorm(hidden_states)
                    gate = None
                else:
                    # Gemma-v1 expert (patched to return (tensor, gate)).
                    h, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[stream_idx])

                gates.append(gate)
                bsize, seq_len, _ = h.shape
                h = h.to(dtype=_preferred_dtype())

                q = layer.self_attn.q_proj(h).view(bsize, seq_len, -1, head_dim)
                k = layer.self_attn.k_proj(h).view(bsize, seq_len, -1, head_dim)
                v = layer.self_attn.v_proj(h).view(bsize, seq_len, -1, head_dim)

                if stream_idx == 0:
                    # Gemma-3 applies an extra per-head RMSNorm on Q and K.
                    q_norm = getattr(layer.self_attn, "q_norm", None)
                    k_norm = getattr(layer.self_attn, "k_norm", None)
                    if q_norm is not None:
                        q = q_norm(q)
                    if k_norm is not None:
                        k = k_norm(k)

                q = apply_rope(q, position_ids, max_wavelength=rope_thetas[stream_idx])
                k = apply_rope(k, position_ids, max_wavelength=rope_thetas[stream_idx])

                query_states.append(q)
                key_states.append(k)
                value_states.append(v)

            # Drop Nones before concatenating.
            q_list = [q for q in query_states if q is not None]
            k_list = [k for k in key_states if k is not None]
            v_list = [v for v in value_states if v is not None]

            q_concat = torch.cat(q_list, dim=1)
            k_concat = torch.cat(k_list, dim=1)
            v_concat = torch.cat(v_list, dim=1)

            if use_cache and past_key_values is not None and layer_idx in past_key_values:
                k_concat = torch.cat([past_key_values[layer_idx]["key_states"], k_concat], dim=1)
                v_concat = torch.cat([past_key_values[layer_idx]["value_states"], v_concat], dim=1)

            if fill_kv_cache:
                if past_key_values is None:
                    past_key_values = {}
                if n_cross_att_tokens is None:
                    raise ValueError("n_cross_att_tokens must be provided when fill_kv_cache is True")
                past_key_values[layer_idx] = {
                    "key_states": k_concat[:, :n_cross_att_tokens, :, :],
                    "value_states": v_concat[:, :n_cross_att_tokens, :, :],
                }

            # π0.6 keeps the prefix block-causal mask at every layer — the
            # Gemma 3 sliding-window pattern is deliberately not applied
            # (see the note next to `apply_rope`).
            layer_attention_mask = attention_mask

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                layer_attention_mask,
                batch_size,
                head_dim,
                q_concat,
                k_concat,
                v_concat,
                scaling=self._query_pre_attn_scaling,
            )
            att_output = att_output.to(dtype=_preferred_dtype())

            outputs_embeds: list[torch.Tensor | None] = []
            start = 0
            for stream_idx, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue

                layer = layers_this_step[stream_idx]
                seq_len = hidden_states.shape[1]
                end = start + seq_len
                part = att_output[:, start:end]
                start = end

                if part.dtype != layer.self_attn.o_proj.weight.dtype:
                    part = part.to(layer.self_attn.o_proj.weight.dtype)
                part = layer.self_attn.o_proj(part)
                part = self.dropout(part)

                if stream_idx == 0:
                    # Gemma 3 block: residual + post_attn_norm(attn); then a second
                    # residual with pre_feedforward_layernorm / mlp / post_feedforward_layernorm.
                    post_attn = layer.post_attention_layernorm(part)
                    h = backbone_preattn_residual + post_attn

                    ff_residual = h
                    h = layer.pre_feedforward_layernorm(h)
                    h = layer.mlp(h)
                    h = self.dropout(h)
                    h = layer.post_feedforward_layernorm(h)
                    h = ff_residual + h
                    outputs_embeds.append(h)
                else:
                    # Gemma-v1 expert block with AdaRMS gates.
                    h = modeling_gemma._gated_residual(hidden_states, part, gates[stream_idx])  # noqa: SLF001
                    ff_residual = h.clone()
                    h, gate2 = layer.post_attention_layernorm(h, cond=adarms_cond[stream_idx])
                    h = layer.mlp(h)
                    h = self.dropout(h)
                    h = modeling_gemma._gated_residual(ff_residual, h, gate2)  # noqa: SLF001
                    outputs_embeds.append(h)

            inputs_embeds = outputs_embeds

        # Final norms.
        final_outputs: list[torch.Tensor | None] = []
        for stream_idx, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                final_outputs.append(None)
                continue
            if stream_idx == 0:
                final_outputs.append(backbone_norm(hidden_states))
            else:
                out, _ = expert_norm(hidden_states, cond=adarms_cond[stream_idx])
                final_outputs.append(out)

        return final_outputs, past_key_values

    # Gemma 3 structural accessors

    def _backbone_text_model(self):
        # Different transformers versions expose Gemma 3 under slightly different
        # attribute paths. Resolve once.
        if hasattr(self.gemma3, "language_model"):
            return self.gemma3.language_model
        return self.gemma3.model.language_model

    def _backbone_layers(self):
        text_model = self._backbone_text_model()
        if hasattr(text_model, "layers"):
            return text_model.layers
        return text_model.model.layers

    def _backbone_final_norm(self):
        text_model = self._backbone_text_model()
        if hasattr(text_model, "norm"):
            return text_model.norm
        return text_model.model.norm
