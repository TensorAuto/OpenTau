#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import logging
import math
from collections import deque
from itertools import chain
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoTokenizer

from opentau.constants import OBS_ROBOT
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.tau0.configuration_tau0 import TAU0Config
from opentau.policies.tau0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.utils import log_model_loading_keys
from opentau.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks, n_cross_att_tokens=None, cross_att_pad_masks=None):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
        n_cross_att_tokens: int, optional Add attention mask for cross-attention tokens if
        `n_cross_att_tokens` is provided.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    # If `n_cross_att_tokens` is provided, we add a mask for cross-attention tokens at the end of the sequence.
    if n_cross_att_tokens is not None:
        assert cross_att_pad_masks is not None, (
            "cross_att_pad_masks must be provided if n_cross_att_tokens is provided"
        )
        assert cross_att_pad_masks.shape == (att_masks.size(0), n_cross_att_tokens), (
            "cross_att_pad_masks must have shape (batch_size, n_cross_att_tokens)"
        )

        cross_att_mask = torch.full(
            (att_masks.size(0), att_masks.size(1), n_cross_att_tokens),
            True,
            dtype=torch.bool,
            device=att_masks.device,
        )

        # Apply padding masks: pad_masks for rows, cross_att_pad_masks for columns
        cross_att_mask = cross_att_mask & pad_masks[:, :, None] & cross_att_pad_masks[:, None, :]

        att_2d_masks = torch.cat((att_2d_masks, cross_att_mask), dim=2)

    return att_2d_masks


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


class TAU0Policy(PreTrainedPolicy):
    """Wrapper class around TAU0FlowMatching model to train and run inference within LeRobot."""

    config_class = TAU0Config
    name = "tau0"

    def __init__(
        self,
        config: TAU0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        execution_target: Optional[
            str
        ] = None,  # None for unified training, "robot" for robot action decoder inference, "cloud" for VLM on cloud inference
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        if execution_target is None or execution_target == "cloud":
            self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.model = TAU0FlowMatching(config, execution_target=execution_target)
        self.execution_target = execution_target

        # VLM token cache: tuple containing (past_key_values, prefix_offsets, num_cross_att_tokens)
        self.vlm_token_cache = None

        self.reset()

    def set_execution_target(self, execution_target: str):
        """Used for setting the execution target of pretrained models"""
        if execution_target not in [None, "robot", "cloud"]:
            raise KeyError(f"{execution_target} must be one of the following: None, robot, cloud")
        self.execution_target = execution_target
        if execution_target == "robot":
            del self.language_tokenizer
        self.model.set_execution_target(execution_target)

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque()

    @classmethod
    def _transform_state_dict_keys(cls, state_dict: dict) -> dict:
        """
        Transform state dict keys to match expected model structure.
        Transformations:
        - model.paligemma_with_expert.paligemma.language_model.lm_head ->
          model.paligemma_with_expert.paligemma.lm_head
        - model.paligemma_with_expert.paligemma.language_model.model ->
          model.paligemma_with_expert.paligemma.model.language_model
        - model.paligemma_with_expert.paligemma.vision_tower ->
          model.paligemma_with_expert.paligemma.model.vision_tower
        - model.paligemma_with_expert.paligemma.multi_modal_projector ->
          model.paligemma_with_expert.paligemma.model.multi_modal_projector
        Also handles tied weights between lm_head.weight and
        embed_tokens.weight.
        """
        import re

        transformed_dict = {}

        transformations = [
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.lm_head"),
                ".paligemma_with_expert.paligemma.lm_head",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.model"),
                ".paligemma_with_expert.paligemma.model.language_model",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.vision_tower"),
                ".paligemma_with_expert.paligemma.model.vision_tower",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.multi_modal_projector"),
                ".paligemma_with_expert.paligemma.model.multi_modal_projector",
            ),
        ]

        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in transformations:
                new_key = pattern.sub(replacement, new_key)
            transformed_dict[new_key] = value

        # Handle tied weights: lm_head.weight and embed_tokens.weight share memory
        lm_head_key = None
        embed_tokens_key = None

        for key in transformed_dict:
            if key.endswith(".paligemma_with_expert.paligemma.lm_head.weight"):
                lm_head_key = key
            elif key.endswith(".paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"):
                embed_tokens_key = key
            if lm_head_key and embed_tokens_key:
                break

        if lm_head_key and not embed_tokens_key:
            embed_tokens_key = lm_head_key.replace(
                ".lm_head.weight", ".model.language_model.embed_tokens.weight"
            )
            transformed_dict[embed_tokens_key] = transformed_dict[lm_head_key]
        elif embed_tokens_key and not lm_head_key:
            lm_head_key = embed_tokens_key.replace(
                ".model.language_model.embed_tokens.weight", ".lm_head.weight"
            )
            transformed_dict[lm_head_key] = transformed_dict[embed_tokens_key]

        return transformed_dict

    @classmethod
    def _load_as_safetensor(
        cls, model: "TAU0Policy", model_file: str, map_location: str, strict: bool
    ) -> "TAU0Policy":
        """Override to apply key transformations before loading."""
        from safetensors.torch import load_file

        # Load the state dict from file safely
        state_dict = load_file(model_file, device=map_location)

        # Apply key transformations
        transformed_state_dict = cls._transform_state_dict_keys(state_dict)

        # Apply tiling of linear input weights if needed
        model._tile_linear_input_weight(transformed_state_dict)

        # Load the transformed state dict
        msg = model.load_state_dict(transformed_state_dict, strict=strict)

        # Log message
        log_model_loading_keys(msg.missing_keys, msg.unexpected_keys)
        return model

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def get_vlm_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the VLM without the action expert to get the VLM cross attention tokens for the given batch."""
        self.eval()
        images, img_masks = self.prepare_cloud_vlm_images(batch)
        prompt_tokens, prompt_is_pad = self.prepare_prompt(batch)
        return self.model.sample_vlm_tokens(images, img_masks, prompt_tokens, ~prompt_is_pad)

    def update_vlm_token_cache(self, vlm_tokens: tuple):
        """Update the VLM token cache with the given tokens."""
        self.vlm_token_cache = vlm_tokens

    def _get_frozen_actions(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Get the frozen actions from the action queue"""
        # Convert deque to list to enable slicing
        action_list = list(self._action_queue)
        if len(action_list) > 0:
            frozen_actions = torch.stack(
                action_list[: self.config.frozen_actions]
            )  # shape: (num_real_frozen_actions, batch_size, action_dim)
        else:
            frozen_actions = torch.zeros(0, batch_size, self.config.max_action_dim)
        num_real_frozen_actions = frozen_actions.shape[0]
        num_noop_padding = self.config.frozen_actions - num_real_frozen_actions

        # if there are not enough frozen actions in the queue, pad with zeros
        if num_real_frozen_actions < self.config.frozen_actions:
            frozen_actions = torch.cat(
                [torch.zeros(num_noop_padding, batch_size, self.config.max_action_dim), frozen_actions]
            )

        # create pad mask for the frozen actions
        frozen_action_is_pad = torch.cat(
            [torch.ones(batch_size, num_noop_padding), torch.zeros(batch_size, num_real_frozen_actions)],
            dim=1,
        )
        frozen_action_is_pad = frozen_action_is_pad.to(dtype=torch.bool)

        frozen_actions = rearrange(
            frozen_actions, "s b a -> b s a"
        )  # shape: (batch_size, self.config.frozen_actions, action_dim)

        return frozen_actions, frozen_action_is_pad

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # repopulate the action queue if it is smaller than the safety buffer
        if len(self._action_queue) <= self.config.safety_buffer:
            # ONNX cannot trace python control flow, so the `if`-condition will be a constant true
            # if we trace `select_action` directly. Hence, we should trace `sample_actions` instead.
            frozen_actions, frozen_action_is_pad = self._get_frozen_actions(batch["state"].shape[0])
            batch["frozen_actions"] = frozen_actions.to(device=batch["state"].device)
            batch["frozen_action_is_pad"] = frozen_action_is_pad.to(device=batch["state"].device)
            actions = self.sample_actions(batch, noise)

            # empty the more distant actions from the action queue until there are only `frozen_actions` left
            while len(self._action_queue) > self.config.frozen_actions:
                self._action_queue.pop()

            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    def sample_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        vlm_token_cache_override: tuple | None = None,
    ):
        r"""vlm_token_cache_override is used to override the VLM token cache when the model is converted to ONNX"""
        batch = self.normalize_inputs(batch)
        frozen_actions = self.normalize_targets({"actions": batch["frozen_actions"]})["actions"]

        # If we are running in unified mode, the vlm token cache is updated everytime we sample actions
        # If we are running in robot mode, the vlm token cache needs to be updated asynchronously everytime the VLM runs.
        if self.execution_target is None:
            vlm_tokens = self.get_vlm_tokens(batch)
            self.update_vlm_token_cache(vlm_tokens)

        state = batch["state"]
        past_key_values, prefix_pad_masks, prefix_offsets, num_cross_att_tokens = (
            vlm_token_cache_override or self.vlm_token_cache
        )
        actions = self.model.sample_actions(
            [batch[f"local_camera{i}"] for i in range(self.config.action_expert_num_cams)],
            state,
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            prefix_offsets=prefix_offsets,
            num_cross_att_tokens=num_cross_att_tokens,
            frozen_actions=frozen_actions,
            frozen_action_is_pad=batch["frozen_action_is_pad"],
            noise=noise,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({"actions": actions})["actions"]
        # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
        # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
        actions = actions.transpose(0, 1)
        return actions

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        batch["frozen_actions"] = self.normalize_targets({"actions": batch["frozen_actions"]})["actions"]

        vlm_images, vlm_img_masks = self.prepare_cloud_vlm_images(batch)
        act_images, act_img_masks = self.prepare_action_expert_images(batch)
        state = batch["state"]
        prompt_tokens, prompt_is_pad = self.prepare_prompt(batch)
        response_tokens, response_is_pad = self.prepare_response(batch)
        actions = batch["actions"]
        actions_is_pad = batch["action_is_pad"]
        frozen_actions = batch["frozen_actions"]
        frozen_action_is_pad = batch["frozen_action_is_pad"]
        loss_type = batch["loss_type"]

        losses = self.model.forward(
            vlm_images,
            vlm_img_masks,
            act_images,
            act_img_masks,
            prompt_tokens,
            prompt_is_pad,
            response_tokens,
            response_is_pad,
            state,
            actions,
            frozen_actions,
            loss_type,
            actions_is_pad,
            frozen_action_is_pad,
            noise,
            time,
        )

        return losses

    def prepare_cloud_vlm_images(self, batch: dict):
        return self._prepare_images(batch, "camera", "img_is_pad")

    def prepare_action_expert_images(self, batch: dict):
        return self._prepare_images(batch, "local_camera", "local_img_is_pad")

    def _prepare_images(self, batch: dict, img_key_prefix: str, img_is_pad_key: str):
        """Apply TAU0 preprocessing to the images and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        The img_masks expected by the model contains 1 for real image and 0 for padded images.
        """
        images = []
        camera_keys = [input for input in self.config.input_features if input.startswith(img_key_prefix)]

        # Preprocess image features present in the batch
        for key in camera_keys:
            img = batch[key]

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0
            images.append(img)

        # invert pad mask to get img_mask expected by the model
        img_masks = ~batch[img_is_pad_key]

        # model expects shape (num_cams, batch_size)
        img_masks = rearrange(img_masks, "b n -> n b")

        return images, img_masks

    def prepare_prompt(self, batch: dict):
        return self._prepare_language(batch, "prompt", self.config.tokenizer_max_length)

    def prepare_response(self, batch: dict):
        return self._prepare_language(batch, "response", self.config.response_max_tokens)

    def _prepare_language(self, batch: dict, key: str, max_tokens: int) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        langs = batch[key]

        # PaliGemma prompt has to end with a new line
        langs = [lang if lang.endswith("\n") else f"{lang}\n" for lang in langs]

        tokenizer_kwargs = dict(  # noqa: C408
            padding="max_length",
            padding_side="right",
            max_length=max_tokens,
            return_tensors="pt",
        )
        try:
            tokenized_prompt = self.language_tokenizer(langs, **tokenizer_kwargs)
        except ValueError as e:
            tokenized_prompt = self.language_tokenizer(langs, **tokenizer_kwargs, truncation=True)
            # if the original tokenization failed but adding truncation worked, we log a warning
            warning_msg = (
                f"Unable to tokenize `{key}` with max length {max_tokens}. Truncated to max length. \n"
                f"Batch={langs}. \n"
                f'Original error = "{e}"'
            )
            # e.__cause__ should exist and contain info regarding the suggested max length.
            # Still we do this checking just in case.
            while e.__cause__:
                e = e.__cause__
                warning_msg += f', caused by "{e}"'
            warning_msg += "."
            logging.warning(warning_msg)

        # TODO: change the "to device" to allow accelerate to handle the device placement
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(
            device=device, dtype=torch.bool
        )  # false for pad tokens
        lang_is_pad = ~lang_masks  # true for pad tokens

        return lang_tokens, lang_is_pad


class TAU0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config: TAU0Config, execution_target: Optional[str] = None):
        super().__init__()
        self.config = config

        load_pretrained_paligemma = (
            self.config.init_strategy == "expert_only_he_init"
        )  # only load pretrained paligemma if we are He-initializing the expert only
        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            load_pretrained_paligemma=load_pretrained_paligemma,
            use_cache_layer=self.config.use_cache_layer,
            dropout=self.config.dropout,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config, execution_target)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.frozen_action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.noop_action_emb = nn.Embedding(1, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

        self._init_model()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def _init_weights(self, module):
        """Initialize weights using He (Kaiming) initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_model(self):
        """Initialize the model weights."""
        if self.config.init_strategy == "no_init":
            return
        elif self.config.init_strategy == "full_he_init":
            for m in self.modules():
                self._init_weights(m)
        elif self.config.init_strategy == "expert_only_he_init":
            for m in chain(
                self.paligemma_with_expert.gemma_expert.modules(),
                self.paligemma_with_expert.onboard_vision_encoder.modules(),
            ):
                self._init_weights(m)
        else:
            raise ValueError(f"Invalid init strategy: {self.config.init_strategy}")

    def set_execution_target(self, execution_target: str):
        """Used for setting the execution target of pretrained models"""
        self.paligemma_with_expert.set_execution_target(execution_target)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, prompt_tokens, prompt_masks, response_tokens, response_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        prompt_masks and response_masks should be True for non-padded tokens and False for padded tokens.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)

            # image embeddings don't need to be unnormalized because `fix/lerobot_openpi` branch of huggingface
            # already removed the normalization inside PaliGemma
            pass

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        prompt_emb = self.paligemma_with_expert.embed_language_tokens(prompt_tokens)
        response_emb = self.paligemma_with_expert.embed_language_tokens(response_tokens)

        # Normalize language embeddings
        lang_emb_dim = prompt_emb.shape[-1]
        prompt_emb = prompt_emb * math.sqrt(lang_emb_dim)
        response_emb = response_emb * math.sqrt(lang_emb_dim)

        embs.append(prompt_emb)
        pad_masks.append(prompt_masks)
        embs.append(response_emb)
        pad_masks.append(response_masks)

        # full attention between image and language inputs
        att_masks += [0] * prompt_emb.shape[1] + [1] * response_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, images, image_masks, state, noisy_actions, frozen_actions, frozen_action_is_pad, timestep
    ):
        r"""Embed local image, state, noisy_actions, timestep to prepare for Expert Gemma processing.
        images: list of length #action_cams, consisting of tensors with shape (batch_size, channel, height, width)
        image_masks: boolean tensor with shape (# action cams, batch_size). A value of 1 indicates that the image is valid, and 0 indicates that the image is padded.
        state: tensor with shape (batch_size, max_state_dim)
        noisy_actions: predicted actions tensor with shape (batch_size, chunk_size, max_action_dim)
        frozen_actions: frozen actions tensor with shape (batch_size, frozen_actions, max_action_dim)
        frozen_action_is_pad: boolean tensor with shape (batch_size, frozen_actions). A value of 1 indicates that the frozen action is padded
        timestep: tensor with shape (batch_size,)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # embed local images
        for img, img_mask in zip(images, image_masks, strict=False):
            img_emb = self.paligemma_with_expert.onboard_vision_encoder(img)
            assert img_emb.shape[1] == self.config.num_local_image_tokens, (
                f"Expected {self.config.num_local_image_tokens} tokens, got {img_emb.shape[1]}"
            )
            embs.append(img_emb)

            pad_masks.append(img_mask[:, None].expand(-1, self.config.num_local_image_tokens))
            att_masks += [0] * self.config.num_local_image_tokens

        # Embed frozen actions
        noop_emb = self.noop_action_emb(torch.tensor(0, device=device)).to(dtype=dtype)
        frozen_actions = frozen_actions.to(dtype=dtype)
        frozen_actions[frozen_action_is_pad] = noop_emb
        frozen_actions_emb = self.frozen_action_in_proj(frozen_actions)
        embs.append(frozen_actions_emb)
        pad_masks.append(torch.ones(bsize, self.config.frozen_actions, dtype=torch.bool, device=device))
        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [0] * self.config.frozen_actions

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        noisy_actions = noisy_actions.to(dtype=dtype)
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Nothing should attend to noisy actions
        att_masks += [1] + [0] * (self.config.chunk_size - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        return embs, pad_masks, att_masks

    def forward(
        self,
        cloud_vlm_images,
        cloud_vlm_img_masks,
        action_expert_images,
        action_expert_img_masks,
        prompt_tokens,
        prompt_is_pad,
        response_tokens,
        response_is_pad,
        state,
        actions,
        frozen_actions,
        loss_type,
        actions_is_pad,
        frozen_action_is_pad,
        noise=None,
        time=None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        device = actions.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)

        if time is None:
            time = self.sample_time(actions.shape[0], device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            cloud_vlm_images,
            cloud_vlm_img_masks,
            prompt_tokens,
            ~prompt_is_pad,
            response_tokens,
            ~response_is_pad,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            action_expert_images,
            action_expert_img_masks,
            state,
            x_t,
            frozen_actions,
            frozen_action_is_pad,
            time,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = (
            self.config.n_cross_att_tokens
            if self.config.n_cross_att_tokens
            else prefix_embs.shape[1] - self.config.response_max_tokens
        )
        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        # We should skip the response tokens when numbering the position ids for the action expert
        prefix_offsets = torch.sum(prefix_pad_masks[:, : -self.config.response_max_tokens], dim=-1)[
            :, None
        ]  # action expert position ids start after prefix
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            inputs_embeds=[prefix_embs, suffix_embs],
            n_cross_att_tokens=num_cross_att_tokens,
            vlm_attention_mask=vlm_2d_attention_mask,
            vlm_position_ids=vlm_position_ids,
            action_expert_attention_mask=action_expert_2d_attention_mask,
            action_expert_position_ids=action_expert_position_ids,
        )

        # compute final actions (flow vector field)
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)  # upcast to float32 for loss calculation

        # compute final language response tokens
        prefix_out = prefix_out[:, -self.config.response_max_tokens - 1 : -1]  # -1 for next token prediction
        logits = self.paligemma_with_expert.paligemma.lm_head(prefix_out)

        # --- Loss Calculation ---
        # Calculate MSE loss
        u_t = u_t.to(dtype=v_t.dtype)
        mse_loss = F.mse_loss(u_t, v_t, reduction="none")

        # remove pad tokens
        in_episode_bound = ~actions_is_pad
        mse_loss = mse_loss * in_episode_bound.unsqueeze(-1)

        # compute mean
        batch_size, seq_len = response_tokens.shape
        mse_loss = mse_loss.mean()
        mse_loss_count = loss_type.count("MSE")
        if mse_loss_count > 0:
            mse_loss = mse_loss * batch_size / mse_loss_count

        # Calculate Cross-Entropy loss
        labels = response_tokens
        logits = logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        logits = rearrange(logits, "b s d -> (b s) d")
        labels = rearrange(labels, "b s -> (b s)")
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        ce_loss = rearrange(ce_loss, "(b s) -> b s", b=batch_size, s=seq_len)

        # remove pad tokens
        ce_loss = ce_loss * ~response_is_pad

        # remove residual CE loss on empty response strings from robotic data
        ce_loss_mask = torch.tensor(
            [lt == "CE" for lt in loss_type], dtype=ce_loss.dtype, device=ce_loss.device
        )
        ce_loss_mask = rearrange(ce_loss_mask, "b -> b 1")
        ce_loss = ce_loss * ce_loss_mask

        # compute mean
        ce_loss = ce_loss.mean()
        ce_loss_count = loss_type.count("CE")
        if ce_loss_count > 0:
            ce_loss = ce_loss * batch_size / ce_loss_count

        return {"MSE": mse_loss, "CE": ce_loss}

    def sample_vlm_tokens(self, images, img_masks, lang_tokens, lang_masks) -> Tensor:
        # when sampling, we don't have response tokens, so we create empty tensors
        batch_size = lang_tokens.size(0)
        empty_response_tokens = torch.empty(
            (batch_size, 0), dtype=lang_tokens.dtype, device=lang_tokens.device
        )
        empty_response_masks = torch.empty((batch_size, 0), dtype=lang_masks.dtype, device=lang_masks.device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, empty_response_tokens, empty_response_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # compute prefix offsets and number of cross-attention tokens
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        num_cross_att_tokens = (
            # NOTE: DO NOT subtract by response_max_tokens here, because we don't have response in tokens.
            self.config.n_cross_att_tokens if self.config.n_cross_att_tokens else prefix_embs.shape[1]
        )

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            vlm_attention_mask=prefix_att_2d_masks,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
        )

        return past_key_values, prefix_pad_masks, prefix_offsets, num_cross_att_tokens

    def sample_actions(
        self,
        images,
        state,
        past_key_values,
        prefix_pad_masks,
        prefix_offsets,
        num_cross_att_tokens,
        frozen_actions,
        frozen_action_is_pad,
        noise=None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)

        Args:
            images: The images to sample actions from.
            state: The state to sample actions from.
            past_key_values: The past key values from the VLM.
            prefix_pad_masks: The prefix pad masks from the VLM.
            prefix_offsets: The prefix offsets to compute the position ids.
            num_cross_att_tokens: The number of cross attention tokens used.
            frozen_actions: The frozen actions to condition on.
            frozen_action_is_pad: The frozen actions is pad mask.
            noise: The noise to sample actions from.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, device=device)

        x_t = noise
        time = torch.tensor(1.0, device=device)
        # Can't use `while time >= dt / 2` because torch.onnx doesn't support dynamic graph
        for _ in range(self.config.num_steps):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                images,
                state,
                prefix_pad_masks,
                prefix_offsets,
                num_cross_att_tokens,
                past_key_values,
                x_t,
                frozen_actions,
                frozen_action_is_pad,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        images,
        state,
        prefix_pad_masks,
        prefix_offsets,
        num_cross_att_tokens,
        past_key_values,
        x_t,
        frozen_actions,
        frozen_action_is_pad,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # During inference, the images are never padded, so we can use a dummy mask.
        dummy_img_masks = torch.ones(
            (len(images), images[0].shape[0])
            if images
            else (1),  # dummy mask for when action_expert_num_cams is 0
            dtype=torch.bool,
            device=images[0].device if images else state.device,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            images, dummy_img_masks, state, x_t, frozen_actions, frozen_action_is_pad, timestep
        )

        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            action_expert_attention_mask=action_expert_2d_attention_mask,
            action_expert_position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            n_cross_att_tokens=num_cross_att_tokens,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        return v_t
