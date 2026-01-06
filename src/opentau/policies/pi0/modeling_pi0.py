#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.utils import log_model_loading_keys
from opentau.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device: torch.device | str = "cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: A 1-D tensor of shape (batch_size,).
        dimension: The dimension of the embedding vectors. Must be divisible by 2.
        min_period: The minimum period of the sinusoidal functions.
        max_period: The maximum period of the sinusoidal functions.
        device: The device to create the tensors on. Defaults to "cpu".

    Returns:
        A tensor of shape (batch_size, dimension) containing the positional embeddings.

    Raises:
        ValueError: If dimension is not divisible by 2 or if time tensor is not 1-D.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = (
        get_safe_dtype(torch.float64, device.type)
        if isinstance(device, torch.device)
        else get_safe_dtype(torch.float64, device)
    )
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Creates a 2-D attention mask given padding and 1-D attention masks.

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
        pad_masks: bool[B, N] true if its part of the input, false if padding.
        att_masks: int32[B, N] mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.

    Returns:
        A 2D attention mask tensor of shape (B, N, N).

    Raises:
        ValueError: If att_masks or pad_masks are not 2D.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img: Tensor, width: int, height: int, pad_value: int = -1) -> Tensor:
    """Resizes an image to fit within the specified dimensions while maintaining aspect ratio,
    and pads the remaining area with the specified value.

    Args:
        img: Input image tensor of shape (batch_size, channels, current_height, current_width).
        width: Target width.
        height: Target height.
        pad_value: Value to use for padding. Defaults to -1.

    Returns:
        The resized and padded image tensor of shape (batch_size, channels, height, width).

    Raises:
        ValueError: If the input image tensor does not have 4 dimensions.
    """
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pads the last dimension of a vector to a new size with zeros.

    Args:
        vector: Input tensor. Can be (batch_size x sequence_length x features_dimension)
            or (batch_size x features_dimension).
        new_dim: The new size for the last dimension.

    Returns:
        The padded tensor.
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within OpenTau."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the PI0Policy.

        Args:
            config: Policy configuration class instance.
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

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.model = PI0FlowMatching(config)

        self.reset()

    def reset(self) -> None:
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

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

        Args:
            state_dict: The state dictionary to transform.

        Returns:
            The transformed state dictionary.
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
        cls, model: "PI0Policy", model_file: str, map_location: str, strict: bool
    ) -> "PI0Policy":
        """Override to apply key transformations before loading.

        Args:
            model: The model instance.
            model_file: Path to the model file.
            map_location: Device mapping location.
            strict: Whether to strictly enforce state dict matching.

        Returns:
            The loaded model instance.
        """
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
        """Returns the parameters to be optimized.

        Returns:
            A generator over the model parameters.
        """
        return self.parameters()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the from_pretrained method to display important disclaimer.

        Args:
            *args: Positional arguments passed to super().from_pretrained.
            **kwargs: Keyword arguments passed to super().from_pretrained.

        Returns:
            The loaded model instance.
        """
        print(
            "⚠️  DISCLAIMER: The PI0 model is ported from JAX by the Hugging Face team. \n"
            "   It is not expected to perform as well as the original implementation. \n"
            "   Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        return super().from_pretrained(*args, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Batch of data containing environment observations.

        Returns:
            The predicted action chunk.

        Raises:
            NotImplementedError: Always, as this method is not implemented for PI0.
        """
        raise NotImplementedError("Currently not implemented for PI0")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        Args:
            batch: Batch of data containing environment observations.
            noise: Optional noise tensor to be used during sampling.

        Returns:
            The selected action tensor.
        """
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) <= self.config.safety_buffer:
            actions = self.sample_actions(batch, noise=noise)
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Sample actions from the policy given environment observations.

        Args:
            batch: Batch of data containing environment observations.
            noise: Optional noise tensor.

        Returns:
            The sampled actions tensor of shape (batch_size, action_dim).
        """
        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        state = batch["state"]
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
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

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Batch of data containing environment observations, actions, and targets.
            noise: Optional noise tensor.
            time: Optional time tensor.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE").
        """
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = batch["state"]
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = batch["actions"]
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]

        # For backward pass
        loss = losses.mean()

        return {"MSE": loss, "CE": torch.zeros_like(loss, requires_grad=True)}

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Apply Pi0 preprocessing to the images.

        Resizes to 224x224 and padding to keep aspect ratio, and converts pixel range
        from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.

        Args:
            batch: Batch of data containing image tensors.

        Returns:
            A tuple containing:
                - images: A list of processed image tensors.
                - img_masks: A list of image mask tensors.

        Raises:
            ValueError: If no image features are present in the batch.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenize the text input.

        Args:
            batch: Batch of data containing "prompt" and potentially "advantage".

        Returns:
            A tuple containing:
                - lang_tokens: Tensor of language tokens.
                - lang_masks: Tensor of language attention masks.
        """
        device = batch["state"].device
        tasks = batch["prompt"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        for idx, task in enumerate(tasks):
            if self.config.advantage == "on":  # always add positive advantage
                tasks[idx] = f"{task}Advantage: positive\n"
            elif self.config.advantage == "use":  # add advantage based on threshold
                adv = batch["advantage"][idx] >= self.config.advantage_threshold
                adv = "positive" if adv else "negative"
                tasks[idx] = f"{task}Advantage: {adv}\n"

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)

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

    def __init__(self, config: PI0Config):
        """Initializes the PI0FlowMatching model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        load_pretrained_paligemma = (
            self.config.init_strategy == "expert_only_he_init"
        )  # only load pretrained paligemma if we are He-initializing the expert only
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            load_pretrained_paligemma=load_pretrained_paligemma,
            dropout=self.config.dropout,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

        self._init_model()

    def set_requires_grad(self) -> None:
        """Sets the requires_grad attribute for state projection parameters."""
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using He (Kaiming) initialization.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_model(self) -> None:
        """Initialize the model weights based on the configuration."""
        if self.config.init_strategy == "no_init":
            return
        elif self.config.init_strategy == "full_he_init":
            for m in self.modules():
                self._init_weights(m)
        elif self.config.init_strategy == "expert_only_he_init":
            for m in self.paligemma_with_expert.gemma_expert.modules():
                self._init_weights(m)
        else:
            raise ValueError(f"Invalid init strategy: {self.config.init_strategy}")

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        """Samples Gaussian noise.

        Args:
            shape: The shape of the noise tensor.
            device: The device to create the tensor on.

        Returns:
            A tensor containing the sampled noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize: int, device: torch.device | str) -> Tensor:
        """Samples time steps from a Beta distribution.

        Args:
            bsize: Batch size.
            device: The device to create the tensor on.

        Returns:
            A tensor containing the sampled time steps.
        """
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.

        Returns:
            A tuple containing:
                - embs: Concatenated embeddings tensor.
                - pad_masks: Concatenated padding masks tensor.
                - att_masks: Attention masks tensor.
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
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # image embeddings don't need to be unnormalized because `fix/lerobot_openpi` branch of huggingface
            # already removed the normalization inside PaliGemma
            pass

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self, state: Tensor, noisy_actions: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            state: State tensor.
            noisy_actions: Tensor containing noisy actions.
            timestep: Tensor containing timesteps.

        Returns:
            A tuple containing:
                - embs: Concatenated embeddings tensor.
                - pad_masks: Concatenated padding masks tensor.
                - att_masks: Attention masks tensor.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

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

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors).

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            state: State tensor.
            actions: Action tensor.
            noise: Optional noise tensor.
            time: Optional time tensor.

        Returns:
            The computed loss tensor.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            state: State tensor.
            noise: Optional noise tensor.

        Returns:
            The sampled action tensor.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Apply one denoising step of the noise `x_t` at a given timestep.

        Args:
            state: State tensor.
            prefix_pad_masks: Prefix padding masks.
            past_key_values: Past key values from the VLM.
            x_t: Current noise tensor.
            timestep: Current timestep.

        Returns:
            The predicted velocity tensor (v_t).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
