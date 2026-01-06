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

"""Value Function Model using SIGLIP and Gemma 3 270M

A value function model that estimates state values for reinforcement learning.
Uses SIGLIP for vision encoding and Gemma 3 270M for language processing.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoTokenizer

from opentau.policies.normalize import Normalize
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.value.configuration_value import ValueConfig
from opentau.policies.value.siglip_gemma import (
    SiglipGemmaValueConfig,
    SiglipGemmaValueModel,
)


def make_att_2d_masks(pad_masks, att_masks):
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
        torch.Tensor: The 2D attention masks.

    Raises:
        ValueError: If input masks are not 2D.
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


def resize_with_pad(img, width, height, pad_value=-1):
    """Resizes an image while preserving aspect ratio and padding to target dimensions.

    Args:
        img: Input image tensor of shape (B, C, H, W).
        width: Target width.
        height: Target height.
        pad_value: Value to use for padding. Defaults to -1.

    Returns:
        torch.Tensor: Resized and padded image tensor.

    Raises:
        ValueError: If image dimensions are not 4D (B, C, H, W).
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


class ValueFunction(PreTrainedPolicy):
    """Wrapper class around Value Function model to train and run inference within OpenTau."""

    config_class = ValueConfig
    name = "value"

    def __init__(
        self,
        config: ValueConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the ValueFunction policy.

        Args:
            config: Value Function configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        self.model = ValueModel(config)

    def reset(self):
        """Resets the internal state of the policy.

        This method is called at the beginning of each episode. For value functions,
        there is no internal state to reset.
        """
        pass  # Value functions don't need state reset

    def get_optim_params(self) -> dict:
        """Returns the parameters to be optimized.

        Returns:
            dict: Dictionary of parameters to optimize.
        """
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Selects an action based on the current policy.

        Args:
            batch: Dictionary containing observation data.

        Returns:
            Tensor: The selected action.

        Raises:
            NotImplementedError: Value functions do not select actions.
        """
        raise NotImplementedError("Value functions do not select actions. Use predict_value() instead.")

    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor = None):
        """Samples actions from the policy.

        Args:
            batch: Dictionary containing observation data.
            noise: Optional noise tensor.

        Raises:
            NotImplementedError: Value functions do not sample actions.
        """
        raise NotImplementedError("Value functions do not sample actions. Use predict_value() instead.")

    def calculate_value(self, logits: Tensor) -> Tensor:
        """Calculates the expected value from the logits distribution.

        Args:
            logits: Tensor containing the logits for value bins.

        Returns:
            Tensor: The expected value.
        """
        start_idx = torch.linspace(
            -1,
            -1 / self.config.reward_config.number_of_bins,
            self.config.reward_config.number_of_bins,
            device=logits.device,
        )
        end_idx = torch.linspace(
            -1 + 1 / self.config.reward_config.number_of_bins,
            0,
            self.config.reward_config.number_of_bins,
            device=logits.device,
        )

        mid_idx = rearrange(
            (start_idx + end_idx) / 2,
            "n -> 1 n",
        )

        value = torch.softmax(logits, dim=-1).to(dtype=torch.float32) @ mid_idx.T

        return rearrange(value, "b 1 -> b")

    @torch.no_grad()
    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict value estimates given environment observations.

        Args:
            batch: Dictionary containing observations (images, state, prompt)

        Returns:
            Tensor of shape [batch_size, 1] containing value estimates
        """
        self.eval()

        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        state = batch.get("state")

        logits = self.model.forward(images, img_masks, lang_tokens, lang_masks, state)
        return self.calculate_value(logits)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Do a full training forward pass to compute the value loss.

        Args:
            batch: Dictionary containing observations and target values

        Returns:
            Tuple of (loss_dict, None) where loss_dict contains the MSE loss
        """
        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        state = batch.get("state")

        logits = self.model.forward(images, img_masks, lang_tokens, lang_masks, state)
        values = self.calculate_value(logits)
        # Compute Cross-Entropy loss
        logits = logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        batch["return_bin_idx"] = batch["return_bin_idx"].to(dtype=torch.long)
        loss = F.cross_entropy(logits, batch["return_bin_idx"])

        l1_loss = F.l1_loss(values, batch["return_continuous"])

        accuracy = (logits.argmax(dim=-1) == batch["return_bin_idx"]).float().mean()

        return {
            "MSE": torch.zeros_like(loss, requires_grad=False),
            "CE": loss,
            "L1": l1_loss,
            "Accuracy": accuracy,
        }

    def prepare_images(self, batch):
        """Preprocesses images for the model.

        Resizes images to 224x224, pads to keep aspect ratio, and converts pixel range
        from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP. Also handles missing images
        by creating empty placeholders.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            tuple: A tuple containing a list of image tensors and a list of mask tensors.

        Raises:
            ValueError: If all image features are missing from the batch.
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

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenizes the text input for the model.

        Args:
            batch: Dictionary containing batch data, including "prompt".

        Returns:
            tuple: A tuple containing language token tensors and attention mask tensors.
        """
        device = batch.get("state", list(batch.values())[0]).device
        tasks = batch["prompt"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

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


class ValueModel(nn.Module):
    """
    Value Function Model using SIGLIP and Gemma 3 270M

    Estimates state values for reinforcement learning by processing:
    - Images through SIGLIP vision encoder
    - Language tokens through Gemma 3 270M
    - Optional state information

    ┌──────────────────────────────┐
    │               value          │
    │               ▲              │
    │              ┌┴─────┐        │
    │              │Gemma │        │
    │              │3 270M│        │
    │              │      │        │
    │ ┌──────────┐ └▲──▲──┘        │
    │ │          │  │  │           │
    │ │  SIGLIP  ├──┘  │           │
    │ │          │     language    │
    │ └────▲─────┘                 │
    │      │                       │
    │      image(s)                │
    │                              │
    └──────────────────────────────┘
    """

    CLASSIFICATION_TOKEN_ID = 6  # unused token id in Gemma 3 270M that we repurpose for classification

    def __init__(self, config):
        """Initializes the ValueModel.

        Args:
            config: Configuration object for the model.
        """
        super().__init__()
        self.config = config

        siglip_gemma_value_config = SiglipGemmaValueConfig(
            num_value_bins=self.config.reward_config.number_of_bins
        )
        self.siglip_gemma_value = SiglipGemmaValueModel(siglip_gemma_value_config)

        # Projection for state if provided
        self.state_proj = nn.Linear(self.config.max_state_dim, 640)
        self.multi_modal_proj = nn.Linear(1152, 640)
        self.bins = config.reward_config.number_of_bins
        self.c_neg = config.reward_config.C_neg

    def embed_sequence(
        self, images, img_masks, lang_tokens, lang_masks, state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embeds sequence of images and language tokens.

        Prepares embeddings for SiglipGemmaValueModel transformer processing.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            state: State tensor.

        Returns:
            tuple: A tuple containing embeddings, padding masks, and attention masks.
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
            img_emb = self.siglip_gemma_value.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            img_emb = self.multi_modal_proj(img_emb)

            # image embeddings don't need to be unnormalized because they were not normalized in the first place
            pass

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Gemma3 already scales by sqrt(d)
        lang_emb = self.siglip_gemma_value.embed_language_tokens(lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])

        state_mask = torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state_emb.device)
        pad_masks.append(state_mask)

        # full attention between state and image and language inputs
        att_masks += [0]

        # add classification token
        cls_token = torch.full(
            (bsize, 1), self.CLASSIFICATION_TOKEN_ID, device=state_emb.device, dtype=torch.long
        )
        cls_token_emb = self.siglip_gemma_value.gemma.embed_tokens(cls_token)
        embs.append(cls_token_emb)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device))
        att_masks += [0]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict value estimates given observations.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            lang_tokens: Language token IDs
            lang_masks: Language attention masks
            state: Optional state tensor

        Returns:
            Tensor of shape [batch_size, 1] containing value estimates
        """
        embs, pad_masks, att_masks = self.embed_sequence(images, img_masks, lang_tokens, lang_masks, state)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        logits = self.siglip_gemma_value.forward(
            inputs_embeds=embs,
            attention_mask=att_2d_masks,
            position_ids=position_ids,
        )

        return logits
