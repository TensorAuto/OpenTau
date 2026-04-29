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

"""π07 High-Level Planner: A Vision-Language Model for Memory and Subtask Prediction.

This module implements the high-level planner for π07, built on top of the PaliGemma
VLM architecture. Given images, language instructions, robot state, and past memory,
the planner autoregressively predicts updated memory and a subtask string.
"""

import builtins
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.normalize import Normalize
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pi07_paligemma.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig,
)
from opentau.policies.pretrained import PreTrainedPolicy, T
from opentau.utils.accelerate_utils import get_proc_accelerator


def _preferred_dtype() -> torch.dtype:
    """Returns the preferred compute dtype for the current execution context.

    Returns:
        ``torch.float32`` during ONNX export, ``torch.bfloat16`` otherwise.
    """
    return torch.float32 if torch.onnx.is_in_onnx_export() else torch.bfloat16


def make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
    n_cross_att_tokens: int | None = None,
    cross_att_pad_masks: Tensor | None = None,
) -> Tensor:
    """Creates a 2-D attention mask given padding and 1-D attention masks.

    Tokens can attend to valid inputs tokens which have a cumulative `att_masks`
    smaller or equal to theirs. This way `att_masks` int[B, N] can be used to
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
        n_cross_att_tokens: Add attention mask for cross-attention tokens if
            `n_cross_att_tokens` is provided.
        cross_att_pad_masks: Padding masks for cross attention tokens. Required if
            `n_cross_att_tokens` is provided.

    Returns:
        A 2D attention mask tensor of shape (B, N + n_cross_att_tokens, N + n_cross_att_tokens)
        if n_cross_att_tokens is provided, else (B, N, N).

    Raises:
        ValueError: If att_masks or pad_masks are not 2D (including batch dimension).
        AssertionError: If cross_att_pad_masks is missing when n_cross_att_tokens is set,
            or if its shape is incorrect.
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
        # The cross_att_masks are concatenated before the att_2d_masks
        att_2d_masks = torch.cat((cross_att_mask, att_2d_masks), dim=2)

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


class PI07HighLevelPlannerPolicy(PreTrainedPolicy):
    """Policy wrapper for the π07 high-level planner.

    Handles input normalisation, tokenisation of language/memory/response,
    and delegates to :class:`PI07HighLevelPlannerModel` for autoregressive
    prediction of updated memory and subtask strings.
    """

    config_class = PI07HighLevelPlannerConfig
    name = "pi07_high_level_planner"

    def __init__(
        self,
        config: PI07HighLevelPlannerConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the PI07HighLevelPlannerPolicy.

        Args:
            config: Policy configuration instance.
            dataset_stats: Dataset statistics for input normalization. If not
                provided here, they must be supplied via ``load_state_dict``
                before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        # Get vocab size from processor
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI07HighLevelPlannerModel(config, discrete_action_vocab_size=discrete_action_vocab_size)

        self.reset()

    def reset(self) -> None:
        """Resets any internal state. Call when the environment resets."""
        pass

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping.

        Args:
            pretrained_name_or_path: Path to the pretrained model or its name on the Hub.
            config: Configuration object.
            force_download: Whether to force download the model weights.
            resume_download: Whether to resume download.
            proxies: Proxy configuration.
            token: Authentication token.
            cache_dir: Directory to cache downloaded files.
            local_files_only: Whether to only look for files locally.
            revision: Specific model revision.
            strict: Whether to strictly enforce state dict matching.
            **kwargs: Additional keyword arguments.

        Returns:
            The loaded model instance.

        Raises:
            ValueError: If pretrained_name_or_path is None.
        """
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        acc = get_proc_accelerator()
        is_main_process = acc.is_main_process if acc else True
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            if is_main_process:
                print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                if is_main_process:
                    print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                if is_main_process:
                    print(f"Could not load state dict from remote files: {e}")
                    print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model.") and "normalize" not in key:
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10 and is_main_process:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0 and is_main_process:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            if missing_keys and is_main_process:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 20:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:20]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 20} more")

            if unexpected_keys and is_main_process:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 20:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:20]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 20} more")

            if not missing_keys and not unexpected_keys and is_main_process:
                print("All keys loaded successfully!")

        except Exception as e:
            if is_main_process:
                print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, Tensor], model_config: PreTrainedConfig
    ) -> dict[str, Tensor]:  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture.

        Args:
            state_dict: The state dictionary to fix.
            model_config: The model configuration.

        Returns:
            The fixed state dictionary.
        """
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        """Returns the parameters to be optimized.

        Returns:
            A generator over the model parameters.
        """
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Not implemented for the high-level planner.

        Args:
            batch: Batch of data containing environment observations.

        Raises:
            NotImplementedError: Always, since the high-level planner predicts
                memory and subtask strings, not action chunks.
        """
        raise NotImplementedError("The high-level planner does not predict action chunks.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Not implemented for the high-level planner.

        Args:
            batch: Batch of data containing environment observations.

        Raises:
            NotImplementedError: Always, since the high-level planner predicts
                memory and subtask strings, not action chunks.
        """
        raise NotImplementedError("The high-level planner does not use select_action.")

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Run inference to predict updated memory and subtask tokens.

        Normalizes inputs, prepares image and language embeddings, then
        delegates to the inner model for autoregressive generation.

        Args:
            batch: Batch of observations. Expected keys include images,
                ``"prompt"``, ``"state"``, and ``"past_memory"``.

        Returns:
            A tuple ``(memory_tokens, response_tokens)`` where each is a
            ``Tensor`` of token IDs with shape ``(batch_size, seq_len)``.
        """

        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        metadata_tokens, metadata_masks = self.prepare_metadata(batch)

        memory_tokens, response_tokens = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, metadata_tokens, metadata_masks
        )

        return memory_tokens, response_tokens

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Runs a full training forward pass and computes the loss.

        Tokenizes images, language (with state and past memory), target memory,
        and target response, then computes cross-entropy losses for both the
        memory and response token predictions.

        Args:
            batch: Batch of training data. Expected keys include images,
                ``"prompt"``, ``"state"``, ``"past_memory"``, ``"response"``,
                and ``"next_memory"``.

        Returns:
            A dict with ``"MSE"`` (always zero, kept for interface
            compatibility) and ``"CE"`` (sum of memory and response
            cross-entropy losses).
        """
        batch = self.normalize_inputs(batch)

        images, img_masks = self.prepare_images(
            batch
        )  # in img_masks we have True for real images and False for padded images
        lang_tokens, lang_masks = self.prepare_language(
            batch
        )  # in lang_masks we have True for real tokens and False for padded tokens
        # response prediction is to predict the response . It will attend to image and language inputs.

        metadata_tokens, metadata_masks = self.prepare_metadata(
            batch
        )  # in metadata_masks we have True for real tokens and False for padded tokens
        response_tokens, response_masks = self.prepare_response(
            batch
        )  # in response_masks we have True for real tokens and False for padded tokens

        # memory prediction is to predict the memory . It will attend to image and language inputs.
        memory_tokens, memory_masks = self.prepare_next_memory(
            batch
        )  # in memory_masks we have True for real tokens and False for padded tokens
        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            response_tokens,
            response_masks,
            memory_tokens,
            memory_masks,
            metadata_tokens,
            metadata_masks,
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]

        return {"MSE": mse_loss, "CE": ce_loss}

    def prepare_discrete_state(self, batch: dict[str, Tensor]) -> list[str]:
        """Discretizes the state into bins and converts it to a string representation.

        Each dimension of the state vector is discretized into 256 bins.
        The values of each dimension of the state are expected to be in the range [-1, 1].
        The discretization bins are linearly spaced between -1 and 1.
        The index of the bin for each dimension is then concatenated into a space-separated string.

        Args:
            batch: Batch of data containing the "state" tensor.

        Returns:
            A list of strings, where each string is a space-separated list of discretized state values.

        Raises:
            ValueError: If the state values are not normalized between -1 and 1.
        """
        state = batch["state"]
        state_cpu = state.to(device="cpu", dtype=torch.float32)
        if torch.any(state_cpu < -1.0) or torch.any(state_cpu > 1.0):
            logging.warning(
                f"State values are not normalized between -1 and 1. Min: {state_cpu.min().item()}, Max: {state_cpu.max().item()}"
            )
        state_clipped = torch.clamp(state_cpu, -1.0, 1.0)
        # replicate np.digitize with torch for torch.compile compatibility
        bin_indices = ((state_clipped + 1.0) * 128.0).long().clamp(0, 255)
        discretized_states = bin_indices.cpu().tolist()
        return [
            " ".join(map(str, row)) for row in discretized_states
        ]  # TODO: return a tensor instead of a list of strings?

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Apply preprocessing to the images.

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
        """Tokenizes the composite language prompt.

        Builds a prompt string from the task instruction, past memory, and
        discretized robot state separated by ``<eos>`` tokens, then tokenizes
        and pads to ``prompt_max_length``.

        Args:
            batch: Batch containing ``"prompt"`` (task strings),
                ``"state"`` (state tensor), and ``"past_memory"`` (list of
                past memory strings).

        Returns:
            A tuple ``(lang_tokens, lang_masks)`` where:
                - lang_tokens: Token IDs of shape ``(batch_size, prompt_max_length)``.
                - lang_masks: Boolean attention mask of the same shape.
        """
        device = batch["state"].device
        tasks = batch["prompt"]

        # add state to the prompt
        state = self.prepare_discrete_state(batch)
        # using <eos> to separate each modality
        past_memory = batch["past_memory"]
        prompt = [
            f"Task: {task}, Past Memory: {past_mem}, State: {state}, "
            for task, past_mem, state in zip(tasks, past_memory, state, strict=False)
        ]
        tokenized_prompt = self.language_tokenizer.__call__(
            prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.prompt_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_metadata(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenizes the metadata for training.

        Wraps each metadata string with an ``<eos>`` suffix, then tokenizes and
        pads to ``metadata_max_length``.
        """

        metadata = []
        for speed, quality, mistake, speed_is_pad, quality_is_pad, mistake_is_pad in zip(
            batch["speed"],
            batch["quality"],
            batch["mistake"],
            batch["speed_is_pad"],
            batch["quality_is_pad"],
            batch["mistake_is_pad"],
            strict=True,
        ):
            segments = []
            if not speed_is_pad:
                segments.append(f"Speed: {str(speed.item())}, ")

            if not quality_is_pad:
                segments.append(f"Quality: {str(quality.item())}, ")

            if not mistake_is_pad:
                segments.append(f"Mistake: {str(mistake.item())}, ")

            metadata.append(f"Metadata: {' '.join(segments)}")

        device = batch["state"].device
        tokenized_metadata = self.language_tokenizer.__call__(
            metadata,
            padding="max_length",
            padding_side="right",
            max_length=self.config.metadata_max_length,
            return_tensors="pt",
            truncation=True,
        )
        metadata_tokens = tokenized_metadata["input_ids"].to(device=device)
        metadata_masks = tokenized_metadata["attention_mask"].to(device=device, dtype=torch.bool)

        return metadata_tokens, metadata_masks

    def prepare_response(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenizes the target subtask response for training.

        Wraps each response string with an ``<eos>Actions:`` suffix, then
        tokenizes and pads to ``response_max_length``.

        Args:
            batch: Batch containing ``"response"`` (list of subtask strings)
                and ``"state"`` (used only to determine the device).

        Returns:
            A tuple ``(response_tokens, response_masks)`` where:
                - response_tokens: Token IDs of shape
                  ``(batch_size, response_max_length)``.
                - response_masks: Boolean attention mask of the same shape
                  (``True`` for real tokens, ``False`` for padding).
        """

        device = batch["state"].device
        responses = batch["response"]

        # if '' is found in response then response is not for loss calculation (used for robotic dataset with no subtask), so add pad token to the response.
        response_prompt = [f"{response}<eos>" for response in responses]

        tokenized_response = self.language_tokenizer.__call__(
            response_prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.response_max_length,
            return_tensors="pt",
            truncation=True,
        )
        response_tokens = tokenized_response["input_ids"].to(device=device)
        response_masks = tokenized_response["attention_mask"].to(device=device, dtype=torch.bool)

        return response_tokens, response_masks

    def prepare_next_memory(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Tokenizes the target updated memory for training.

        Wraps each memory string with an ``<eos>`` suffix, then tokenizes and
        pads to ``memory_max_length``.

        Args:
            batch: Batch containing ``"next_memory"`` (list of target memory
                strings) and ``"state"`` (used only to determine the device).

        Returns:
            A tuple ``(memory_tokens, memory_masks)`` where:
                - memory_tokens: Token IDs of shape
                  ``(batch_size, memory_max_length)``.
                - memory_masks: Boolean attention mask of the same shape
                  (``True`` for real tokens, ``False`` for padding).
        """

        device = batch["state"].device
        next_memory = batch["next_memory"]

        # if '' is found in next_memory then it is not for loss calculation (used for robotic dataset with no subtask), so add pad token.
        memory_prompt = [f"{mem}<eos>" for mem in next_memory]

        tokenized_memory = self.language_tokenizer.__call__(
            memory_prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.memory_max_length,
            return_tensors="pt",
            truncation=True,
        )
        memory_tokens = tokenized_memory["input_ids"].to(device=device)
        memory_masks = tokenized_memory["attention_mask"].to(device=device, dtype=torch.bool)

        return memory_tokens, memory_masks


class PI07HighLevelPlannerModel(nn.Module):
    """π07 High-Level Planner inner model.

    Uses a PaliGemma VLM backbone to encode images and a composite language
    prompt (task + past context) and optional episode metadata, with fixed tokenizer
    spans ``";\\n "``, ``"Updated Memory: "``, and (in full training runs) ``"Subtask: "``
    before the predicted text, then autoregressively
    predicts updated memory and subtask text:

    1. **Updated memory** — next-token CE over ``memory_max_length`` slots after the
       ``"Updated Memory: "`` span.
    2. **Subtask (response)** — next-token CE over ``response_max_length`` slots after
       the ``"Subtask: "`` span (training).

    Inference mirrors training by inserting the live ``"Subtask: "`` token IDs into the
    KV cache after memory decoding and before response decoding.

    Architecture (rough dataflow)::

        ┌───────────────────────────────────────────┐
        │     response content (subtask text)       │
        │                   ▲                       │
        │  memory, ``Subtask: ``, lang, ``";\\n "``, images, … │
        │     ┌───────────────────────┐             │
        │     │       PaliGemma        │            │
        │     │  (autoregressive LM)   │            │
        │     └────────────────────────┘            │
        └───────────────────────────────────────────┘

    Args:
        config: High-level planner configuration.
        discrete_action_vocab_size: Vocabulary size for the discrete action
            tokenizer (passed through to ``PaliGemmaWithExpertModel``).
    """

    def __init__(self, config: PI07HighLevelPlannerConfig, discrete_action_vocab_size: int | None = None):
        """Initializes the PI07HighLevelPlannerModel.

        Args:
            config: High-level planner configuration.
            discrete_action_vocab_size: Vocabulary size for the discrete action
                tokenizer (passed through to ``PaliGemmaWithExpertModel``).
        """
        super().__init__()
        self.config = config

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=False,
            attention_implementation=self.config.attention_implementation,
            load_pretrained_paligemma=False,
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        memory_tokens: Tensor | None = None,
        memory_masks: Tensor | None = None,
        metadata_tokens: Tensor | None = None,
        metadata_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embeds and concatenates all prefix modalities for the transformer.

        Embeds images with SigLIP and language/metadata/memory/response spans with the
        PaliGemma embedding layer. **Concatenation order** (training when memory and response
        are provided):

        ``[images | language | metadata | ";\\n " | "Updated Memory: " | memory_tokens |
        "Subtask: " | response_tokens]``

        When ``memory_tokens`` / ``response_tokens`` are omitted (inference), only the
        fixed spans before those segments are present; memory and subtask text are filled in
        via KV-cache decoding plus an explicit ``"Subtask: "`` injection before response AR.

        Attention pattern (via ``att_masks`` cumsums):
            - Image + language tokens: bidirectional (``0``).
            - Metadata (if present): new bidirectional block (``[1, 0, …, 0]``).
            - ``";\\n "`` (same string as ``encode(";\n ", add_special_tokens=False)``): continues previous block (``0``).
            - ``"Updated Memory: "``: new bidirectional block (``[1, 0, …, 0]``).
            - Memory token slots: causal segment (``1`` per slot).
            - ``"Subtask: "`` (training): new block then causal continuation within span.
            - Response token slots: causal (``1`` per slot).

        Args:
            images: List of image tensors, one per camera.
            img_masks: List of boolean masks indicating real vs. padded images.
            lang_tokens: Language token IDs of shape ``(B, prompt_max_length)``.
            lang_masks: Boolean attention mask for language tokens.
            response_tokens: Optional subtask response token IDs of shape
                ``(B, response_max_length)``. Provided during training.
            response_masks: Optional boolean mask for response tokens.
            memory_tokens: Optional updated memory token IDs of shape
                ``(B, memory_max_length)``. Provided during training.
            memory_masks: Optional boolean mask for memory tokens.
            metadata_tokens: Optional metadata token IDs of shape
                ``(B, metadata_max_length)``.
            metadata_masks: Optional boolean mask for metadata tokens.

        Returns:
            A tuple ``(embs, pad_masks, att_masks)`` where:
                - embs: Concatenated embeddings ``(B, total_seq_len, D)``.
                - pad_masks: Boolean padding mask ``(B, total_seq_len)``.
                - att_masks: 1-D attention pattern ``(B, total_seq_len)``
                  used by :func:`make_att_2d_masks`.
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
            img_emb = img_emb.to(dtype=_preferred_dtype())

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

        if metadata_tokens is not None:
            metadata_emb = self.paligemma_with_expert.embed_language_tokens(metadata_tokens)
            metadata_emb_dim = metadata_emb.shape[-1]
            metadata_emb = metadata_emb * math.sqrt(metadata_emb_dim)
            embs.append(metadata_emb)
            pad_masks.append(metadata_masks)
            att_masks += [1] + [0] * (metadata_emb.shape[1] - 1)

        prefix_end_indicator_ids = self.language_tokenizer.encode(";\n ", add_special_tokens=False)
        prefix_end_tokens = torch.tensor(
            [prefix_end_indicator_ids] * bsize,
            device=lang_tokens.device,
            dtype=torch.long,
        )
        prefix_end_emb = self.paligemma_with_expert.embed_language_tokens(prefix_end_tokens)
        prefix_end_dim = prefix_end_emb.shape[-1]
        prefix_end_emb = prefix_end_emb * math.sqrt(prefix_end_dim)

        num_prefix_end_embs = prefix_end_emb.shape[1]
        prefix_end_mask = torch.ones(bsize, num_prefix_end_embs, dtype=torch.bool, device=lang_tokens.device)

        embs.append(prefix_end_emb)
        pad_masks.append(prefix_end_mask)
        att_masks += [0] * num_prefix_end_embs

        memory_start_indicator_ids = self.language_tokenizer.encode(
            "Updated Memory: ", add_special_tokens=False
        )
        memory_start_tokens = torch.tensor(
            [memory_start_indicator_ids] * bsize,
            device=lang_tokens.device,
            dtype=torch.long,
        )
        memory_start_emb = self.paligemma_with_expert.embed_language_tokens(memory_start_tokens)
        memory_start_dim = memory_start_emb.shape[-1]
        memory_start_emb = memory_start_emb * math.sqrt(memory_start_dim)

        num_memory_start_embs = memory_start_emb.shape[1]
        memory_start_mask = torch.ones(
            bsize, num_memory_start_embs, dtype=torch.bool, device=lang_tokens.device
        )

        embs.append(memory_start_emb)
        pad_masks.append(memory_start_mask)
        att_masks += [1] + [0] * (num_memory_start_embs - 1)

        if memory_tokens is not None:
            memory_emb = self.paligemma_with_expert.embed_language_tokens(memory_tokens)
            # Normalize memory language embeddings
            memory_emb_dim = memory_emb.shape[-1]
            memory_emb = memory_emb * math.sqrt(memory_emb_dim)

            embs.append(memory_emb)
            pad_masks.append(memory_masks)

            # full attention between image, language and memory inputs
            num_memory_embs = memory_emb.shape[1]
            att_masks += [1] * num_memory_embs

        if response_tokens is not None:
            response_start_indicator_ids = self.language_tokenizer.encode(
                "Subtask: ", add_special_tokens=False
            )
            response_start_tokens = torch.tensor(
                [response_start_indicator_ids] * bsize,
                device=lang_tokens.device,
                dtype=torch.long,
            )
            response_start_emb = self.paligemma_with_expert.embed_language_tokens(response_start_tokens)
            response_start_dim = response_start_emb.shape[-1]
            response_start_emb = response_start_emb * math.sqrt(response_start_dim)

            num_response_start_embs = response_start_emb.shape[1]
            response_start_mask = torch.ones(
                bsize, num_response_start_embs, dtype=torch.bool, device=lang_tokens.device
            )

            embs.append(response_start_emb)
            pad_masks.append(response_start_mask)
            att_masks += [1] + [0] * (num_response_start_embs - 1)

            response_emb = self.paligemma_with_expert.embed_language_tokens(response_tokens)

            # Normalize response language embeddings
            response_emb_dim = response_emb.shape[-1]
            response_emb = response_emb * math.sqrt(response_emb_dim)

            embs.append(response_emb)
            pad_masks.append(response_masks)

            # full attention between image, language and response inputs
            num_response_embs = response_emb.shape[1]
            att_masks += [1] * num_response_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        memory_tokens: Tensor | None = None,
        memory_masks: Tensor | None = None,
        metadata_tokens: Tensor | None = None,
        metadata_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass: embeds all modalities and computes CE losses.

        The prefix matches :meth:`embed_prefix` when memory and response tensors are set:
        fixed separators ``";\\n "``, ``"Updated Memory: "``, and ``"Subtask: "`` appear
        in addition to ``metadata``, ``memory_tokens``, and ``response_tokens``. CE slices use
        negative offsets from the **sequence tail**, relying on
        ``config.subtask_indicator_max_length`` so memory logits align with memory contents
        even though ``"Subtask: "`` sits between memory and response text.

        Args:
            images: List of image tensors, one per camera.
            img_masks: List of boolean masks for real vs. padded images.
            lang_tokens: Language token IDs ``(B, prompt_max_length)``.
            lang_masks: Boolean attention mask for language tokens.
            response_tokens: Subtask response token IDs
                ``(B, response_max_length)``.
            response_masks: Boolean mask for response tokens.
            memory_tokens: Updated memory token IDs
                ``(B, memory_max_length)``.
            memory_masks: Boolean mask for memory tokens.
            metadata_tokens: Optional metadata token IDs
                ``(B, metadata_max_length)``.
            metadata_masks: Optional boolean mask for metadata tokens.

        Returns:
            A dict with ``"MSE"`` (zero tensor, for interface compatibility)
            and ``"CE"`` (sum of memory and response cross-entropy losses).
        """
        # Run VLM first to get key value cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            response_tokens,
            response_masks,
            memory_tokens,
            memory_masks,
            metadata_tokens,
            metadata_masks,
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # avoids using discrete action for predicting continuous flow matching action
        num_cross_att_tokens = prefix_embs.shape[1]

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        batch_size, seq_len = response_tokens.shape
        response_token_start = -self.config.response_max_length
        # Slice covers only response **content** slots at the tail (after ``Subtask: ``).
        response_token_end = -1
        response_slice_object = slice(response_token_start, response_token_end)
        response_out = prefix_out[
            :,
            response_slice_object,
        ]
        response_logits = self.paligemma_with_expert.paligemma.lm_head(response_out)
        # response slice to exclude the <BOS> token from response while calculating loss.
        response_slice = slice(1, None)
        response_logits = response_logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        response_logits = rearrange(response_logits, "b s d -> (b s) d")
        response_labels = rearrange(response_tokens[:, response_slice], "b s -> (b s)")
        response_ce_loss = F.cross_entropy(response_logits, response_labels, reduction="none")

        response_ce_loss = rearrange(response_ce_loss, "(b s) -> b s", b=batch_size, s=seq_len - 1)

        # remove pad tokens
        response_is_pad = ~response_masks  # convert into format where value for pad is True
        # helps to control loss for response tokens in case of robotic data and VQA data
        response_ce_loss = response_ce_loss * ~response_is_pad[:, response_slice]

        # compute mean
        response_ce_loss = response_ce_loss.mean()

        batch_size, seq_len = memory_tokens.shape
        memory_token_start = (
            -self.config.memory_max_length
            - self.config.response_max_length
            - self.config.subtask_indicator_max_length
        )
        # Memory **content** span: immediately after ``Subtask: `` and response (from the end).
        memory_token_end = -self.config.response_max_length - self.config.subtask_indicator_max_length - 1
        memory_slice_object = slice(memory_token_start, memory_token_end)
        memory_out = prefix_out[
            :,
            memory_slice_object,
        ]
        memory_logits = self.paligemma_with_expert.paligemma.lm_head(memory_out)
        # memory slice to exclude the <BOS> token from memory while calculating loss.
        memory_slice = slice(1, None)
        memory_logits = memory_logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        memory_logits = rearrange(memory_logits, "b s d -> (b s) d")
        memory_labels = rearrange(memory_tokens[:, memory_slice], "b s -> (b s)")
        memory_ce_loss = F.cross_entropy(memory_logits, memory_labels, reduction="none")

        memory_ce_loss = rearrange(memory_ce_loss, "(b s) -> b s", b=batch_size, s=seq_len - 1)

        # remove pad tokens
        memory_is_pad = ~memory_masks  # convert into format where value for pad is True
        # helps to control loss for memory tokens in case of robotic data and VQA data
        memory_ce_loss = memory_ce_loss * ~memory_is_pad[:, memory_slice]

        # compute mean
        memory_ce_loss = memory_ce_loss.mean()

        ce_loss = response_ce_loss + memory_ce_loss

        return {"MSE": torch.zeros_like(ce_loss, requires_grad=False), "CE": ce_loss}

    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        metadata_tokens: Tensor | None = None,
        metadata_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Inference forward: autoregressively generates memory and subtask tokens.

        Runs ``memory_max_length`` ``infer_autoregressive`` steps, then feeds the same
        ``"Subtask: "`` token IDs used in training (tokenizer-dependent length
        ``subtask_indicator_max_length``) through the cache, then runs ``response_max_length``
        response steps. Each step conditions on prior KV-cache entries.

        Args:
            images: List of image tensors, one per camera.
            img_masks: List of boolean masks for real vs. padded images.
            lang_tokens: Language token IDs ``(B, prompt_max_length)``.
            lang_masks: Boolean attention mask for language tokens.
            metadata_tokens: Optional metadata token IDs
                ``(B, metadata_max_length)``.
            metadata_masks: Optional boolean mask for metadata tokens.

        Returns:
            A tuple ``(memory_tokens, response_tokens)`` where each is a
            ``Tensor`` of generated token IDs with shape
            ``(B, memory_max_length)`` and ``(B, response_max_length)``
            respectively.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None] - 1

        num_cross_att_tokens = prefix_embs.shape[1]

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        # initialize memory tokens to empty tensor for storing memory tokens during inference
        memory_tokens = torch.empty((bsize, 0), device=device, dtype=torch.long)
        # if memory prediction is enabled, then predict memory tokens autoregressively
        for auto_step in range(self.config.memory_max_length):
            (
                prefix_out,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                prefix_offsets,
                memory_tokens,
                past_key_values,
            ) = self.infer_autoregressive(
                prefix_out=prefix_out,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                past_key_values=past_key_values,
                prefix_offsets=prefix_offsets,
                tokens=memory_tokens,
                auto_step=auto_step,
                bsize=bsize,
                device=device,
            )

        # Match training `embed_prefix`: "Subtask: " must be in the KV cache before subtask
        # autoregression (inference does not call `embed_prefix` with `response_tokens`).
        response_start_indicator_ids = self.language_tokenizer.encode("Subtask: ", add_special_tokens=False)
        for i, tid in enumerate(response_start_indicator_ids):
            token = torch.full((bsize, 1), int(tid), device=device, dtype=torch.long)
            emb = self.paligemma_with_expert.embed_language_tokens(token)
            emb = emb * math.sqrt(emb.shape[-1])
            pad_row = torch.ones((bsize, 1), device=device, dtype=prefix_pad_masks.dtype)
            if prefix_att_masks.dtype == torch.bool:
                new_att = torch.full((bsize, 1), i == 0, device=device, dtype=torch.bool)
            else:
                new_att = torch.full(
                    (bsize, 1),
                    1.0 if i == 0 else 0.0,
                    device=device,
                    dtype=prefix_att_masks.dtype,
                )
            prefix_embs = torch.cat([prefix_embs, emb], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, pad_row], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, new_att], dim=1)
            num_cross = prefix_pad_masks.shape[1]
            att_2d_masks = make_att_2d_masks(
                pad_row,
                new_att,
                n_cross_att_tokens=num_cross - 1,
                cross_att_pad_masks=prefix_pad_masks[:, : num_cross - 1],
            )
            prefix_offsets = prefix_offsets + pad_row.long()
            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=prefix_offsets,
                past_key_values=past_key_values,
                inputs_embeds=[emb, None],
                n_cross_att_tokens=num_cross,
                use_cache=True,
                fill_kv_cache=True,
            )

        # initialize response tokens to empty tensor for storing response tokens during inference
        response_tokens = torch.empty((bsize, 0), device=device, dtype=torch.long)
        # if response prediction is enabled, then predict response tokens autoregressively
        for auto_step in range(self.config.response_max_length):
            (
                prefix_out,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                prefix_offsets,
                response_tokens,
                past_key_values,
            ) = self.infer_autoregressive(
                prefix_out=prefix_out,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                past_key_values=past_key_values,
                prefix_offsets=prefix_offsets,
                tokens=response_tokens,
                auto_step=auto_step,
                bsize=bsize,
                device=device,
            )

        return memory_tokens, response_tokens

    def infer_autoregressive(
        self,
        prefix_out: Tensor,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        prefix_offsets: Tensor,
        tokens: Tensor,
        auto_step: int,
        bsize: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
        """Performs one autoregressive generation step.

        At ``auto_step == 0`` a ``<bos>`` token seeds the generation; on
        subsequent steps the most-recent logits are argmax-decoded into the
        next token. Once an ``<eos>`` or ``<pad>`` token appears in the
        accumulated sequence the remaining positions are filled with padding.

        The method updates the KV-cache, prefix embeddings, and masks so that
        the next call can attend to all previously generated tokens.

        Args:
            prefix_out: Transformer output from the previous step
                ``(B, 1, D)`` or ``(B, seq, D)`` on the first call.
            prefix_embs: Running concatenation of all embeddings fed to the
                transformer so far ``(B, current_seq, D)``.
            prefix_pad_masks: Boolean padding mask ``(B, current_seq)``.
            prefix_att_masks: 1-D attention pattern ``(B, current_seq)``.
            past_key_values: KV-cache list from previous transformer calls.
            prefix_offsets: Position ID offsets ``(B, 1)`` tracking the
                current absolute position for each batch element.
            tokens: Accumulated generated token IDs ``(B, steps_so_far)``.
            auto_step: Current step index (0-based).
            bsize: Batch size.
            device: Torch device for tensor creation.

        Returns:
            A tuple of updated state tensors for the next step:
            ``(prefix_out, prefix_embs, prefix_pad_masks, prefix_att_masks,
            prefix_offsets, tokens, past_key_values)``.
        """
        EOS_TOKEN = self.language_tokenizer.convert_tokens_to_ids(self.language_tokenizer.eos_token)  # noqa: N806
        if auto_step == 0:
            # Start the autoregressive inference with <bos> token
            token = torch.full(
                (bsize, 1),
                self.language_tokenizer.bos_token_id,
                device=device,
                dtype=torch.long,
            )
        else:
            # get the last predicted token from the prefix output which is predicted response
            token = prefix_out[:, -1:]
            token = self.paligemma_with_expert.paligemma.lm_head(token).argmax(dim=-1)

        PAD_TOKEN = self.language_tokenizer.pad_token_id  # noqa: N806
        # Create pad masks: False if previous token was EOS or PAD
        if tokens.shape[1] > 1:
            prev_tokens = tokens
            has_eos = (prev_tokens == EOS_TOKEN).any(dim=1, keepdim=True)
            has_pad = (prev_tokens == PAD_TOKEN).any(dim=1, keepdim=True)
            # check if the previous token was EOS or PAD. If so, then the current token should be padded, so its not attended by flow matching action expert.
            pad_masks = ~(has_eos | has_pad)
            token = torch.where(
                pad_masks,
                token,
                torch.tensor(PAD_TOKEN, device=device, dtype=token.dtype),
            )
        else:
            pad_masks = torch.ones((bsize, 1), device=device, dtype=torch.bool)

        # Updating response tokens with current predicted token
        tokens = torch.cat([tokens, token], dim=1)

        # Embed the current predicted token
        emb = self.paligemma_with_expert.embed_language_tokens(token)

        # Normalize response language embeddings
        emb_dim = emb.shape[-1]
        emb = emb * math.sqrt(emb_dim)

        att_masks = torch.ones((bsize, 1), device=device, dtype=emb.dtype)

        # update the prefix embs, pad masks and att masks, so it can be used by action experts
        prefix_embs = torch.cat([prefix_embs, emb], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, pad_masks], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, att_masks], dim=1)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        # create the attention mask for the response tokens
        att_2d_masks = make_att_2d_masks(
            pad_masks,
            att_masks,
            n_cross_att_tokens=num_cross_att_tokens - 1,
            cross_att_pad_masks=prefix_pad_masks[:, : num_cross_att_tokens - 1],
        )
        prefix_offsets = prefix_offsets + pad_masks.long()
        prefix_position_ids = prefix_offsets

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[emb, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=True,
            fill_kv_cache=True,
        )

        return (
            prefix_out,
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_offsets,
            tokens,
            past_key_values,
        )
