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

"""π05: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi05.pdf)
"""

import builtins
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn
from transformers import AutoProcessor, AutoTokenizer

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import NormalizationMode
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from opentau.policies.pretrained import PreTrainedPolicy, T
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

        att_2d_masks = torch.cat((att_2d_masks, cross_att_mask), dim=2)

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


def pad_discrete_tokens(tokens: list[list[int]], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Pads or truncates a list of discrete action token sequences to a fixed length.

    Args:
        tokens: A list of discrete action token sequences (lists of integers).
        max_length: The target length for the discrete action token sequences.

    Returns:
        A tuple containing:
            - discrete_action_tokens: A numpy array of shape (len(tokens), max_length) containing the padded discrete action tokens.
            - discrete_action_masks: A boolean numpy array of shape (len(tokens), max_length) indicating valid discrete action tokens (True) and padding (False).
    """
    discrete_action_tokens = []
    discrete_action_masks = []
    for token in tokens:
        if len(token) > max_length:
            discrete_action_tokens.append(np.array(token[:max_length]))
            discrete_action_masks.append(np.ones(max_length, dtype=bool))
        else:
            discrete_action_masks.append(
                np.concatenate(
                    [np.ones(len(token), dtype=bool), np.zeros(max_length - len(token), dtype=bool)]
                )
            )
            discrete_action_tokens.append(np.pad(token, (0, max_length - len(token)), constant_values=0))
    return np.array(discrete_action_tokens), np.array(discrete_action_masks)


class PI05Policy(PreTrainedPolicy):
    """Wrapper class around PI05FlowMatching model to train and run inference within OpenTau."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the PI05Policy.

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
        self.normalize_actions = Normalize(
            config.output_features, {"ACTION": NormalizationMode.MIN_MAX}, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self.discrete_action_processor = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        # Get vocab size from processor
        discrete_action_vocab_size = getattr(self.discrete_action_processor, "vocab_size", None)
        self.model = PI05FlowMatching(config, discrete_action_vocab_size=discrete_action_vocab_size)

        self.reset()

    def reset(self) -> None:
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

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
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
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
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
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
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 20:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:20]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 20} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 20:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:20]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 20} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
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
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Batch of data containing environment observations.

        Returns:
            The predicted action chunk.

        Raises:
            NotImplementedError: Always, as this method is not implemented for PI05.
        """
        raise NotImplementedError("Currently not implemented for PI05")

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
        if len(self._action_queue) == 0:
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

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
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
        batch["discrete_actions"] = self.normalize_actions(dict(batch))["actions"]
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(
            batch
        )  # in img_masks we have True for real images and False for padded images
        lang_tokens, lang_masks = self.prepare_language(
            batch
        )  # in lang_masks we have True for real tokens and False for padded tokens
        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(
            batch
        )  # in discrete_action_masks we have True for real tokens and False for padded tokens
        actions = batch["actions"]
        actions_is_pad = batch.get(
            "action_is_pad"
        )  # in actions_is_pad we have False for real actions and True for padded actions

        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            actions,
            noise,
            time,
            discrete_actions,
            discrete_action_masks,
        )

        mse_loss = losses["MSE"]
        ce_loss = losses["CE"]
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            mse_loss = mse_loss * in_episode_bound.unsqueeze(-1)

        # Remove padding
        mse_loss = mse_loss[:, :, : self.config.max_action_dim]

        # For backward pass
        loss = mse_loss.mean()

        return {"MSE": loss, "CE": ce_loss}

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
        state_np = state.to(device="cpu", dtype=torch.float32).numpy()
        if np.any(state_np < -1.0) or np.any(state_np > 1.0):
            logging.warning(
                f"State values are not normalized between -1 and 1. Min: {state_np.min()}, Max: {state_np.max()}"
            )
        state_np = np.clip(state_np, -1.0, 1.0)
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return [
            " ".join(map(str, row)) for row in discretized_states
        ]  # TODO: return a tensor instead of a list of strings?

    def prepare_discrete_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Prepares discrete actions for the model by tokenizing and padding them.

        Args:
            batch: Batch of data containing the key "discrete_actions".

        Returns:
            A tuple containing:
                - discrete_action_tokens: A tensor of shape (batch_size, max_length) containing the tokenized actions.
                - discrete_action_masks: A tensor of shape (batch_size, max_length) indicating valid tokens.
        """
        device = batch["discrete_actions"].device
        discrete_actions = batch["discrete_actions"].to(device="cpu", dtype=torch.float32)
        tokens = self.discrete_action_processor.__call__(discrete_actions)
        discrete_action_tokens, discrete_action_masks = pad_discrete_tokens(
            tokens, self.config.discrete_action_max_length
        )
        return torch.from_numpy(discrete_action_tokens).to(device=device, dtype=torch.long), torch.from_numpy(
            discrete_action_masks
        ).to(device=device, dtype=torch.bool)

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
        """Tokenize the text input.

        The state is already expected to be discretized into a space-separated string.

        Args:
            batch: Batch of data containing the key "prompt" and "state".

        Returns:
            A tuple containing:
                - lang_tokens: Tensor of language tokens.
                - lang_masks: Tensor of language attention masks.
        """
        device = batch["state"].device
        tasks = batch["prompt"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        # add state to the prompt
        state = self.prepare_discrete_state(batch)
        prompt = [f"Task: {task}State: {state}\nActions:" for task, state in zip(tasks, state, strict=False)]

        tokenized_prompt = self.language_tokenizer.__call__(
            prompt,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks


class PI05FlowMatching(nn.Module):
    """
    π05: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi05.pdf)

    ┌──────────────────────────────────────────┐
    │                   actions                │
    │                   ▲                      │
    │                  ┌┴─────┐                │
    │      kv cache    │Gemma │                │
    │      ┌──────────►│Expert│                │
    │      │           │      │                │
    │     ┌┴─────────┐ │x 10  │                │
    │     │          │ └▲─────┘                │
    │     │PaliGemma │  │                      │
    │     │          │  noise                  │
    │     └▲──▲──▲──▲                          │
    │      │  │  │  └── discrete actions       │
    │      │  │  └───── robot state            │
    │      │  └──────── language tokens        │
    │      └─────────── image(s)               │
    └──────────────────────────────────────────┘
    """

    def __init__(self, config: PI05Config, discrete_action_vocab_size: int | None = None):
        """Initializes the PI05FlowMatching model.

        Args:
            config: Model configuration.
            discrete_action_vocab_size: Size of the discrete action vocabulary.
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
            discrete_action_vocab_size=discrete_action_vocab_size,
            dropout=self.config.dropout,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self._init_model()

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
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            discrete_actions: Optional discrete action tensor.
            discrete_action_masks: Optional discrete action mask tensor.

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

        if discrete_actions is not None:
            discrete_action_emb = self.paligemma_with_expert.embed_discrete_actions(discrete_actions)
            embs.append(discrete_action_emb.to(dtype=torch.bfloat16))
            pad_masks.append(discrete_action_masks)
            att_masks += [1] * discrete_action_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            noisy_actions: Tensor containing noisy actions.
            timestep: Tensor containing timesteps.

        Returns:
            A tuple containing:
                - embs: Concatenated embeddings tensor.
                - pad_masks: Concatenated padding masks tensor.
                - att_masks: Attention masks tensor.
                - adarms_cond: AdaRMS conditioning tensor.
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = noisy_actions.shape[0]
        dtype = torch.bfloat16
        device = noisy_actions.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )

        # Fuse timestep + action information using an MLP
        noisy_actions = noisy_actions.to(dtype=dtype)
        action_emb = self.action_in_proj(noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = time_emb.to(dtype=dtype)
        adarms_cond = time_mlp_func(time_emb)

        # Add to input tokens
        embs.append(action_emb)

        bsize, action_dim = action_emb.shape[:2]
        action_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute the loss.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            actions: Action tensor.
            noise: Optional noise tensor.
            time: Optional time tensor.
            discrete_actions: Optional discrete action tensor.
            discrete_action_masks: Optional discrete action mask tensor.

        Returns:
            A dictionary containing the loss components ("MSE" and "CE").
        """
        # Run VLM first to get key value cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, discrete_actions, discrete_action_masks
        )

        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1] - self.config.discrete_action_max_length

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=True,
            fill_kv_cache=True,
        )

        # Now run action expert
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        # We should skip the response tokens when numbering the position ids for the action expert
        prefix_offsets = torch.sum(prefix_pad_masks[:, : -self.config.discrete_action_max_length], dim=-1)[
            :, None
        ]  # action expert position ids start after prefix
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # stop gradient to avoid backpropagating from action expert to VLM
        for layer_idx in past_key_values:
            past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].detach()
            past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx]["value_states"].detach()

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # compute mse loss for velocity
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        # compute cross entropy loss for discrete actions
        batch_size, seq_len = discrete_actions.shape
        discrete_action_out = prefix_out[:, -self.config.discrete_action_max_length - 1 : -1]
        logits = self.paligemma_with_expert.da_head(discrete_action_out)

        logits = logits.to(dtype=torch.float32)  # upcast to float32 for loss calculation
        logits = rearrange(logits, "b s d -> (b s) d")
        labels = rearrange(discrete_actions, "b s -> (b s)")
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        ce_loss = rearrange(ce_loss, "(b s) -> b s", b=batch_size, s=seq_len)

        # remove pad tokens
        discrete_action_is_pad = ~discrete_action_masks  # convert into format where value for pad is True
        ce_loss = ce_loss * ~discrete_action_is_pad

        # compute mean
        ce_loss = ce_loss.mean()

        return {"MSE": losses, "CE": ce_loss}

    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token tensor.
            lang_masks: Language mask tensor.
            noise: Optional noise tensor.

        Returns:
            The sampled action tensor.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = prefix_embs.shape[1]

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
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
        prefix_pad_masks: Tensor,
        past_key_values: list[dict[str, Tensor]],
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Apply one denoising step of the noise `x_t` at a given timestep.

        Args:
            prefix_pad_masks: Prefix padding masks.
            past_key_values: Past key values from the VLM.
            x_t: Current noise tensor.
            timestep: Current timestep.

        Returns:
            The predicted velocity tensor (v_t).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        # We should skip the response tokens when numbering the position ids for the action expert
        prefix_offsets = torch.sum(prefix_pad_masks[:, : -self.config.discrete_action_max_length], dim=-1)[
            :, None
        ]  # action expert position ids start after prefix
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t
