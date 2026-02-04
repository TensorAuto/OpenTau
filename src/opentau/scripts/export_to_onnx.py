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

"""ONNX export script for PI0/PI05 policies.

This script exports the VLA policy to ONNX format. Due to the complexity of the
PI05 model (text tokenization, autoregressive generation, variable-length loops),
we export only the core tensor operations with pre-computed tokens.

The ONNX model accepts:
- Pre-tokenized language tokens (computed externally)
- Image tensors (already preprocessed)
- Optional noise tensor

This allows the ONNX model to focus on the traceable neural network operations
while non-traceable operations (tokenization, state discretization) are handled
externally in Python.

For large models (>2GB), the script automatically uses ONNX external data format
to store weights in a separate file, bypassing the protobuf 2GB limit.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.utils.monkey_patch import (
    torch_cumsum_patch,
    torch_full_patch,
    torch_pow_patch,
)
from opentau.utils.utils import auto_torch_device, init_logging

# Some patches are necessary only for dynamo export, which has current upstream bugs.
# Nonetheless, we apply them here to ensure future compatibility.
patches = [
    torch_cumsum_patch,  # This is always necessary to load the ONNX artifact without error.
    torch_full_patch,
    torch_pow_patch,
]


class PI05OnnxWrapper(torch.nn.Module):
    """ONNX-exportable wrapper for PI05 model.

    This wrapper takes pre-tokenized inputs and performs only the traceable
    tensor operations. Non-traceable operations like tokenization and
    state discretization must be done externally.

    The wrapper:
    1. Takes pre-tokenized language tokens (computed externally)
    2. Processes images using PyTorch operations
    3. Runs the flow matching denoising with a fixed number of steps
    """

    def __init__(self, policy: PI05Policy):
        """Initialize the ONNX wrapper.

        Args:
            policy: The PI05Policy to wrap.
            num_cameras: Number of camera inputs.
        """
        super().__init__()
        self.policy = policy

    def forward(self, lang_tokens: Tensor, lang_masks: Tensor, noise: Tensor, *images: Tensor) -> Tensor:
        """Forward pass for ONNX export.

        Args:
            lang_tokens: Pre-tokenized language tokens of shape (batch, seq_len).
            lang_masks: Language attention masks of shape (batch, seq_len).
            noise: Initial noise tensor of shape (batch, n_action_steps, max_action_dim).
                   This should be sampled from N(0, 1) externally.
            *images: Variable number of image tensors, each of shape (batch, 3, H, W).

        Returns:
            Action tensor of shape (batch, n_action_steps, action_dim).
        """
        print("Starting forward pass of the wrapper...")

        # Process images
        image_batch = {f"camera{i}": img for i, img in enumerate(images)}
        processed_images, img_masks = self.policy.prepare_images(image_batch)

        actions = self.policy.model.sample_actions(
            processed_images,
            img_masks,
            lang_tokens,
            lang_masks,
            noise=noise,
        )

        # Unpad actions
        original_action_dim = self.policy.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.policy.unnormalize_outputs({"actions": actions})["actions"]

        # `policy.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
        # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
        actions = actions.transpose(0, 1)
        print("Finished forward pass of the wrapper")
        return actions


def create_onnx_inputs(policy: PI05Policy, cfg, device, dtype):
    """Create dummy inputs for ONNX export by pre-tokenizing a sample prompt.

    Args:
        policy: The PI05Policy instance (for tokenization).
        cfg: Configuration object.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Tuple of (lang_tokens, lang_masks, noise, images_list, input_names_list).
    """
    # Create a sample prompt and tokenize it
    sample_prompt = "Pick up the object and place it in the target location"
    sample_state_str = " ".join(["128"] * cfg.max_state_dim)  # Middle bin values

    if policy.config.predict_response:
        full_prompt = f"Task: {sample_prompt}<eos>State: {sample_state_str}<eos>Response:"
    else:
        full_prompt = f"Task: {sample_prompt}<eos>State: {sample_state_str}<eos>Actions:"

    tokenized = policy.language_tokenizer(
        [full_prompt],
        padding="max_length",
        padding_side="right",
        max_length=policy.config.prompt_max_length,
        return_tensors="pt",
        truncation=True,
    )

    lang_tokens = tokenized["input_ids"].to(device=device)
    lang_masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)

    # Create dummy noise (sampled from N(0, 1))
    # Shape: (batch_size, n_action_steps, max_action_dim)
    noise_shape = (1, policy.config.n_action_steps, policy.config.max_action_dim)
    noise = torch.randn(noise_shape, dtype=dtype, device=device)

    # Create dummy images
    resolution = cfg.resolution if hasattr(cfg, "resolution") else (224, 224)
    images = []
    for _ in range(cfg.num_cams):
        img = torch.zeros((1, 3, *resolution), dtype=dtype, device=device)
        images.append(img)

    # Build input names: lang_tokens, lang_masks, noise, image0, image1, ...
    input_names = ["lang_tokens", "lang_masks", "noise"] + [f"image{i}" for i in range(len(images))]

    return lang_tokens, lang_masks, noise, images, input_names


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    """Main export function."""
    device = auto_torch_device()
    dtype = torch.float32

    logging.info("Applying monkey patches...")
    for patch in patches:
        patch()

    logging.info("Loading policy...")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device)
    policy.to(dtype=dtype)
    policy.eval()

    if not isinstance(policy, PI05Policy):
        raise ValueError(f"ONNX export currently only supports PI05Policy, got {type(policy)}")

    # Create ONNX-compatible wrapper
    wrapper = PI05OnnxWrapper(policy)
    wrapper.to(device)
    wrapper.eval()
    logging.info("Created ONNX inference wrapper")

    # Create dummy inputs by pre-tokenizing
    lang_tokens, lang_masks, noise, images, input_names = create_onnx_inputs(policy, cfg, device, dtype)
    logging.info(f"Generated example inputs with {len(images)} cameras")
    logging.info(f"Language tokens shape: {lang_tokens.shape}")
    logging.info(f"Noise shape: {noise.shape}")
    logging.info(f"Input names: {input_names}")

    # Build args tuple: (lang_tokens, lang_masks, noise, image0, image1, ...)
    args = (lang_tokens, lang_masks, noise) + tuple(images)

    logging.info("Exporting model to ONNX with Dynamo exporter...")
    output_path = Path(cfg.policy.pretrained_path) / "model.onnx"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # External data file is saved alongside the .onnx file with .onnx_data suffix
    weights_path = output_path.with_suffix(".onnx.data")

    with torch.inference_mode():
        logging.info("Running dynamo ONNX export...")
        torch.onnx.export(
            wrapper,
            args,
            str(output_path),
            input_names=input_names,
            output_names=["actions"],
            dynamo=True,
        )

        logging.info(f"Successfully exported model to '{output_path}'")

    logging.info(
        "\nNote: The exported ONNX model uses external data format.\n"
        "When loading the model, ensure both files are in the same directory:\n"
        f"  - {output_path.name} (model structure)\n"
        f"  - {weights_path.name} (model weights)\n"
    )

    logging.info(
        "The exported ONNX model accepts pre-tokenized inputs.\n"
        "For inference, you need to:\n"
        "1. Tokenize your prompt externally using the same tokenizer\n"
        "2. Preprocess images to [0,1] range with correct resolution\n"
        "3. Run the ONNX model with these inputs"
    )


if __name__ == "__main__":
    init_logging()
    main()
