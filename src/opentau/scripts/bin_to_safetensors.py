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

import argparse
import copy  # Import copy module for deepcopy
import os
import sys

import torch
from safetensors.torch import save_file  # Removed load_file as it's not used here


def convert_bin_to_safetensors(input_path: str, output_path: str, map_location: str = "cpu"):
    """
    Converts a PyTorch .bin checkpoint (state_dict) to .safetensors format.

    This script attempts to handle cases where tensors might share memory by
    creating a deep copy of the state_dict before saving to .safetensors.
    This ensures that each tensor has its own memory, which is a requirement
    for `safetensors.torch.save_file` when shared tensors are detected.

    Args:
        input_path (str): Path to the input .bin file.
        output_path (str): Path where the .safetensors file will be saved.
        map_location (str): Device to map the tensors to when loading the .bin file.
                            Defaults to 'cpu' to avoid GPU memory issues during conversion.
                            Can be 'cuda', 'cuda:0', etc.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if not output_path.lower().endswith(".safetensors"):
        print(
            f"Warning: Output path '{output_path}' does not end with '.safetensors'. Appending it.",
            file=sys.stderr,
        )
        output_path += ".safetensors"

    print(f"Attempting to load state_dict from '{input_path}'...")
    try:
        # Load the state_dict from the .bin file
        state_dict = torch.load(input_path, map_location=torch.device(map_location))  # nosec B614
        print("State_dict loaded successfully.")

        # Handle shared memory tensors for safetensors compatibility
        # Create a deep copy of the state_dict to ensure all tensors have unique memory locations.
        # This resolves the "Some tensors share memory" error from safetensors.
        # Note: This might increase the file size if many tensors were originally shared.
        unique_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                unique_state_dict[key] = value.clone().detach()
            else:
                unique_state_dict[key] = copy.deepcopy(value)  # Handle non-tensor items (e.g., lists, dicts)

        # Save the state_dict to .safetensors format
        print(f"Saving state_dict to '{output_path}' in .safetensors format...")
        save_file(unique_state_dict, output_path)
        print(f"Conversion successful! Output saved to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}", file=sys.stderr)
        # Provide more specific guidance if it's the known safetensors shared memory error
        if "Some tensors share memory" in str(e):
            print(
                "\nThis error typically occurs when the PyTorch state_dict contains tensors that share the same underlying memory (e.g., `lm_head.weight` and `embed_tokens.weight`)."
            )
            print("The script attempted to resolve this by deep-copying the state_dict before saving.")
            print(
                "If the issue persists, ensure your `safetensors` library is up-to-date (`pip install --upgrade safetensors`)."
            )
            print(
                "For models with complex shared weight patterns, manually loading the model architecture and using `safetensors.torch.save_model(model, output_path)` might be necessary, as it handles shared weights more robustly."
            )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .bin checkpoint (state_dict) to .safetensors format."
    )
    parser.add_argument("input_file", type=str, help="Path to the input .bin model weights file.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output .safetensors file. If not provided, "
        "it will be inferred from the input filename (e.g., model.bin -> model.safetensors).",
    )
    parser.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="Device to map the tensors to when loading the .bin file. "
        "Defaults to 'cpu'. Can be 'cuda', 'cuda:0', etc.",
    )

    args = parser.parse_args()

    # If output_file is not provided, infer it
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + ".safetensors"

    convert_bin_to_safetensors(args.input_file, args.output_file, args.map_location)
