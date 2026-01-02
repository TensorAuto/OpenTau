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

import base64
import io

import torch
import yaml
from PIL import Image


def tensor_to_base64(image_dict: dict[str, torch.Tensor], task: str) -> list[bytes]:
    """
    converts dictionary of tensors into list of base64

    Args :
        image_dict (dict[str, torch.Tensor]) : Dictionary of tensors (camera images)
        task : two supported tasks, i.e , 'mani' and 'nav'

    Return:
        images (list[bytes]) : List of base64
    """

    images = []

    for img in image_dict.values():
        if task == "mani":
            img_tensor = img.squeeze(0)
            img_tensor = img_tensor.to(dtype=torch.float32, device="cpu")
            img_tensor = img_tensor.clamp(0, 1) * 255.0
            img = Image.fromarray(img_tensor.to(torch.uint8).permute(1, 2, 0).numpy())

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return images


def load_prompt_library(filepath: str) -> dict:
    """
    Loads a YAML file and returns its content as a dictionary.
    """
    try:
        with open(filepath) as file:
            # Use yaml.safe_load() to parse the YAML file
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")
        return None
