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

from abc import ABC, abstractmethod
from typing import Optional

import torch
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    LlamaTokenizer,
)

from opentau.planner.utils.memory import Memory
from opentau.planner.utils.utils import load_prompt_library, tensor_to_base64


class BaseHighLevelPlanner(ABC):
    """
    Represents a High level planner which has ability to infer various open source models and closed models like gpt-4o.
    Generates a list of low level plans given a high level plan
    """

    def __init__(self):
        self.prompts_dict = load_prompt_library("src/opentau/prompts/planner/prompts.yaml")

    @abstractmethod
    def inference(
        self, image_dict: dict[str, torch.Tensor], model_name: str, task: str, mem: Optional[Memory] = None
    ) -> str:
        """
        Handles inferencing the planner given images and language inputs
        """

        pass

    def calculate_usage(self, response) -> float:
        """
        Calculates cost for each call to gpt-4o

        Args:

            response : a response object from gpt chat compeletion method

        Returns:

            cost (float) : cost for one call
        """

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        cost = (prompt_tokens / 1000) * 0.0025 + (completion_tokens / 1000) * 0.01

        return cost


class HighLevelPlanner(BaseHighLevelPlanner):
    """
    Represents a High level planner which has ability to infer various open source models and closed models like gpt-4o.
    Generates a list of low level plans given a high level plan
    """

    def __init__(self):
        super().__init__()

    def model_and_tokenizer(self, model_name: str, device: str):
        if model_name == "cogvlm-chat-hf":
            processor = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
            model = AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm-chat-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        elif model_name == "SmolVLM-256M-Instruct":
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float16
            ).to(device)
        elif model_name == "SmolVLM-500M-Instruct":
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct", torch_dtype=torch.float16
            ).to(device)
        elif model_name == "SmolVLM2-2.2B-Instruct":
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
            model = AutoModelForImageTextToText.from_pretrained(
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct", torch_dtype=torch.float32
            ).to(device)
        else:
            raise RuntimeError(f"The specified model  {model_name} is not supported")

        return model, processor

    def generate_prompt(self, task: str, mem: Memory | None) -> str:
        """
        Generates prompt for gpt-4o model based on memory.

        Args:

            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.

        Returns:

            prompt (str): a prompt
        """

        if mem is None:
            prompt = (
                f"Look at the image <image>. The task is {task}."
                + "\n"
                + self.prompts_dict["prompts"]["robot_user_without_memory_manipulation"]["template"]
            )

        else:
            prompt = (
                f"Look at the image <image>. The task is {task}."
                + "\n"
                + self.prompts_dict["prompts"]["robot_user_with_memory_manipulation"]["template"]
            )

        return prompt

    def gpt_inference(self, image_dict: dict[str, torch.Tensor], task: str, mem: Memory | None) -> str:
        """
        Calls openai Api and passes high level plan and memory

        Args:

            image_dict(dict[str , torch.Tensor]) : dict of tensors of images in base64 format
            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.

        Returns:

            response (str): a low level language command that can be understood by low level planner
        """

        images = tensor_to_base64(image_dict, "mani")

        client = OpenAI()

        prompt = self.generate_prompt(task, mem)

        content = [
            {"type": "text", "text": f"{prompt}"},
        ]

        for image_base64 in images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            )

        if mem is None:
            message = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.prompts_dict['prompts']['robot_system_manipulation']['template']}",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            mem.add_conversation("user", content)
            message = mem.get_conversation()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            max_tokens=500,
            temperature=0.0,
            tool_choice=None,
        )

        res = response.choices[0].message.content

        return res

    def opensource_inference(self, image_path, task, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, processor = self.model_and_tokenizer(model_name, device)

        prompt = self.generate_prompt(task)

        image1 = Image.open(image_path).convert("RGB")

        inputs = processor(text=prompt, images=[image1], return_tensors="pt")
        inputs = inputs.to(device)

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]

    def inference(
        self, image_dict: dict[str, torch.Tensor], model_name: str, task: str, mem: Optional[Memory] = None
    ) -> str:
        """
        Handles calling of open source models and gpt-4o models

        Args:

            image_dict (dict[str , torch.Tensor]) : dict of tensors of images in base64 format
            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.
            model_name (str) : Name of open source model to be inferenced. To use gpt models, pass a gpt4o
            image_path (str) : Path to image file for opensource models.

        Returns:

            response (str): a low level language command that can be understood by low level planner
        """

        if model_name == "gpt4o":
            actions = self.gpt_inference(image_dict, task, mem)
            actions = actions.split("```json")[1].split("[")[1].split("]")[0]
        else:
            actions = self.opensource_inference(model_name, task)

        return actions


class NavHighLevelPlanner(BaseHighLevelPlanner):
    """
    Represents a High level planner which has ability to infer various open source models and closed models like gpt-4o.
    Generates a list of low level plans given a high level plan
    """

    def __init__(self):
        super().__init__()

    def generate_prompt(self, task: str, mem: Memory | None) -> str:
        """
        Generates prompt for gpt-4o model based on memory.

        Args:

            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.

        Returns:

            prompt (str): a prompt
        """

        prompt = (
            self.prompts_dict["prompts"]["robot_user_navigation"]["template"]
            + f"Look at the given images {' '.join(['<image>'] * 21)} from starting point. The task is {task}."
        )

        return prompt

    def gpt_inference(self, image_dict: dict[str, torch.Tensor], task: str, mem: Memory | None) -> str:
        """
        Calls openai Api and passes high level plan and memory

        Args:

            image_dict(dict[str , torch.Tensor]) : dict of tensors of images in base64 format
            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.

        Returns:

            response (str): a low level language command that can be understood by low level planner
        """

        images = tensor_to_base64(image_dict, "nav")

        client = OpenAI()

        prompt = self.generate_prompt(task, mem)

        content = [
            {"type": "text", "text": f"{prompt}"},
        ]

        for image_base64 in images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            )

        if mem is None:
            message = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.prompts_dict['prompts']['robot_system_navigation']['template']}",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            mem.add_conversation("user", content)
            message = mem.get_conversation()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            max_tokens=500,
            temperature=0.0,
            tool_choice=None,
        )

        res = response.choices[0].message.content

        return res

    def inference(
        self, image_dict: dict[str, torch.Tensor], model_name: str, task: str, mem: Optional[Memory] = None
    ) -> str:
        """
        Handles calling of open source models and gpt-4o models

        Args:

            image_dict (dict[str , torch.Tensor]) : dict of tensors of images in base64 format
            task (str): a high level command as a language
            mem (Memory|None): instance of Memory class from utils file, which stores all the previous conservations of relevant task in hand. Its set to None if memory is no required while inferencing.
            model_name (str) : Name of open source model to be inferenced. To use gpt models, pass a gpt4o
            image_path (str) : Path to image file for opensource models.

        Returns:

            response (str): a low level language command that can be understood by low level planner
        """

        if model_name == "gpt4o":
            actions = self.gpt_inference(image_dict, task, mem)

        return actions
