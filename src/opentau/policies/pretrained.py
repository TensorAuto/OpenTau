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

"""Base class for pre-trained policies in OpenTau.

This module defines the abstract base class `PreTrainedPolicy` which handles
loading, saving, and basic interface requirements for all policy implementations
in the OpenTau library. It integrates with Hugging Face Hub for model sharing
and safetensors for efficient serialization.
"""

import abc
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

import packaging
import safetensors
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.utils import log_model_loading_keys
from opentau.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")

DEFAULT_POLICY_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This policy has been pushed to the Hub using [OpenTau](https://github.com/TensorAuto/OpenTau):
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """Base class for all policy models in OpenTau.

    This class extends `nn.Module` and `HubMixin` to provide common functionality
    for policy models, including configuration management, model loading/saving,
    and abstract methods that all policies must implement.

    Attributes:
        config: The configuration instance for this policy.
    """

    config_class: None
    """The configuration class associated with this policy. Must be defined in subclasses."""

    name: None
    """The name of the policy. Must be defined in subclasses."""

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        """Initializes the PreTrainedPolicy.

        Args:
            config: The configuration object for the policy.
            *inputs: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If `config` is not an instance of `PreTrainedConfig`.
        """
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        """Saves the policy and its configuration to a directory.

        Args:
            save_directory: The directory to save the policy to.
        """
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: Type[T],
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
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Loads a pretrained policy from a local path or the Hugging Face Hub.

        The policy is set in evaluation mode by default using `policy.eval()`
        (dropout modules are deactivated). To train it, you should first set it
        back in training mode with `policy.train()`.

        Args:
            pretrained_name_or_path: The name or path of the pretrained model.
            config: Optional configuration object. If None, it will be loaded from the
                pretrained model.
            force_download: Whether to force download the model weights.
            resume_download: Whether to resume an interrupted download.
            proxies: Proxy configuration for downloading.
            token: Hugging Face token for authentication.
            cache_dir: Directory to cache downloaded files.
            local_files_only: Whether to only look for local files.
            revision: The specific model version to use (branch, tag, or commit hash).
            strict: Whether to strictly enforce matching keys in state_dict.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            T: An instance of the loaded policy.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
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
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.eval()
        return policy

    def _tile_linear_input_weight(self, state_dict_to_load: dict):
        """Modifies the `state_dict_to_load` in-place by tiling linear layer input weights.

        This ensures compatibility with the model architecture when weight dimensions don't match exactly,
        typically used for expanding input layers.

        Args:
            state_dict_to_load: The state dictionary to modify.
        """
        for name, submodule in self.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue
            weight_name = f"{name}.weight"
            if weight_name not in state_dict_to_load:
                continue
            weight = state_dict_to_load[weight_name]
            assert len(weight.shape) == 2, f"Shape of {weight_name} must be 2D, got {weight.shape}"
            out_dim, in_dim = weight.shape
            assert submodule.out_features == out_dim, (
                f"Output of {name} = {submodule.out_features} does not match loaded weight output dim {out_dim}"
            )
            if submodule.in_features == in_dim:
                continue

            logging.warning(f"Tiling {weight_name} from shape {weight.shape} to {submodule.weight.shape}")
            repeat, remainder = divmod(submodule.in_features, in_dim)
            weight = torch.cat([weight] * repeat + [weight[:, :remainder]], dim=1)
            state_dict_to_load[weight_name] = weight

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        """Loads model weights from a safetensors file.

        Args:
            model: The model instance to load weights into.
            model_file: Path to the safetensors file.
            map_location: Device to map the weights to.
            strict: Whether to enforce strict key matching.

        Returns:
            T: The model with loaded weights.
        """
        # Create base kwargs
        kwargs = {"strict": strict}

        # Add device parameter for newer versions that support it
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        # Load the model with appropriate kwargs
        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        # For older versions, manually move to device if needed
        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    # def generate_model_card(self, *args, **kwargs) -> ModelCard:
    #     card = ModelCard.from_template(
    #         card_data=self._hub_mixin_info.model_card_data,
    #         template_str=self._hub_mixin_info.model_card_template,
    #         repo_url=self._hub_mixin_info.repo_url,
    #         docs_url=self._hub_mixin_info.docs_url,
    #         **kwargs,
    #     )
    #     return card

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """Returns the policy-specific parameters dict to be passed on to the optimizer.

        Returns:
            dict: A dictionary of parameters to optimize.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Resets the policy state.

        This method should be called whenever the environment is reset.
        It handles tasks like clearing caches or resetting internal states for stateful policies.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Performs a forward pass of the policy.

        Args:
            batch: A dictionary of input tensors.

        Returns:
            tuple[Tensor, dict | None]: A tuple containing:
                - The loss tensor.
                - An optional dictionary of metrics or auxiliary outputs.
                  Apart from the loss, items should be logging-friendly native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Selects an action based on the input batch.

        This method handles action selection during inference, including
        caching for stateful policies (e.g. RNNs, Transformers).

        Args:
            batch: A dictionary of observation tensors.

        Returns:
            Tensor: The selected action(s).
        """
        raise NotImplementedError
