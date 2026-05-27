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

import numpy as np
import packaging
import safetensors
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_file as save_safetensor_file
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.utils import log_model_loading_keys
from opentau.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")

# State-dict key prefixes for the per-feature normalization buffers attached
# by `Normalize` / `Unnormalize`. Used by `_save_pretrained` (when
# `save_normalization_stats=False`) and by `train.py`'s Accelerate save-hook
# to strip the buffers from the on-disk safetensors. Kept here (not in
# normalize.py) so the train script can import it without picking up torch's
# heavy normalize module unless it actually constructs a policy.
NORM_BUFFER_PREFIXES: tuple[str, ...] = (
    "normalize_inputs.buffer_",
    "normalize_targets.buffer_",
    "normalize_discrete_actions.buffer_",
    "unnormalize_outputs.buffer_",
)


def is_norm_buffer_key(key: str) -> bool:
    """Return True iff ``key`` names a Normalize/Unnormalize stat parameter."""
    return any(key.startswith(prefix) for prefix in NORM_BUFFER_PREFIXES)


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
        # Build the name -> index lookup used by `_resolve_dataset_index` to
        # map inference-time `batch["dataset_repo_id"]` (str) to the leading
        # axis of the stacked Normalize buffers. Stays None for legacy
        # checkpoints whose config.json predates the per-dataset rewrite —
        # those callers must pass `batch["dataset_index"]` directly.
        names = getattr(config, "dataset_names", None)
        if names is None:
            self._dataset_name_to_index: dict[str, int] | None = None
        else:
            self._dataset_name_to_index = {name: i for i, name in enumerate(names)}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(
        self,
        save_directory: Path,
        *,
        include_norm_stats: bool | None = None,
    ) -> None:
        """Saves the policy and its configuration to a directory.

        Args:
            save_directory: The directory to save the policy to.
            include_norm_stats: When ``None`` (default), defers to
                ``self.config.save_normalization_stats``. When set, overrides
                the config field for this call. When effectively False the
                ``normalize_*.buffer_*`` / ``unnormalize_*.buffer_*`` keys are
                excluded from the on-disk ``model.safetensors`` — reloading
                then requires the caller to pass ``ds_meta=`` (or call
                ``policy._inject_stats``) so the buffers can be repopulated.
        """
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self

        if include_norm_stats is None:
            include_norm_stats = getattr(self.config, "save_normalization_stats", True)

        out_path = str(save_directory / SAFETENSORS_SINGLE_FILE)
        if include_norm_stats:
            save_model_as_safetensor(model_to_save, out_path)
        else:
            # `save_model` doesn't expose a filter argument, so build the
            # filtered state_dict ourselves and write via `save_file`. Clone
            # tensors that share storage with another retained tensor to
            # avoid safetensors' "duplicated tensors" rejection.
            state_dict = model_to_save.state_dict()
            filtered = {k: v for k, v in state_dict.items() if not is_norm_buffer_key(k)}
            save_safetensor_file(filtered, out_path)

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

    def _infer_batch_size_and_device(self, batch: dict[str, Tensor]) -> tuple[int, torch.device]:
        """Find any tensor in ``batch`` and return (batch_size, device).

        Used by ``_resolve_dataset_index`` when the caller supplied
        ``dataset_repo_id`` as a Python string/list — we need to allocate the
        index tensor on the same device as the rest of the batch.
        """
        for v in batch.values():
            if isinstance(v, Tensor):
                return v.shape[0] if v.ndim > 0 else 1, v.device
        # No tensors in the batch — fall back to CPU and trust caller's len(list).
        return 0, torch.device("cpu")

    def _resolve_dataset_index(self, batch: dict[str, Tensor]) -> Tensor:
        """Resolve per-sample dataset indices from a training or inference batch.

        Training path: the dataloader's ``_TaggedDataset`` wrapper attaches
        ``batch["dataset_index"]`` as a ``(B,)`` ``LongTensor`` already.

        Inference path: the caller passes ``batch["dataset_repo_id"]`` as
        either a single ``str`` (broadcast to the full batch) or a
        ``list[str]`` of length B. We map each name through the policy's
        training-time ``dataset_names`` list (stored on ``self.config``) to
        the corresponding integer index.

        Raises:
            KeyError: neither ``dataset_index`` nor ``dataset_repo_id`` was provided.
            RuntimeError: ``dataset_repo_id`` was provided but the policy was
                loaded without a ``dataset_names`` list.
            ValueError: a name was provided that isn't in ``dataset_names``.
        """
        if "dataset_index" in batch:
            idx = batch["dataset_index"]
            if not isinstance(idx, Tensor):
                idx = torch.as_tensor(idx, dtype=torch.long)
            return idx.to(dtype=torch.long)

        if "dataset_repo_id" not in batch:
            raise KeyError(
                "Per-dataset normalization requires either `dataset_index` "
                "(LongTensor of shape (B,)) or `dataset_repo_id` (str or "
                "list[str] of length B) in the batch."
            )

        if self._dataset_name_to_index is None:
            raise RuntimeError(
                "Policy was loaded without `dataset_names`; cannot resolve "
                "`dataset_repo_id` strings. Either pass `batch['dataset_index']` "
                "directly or rebuild the policy via `make_policy(cfg, ds_meta=...)`."
            )

        raw = batch["dataset_repo_id"]
        batch_size, device = self._infer_batch_size_and_device(batch)
        names = [raw] * (batch_size or 1) if isinstance(raw, str) else list(raw)
        try:
            indices = [self._dataset_name_to_index[n] for n in names]
        except KeyError as e:
            raise ValueError(
                f"dataset_repo_id {e.args[0]!r} not in this policy's training "
                f"set {list(self._dataset_name_to_index)}"
            ) from None
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _inject_stats(
        self,
        per_dataset_stats: list[dict[str, dict[str, Tensor | np.ndarray]]],
        dataset_names: list[str] | None = None,
    ) -> None:
        """Overwrite the Normalize/Unnormalize buffers in-place from ``per_dataset_stats``.

        Used to repopulate stats after loading a checkpoint that was saved
        with ``save_normalization_stats=False``. The buffer shapes and dtypes
        must already match (i.e. the policy must have been constructed with
        the same number of datasets and the same feature set as the stats
        being injected).

        Args:
            per_dataset_stats: Ordered list of per-dataset stat dicts (same
                shape ``Normalize`` accepts at construction).
            dataset_names: Optional ordered list of names; when provided also
                updates ``self.config.dataset_names`` and the cached
                ``_dataset_name_to_index`` lookup so inference-time
                ``dataset_repo_id`` resolution works after the injection.
        """
        from opentau.policies.normalize import _stat_to_float32_tensor

        for module_attr in (
            "normalize_inputs",
            "normalize_targets",
            "normalize_discrete_actions",
            "unnormalize_outputs",
        ):
            module = getattr(self, module_attr, None)
            if module is None:
                continue
            for feature_key, ft in module.features.items():
                norm_mode = module.norm_map.get(ft.type)
                if norm_mode is None or norm_mode.name == "IDENTITY":
                    continue
                buffer_attr = "buffer_" + feature_key.replace(".", "_")
                buffer = getattr(module, buffer_attr, None)
                if buffer is None:
                    continue
                stat_names = ("mean", "std") if norm_mode.name == "MEAN_STD" else ("min", "max")
                for stat in stat_names:
                    rows = [_stat_to_float32_tensor(s[feature_key][stat]) for s in per_dataset_stats]
                    new_tensor = torch.stack(rows, dim=0).to(
                        device=buffer[stat].device, dtype=buffer[stat].dtype
                    )
                    if new_tensor.shape != buffer[stat].shape:
                        raise ValueError(
                            f"Injected stats shape {tuple(new_tensor.shape)} does not match "
                            f"existing buffer shape {tuple(buffer[stat].shape)} for "
                            f"{module_attr}.{buffer_attr}['{stat}']."
                        )
                    with torch.no_grad():
                        buffer[stat].data.copy_(new_tensor)
            # Refresh the module's own dataset_names cache.
            if dataset_names is not None:
                module.dataset_names = list(dataset_names)

        if dataset_names is not None:
            self.config.dataset_names = list(dataset_names)
            self._dataset_name_to_index = {name: i for i, name in enumerate(dataset_names)}

    def _check_norm_stats_loaded(self) -> None:
        """Raise a clear error if any Normalize/Unnormalize buffer is still ∞.

        Called by ``make_policy`` after a checkpoint load when no
        ``per_dataset_stats`` was supplied — surfaces the
        ``save_normalization_stats=False`` round-trip mistake at
        policy-construction time rather than at the first forward.
        """
        bad: list[str] = []
        for module_attr in (
            "normalize_inputs",
            "normalize_targets",
            "normalize_discrete_actions",
            "unnormalize_outputs",
        ):
            module = getattr(self, module_attr, None)
            if module is None:
                continue
            for name, param in module.named_parameters(recurse=True):
                if name.startswith("buffer_") and torch.isinf(param).any():
                    bad.append(f"{module_attr}.{name}")
        if bad:
            raise RuntimeError(
                "Normalization buffers were not initialised from a checkpoint "
                "and no per_dataset_stats / ds_meta was passed. The following "
                f"buffers are still +inf: {bad}. Either re-save the checkpoint "
                "with `save_normalization_stats=True`, or pass `ds_meta=` to "
                "`make_policy(...)` so stats are injected after load."
            )

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
