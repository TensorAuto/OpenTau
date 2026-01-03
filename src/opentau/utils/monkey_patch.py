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

r"""Monkey patches to inject behaviour into computational graph construction. This eliminates the need to trace and
modify source code of 3rd party libraries, such as transformers or even PyTorch itself.

Where necessary, we can do

>>> from opentau.utils.monkey_patch import torch_cumsum_patch, torch_pow_patch
>>> torch_cumsum_patch()  # Apply the patch to handle bool tensors in cumsum
>>> torch_pow_patch()  # Apply the patch to handle mixed number-tensor exponentiation in pow

to apply the patches.

Running the same monkey patch twice will not have any effect, as the 2nd call will return immediately.

Note: currently, there is no way to undo a monkey patch once it has been applied, which can be a future to-do.
For now, only apply the patch when its implications are understood.
"""

import importlib
import logging
import sys
from functools import wraps

import numpy as np
import torch

# global singleton to track which patches have been applied
__patches_applied = set()


def _run_once_only(func):
    """Decorator that ensures the function is run once only.

    Subsequent calls to the function will return immediately without executing
    the function body again.

    Args:
        func: Function to be wrapped.

    Returns:
        Wrapped function that executes only once.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        if func in __patches_applied:
            return

        __patches_applied.add(func)
        logging.debug(f"Applying monkey patch: {func.__name__} in {func.__module__}")
        return func(*args, **kwargs)

    return inner


def patches_applied():
    """Get a list of all patches that have been applied.

    Returns:
        List of patch function names that have been applied.
    """
    return [func.__name__ for func in __patches_applied]


@_run_once_only
def torch_cumsum_patch():
    """Override torch.cumsum to handle bool tensors correctly.

    PyTorch allows cumsum on bool tensors, but ONNX Runtime does not.
    This patch converts bool tensors to int64 before calling cumsum.
    """
    original_cumsum = torch.cumsum

    def _patched_cumsum(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.int64)
        return original_cumsum(tensor, *args, **kwargs)

    torch.cumsum = _patched_cumsum


@_run_once_only
def torch_pow_patch():
    """Override torch.pow to ensure both base and exponent are tensors.

    This patch converts scalar arguments to tensors before calling pow,
    ensuring compatibility with ONNX export.
    """
    original_pow = torch.pow

    def _patched_pow(base, exponent, *args, **kwargs):
        if not isinstance(base, torch.Tensor):
            base = torch.tensor(base, dtype=exponent.dtype, device=exponent.device)
        if not isinstance(exponent, torch.Tensor):
            exponent = torch.tensor(exponent, dtype=base.dtype, device=base.device)
        return original_pow(base, exponent, *args, **kwargs)

    torch.pow = _patched_pow
    torch.Tensor.pow = _patched_pow
    torch.Tensor.__pow__ = _patched_pow
    # At least for torch 2.7, `__rpow__` is already defined like this, but we ensure it for future compatibility
    torch.Tensor.__rpow__ = lambda x, y: _patched_pow(y, x)


@_run_once_only
def torch_full_patch():
    """Override torch.full to convert bool fill values to int.

    This patch ensures that True/False are converted to 1/0 before reaching
    the C++ level, improving compatibility with certain backends.
    """
    original_full = torch.full

    def _patched_full(size, fill_value, *args, **kwargs):
        if isinstance(fill_value, bool):
            fill_value = int(fill_value)
        return original_full(size, fill_value, *args, **kwargs)

    torch.full = _patched_full


@_run_once_only
def torch_fake_tensor_module_to_patch():
    """Fix torch.nn.Module.to(device) behavior in FakeTensorMode.

    Without this patch, Module.to(device) is a no-op in FakeTensorMode, leading
    to device mismatch errors. This patch enables proper device conversion.

    See https://github.com/pytorch/pytorch/issues/119665 for more details.
    """
    torch.__future__.set_overwrite_module_params_on_conversion(True)


@_run_once_only
def torch_fake_tensor_to_numpy_patch():
    """Enable .numpy() calls on FakeTensor to return random numpy arrays.

    This patch allows .numpy() to be called on FakeTensor instances, returning
    numpy arrays with random values. Note that calling .numpy() multiple times
    on the same FakeTensor may return different values.
    """
    _torch2np = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        # torch.bfloat16 is intentionally excluded as it is not supported by numpy
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }

    def _patched_numpy(self: torch._subclasses.fake_tensor.FakeTensor, /):
        if self.device.type != "cpu":
            raise RuntimeError(
                f"FakeTensor.numpy() can only be called on CPU tensors. This tensor is on {self.device}"
            )
        if self.requires_grad:
            raise RuntimeError(
                ".numpy() cannot be called on tensors that require gradients. Call tensor.detach().numpy() instead."
            )
        if self.dtype not in _torch2np:
            raise RuntimeError(f"Unsupported dtype {self.dtype} for FakeTensor.numpy()")

        # `np.random.rand()` returns a float instead of a nil-dim array
        # So we wrap it in np.array() to ensure the shape is preserved
        return np.array(np.random.rand(*self.shape)).astype(_torch2np[self.dtype])

    torch._subclasses.fake_tensor.FakeTensor.numpy = _patched_numpy


@_run_once_only
def torch_fake_tensor_beta_validate_args_patch():
    """Fix torch.distributions.Beta to work in FakeTensorMode.

    This patch sets validate_args=False by default for Beta distributions,
    which is required for FakeTensorMode compatibility.
    """
    original_beta_init = torch.distributions.Beta.__init__

    def _patched_beta_init(self, *args, **kwargs):
        kwargs.setdefault("validate_args", False)
        original_beta_init(self, *args, **kwargs)

    torch.distributions.Beta.__init__ = _patched_beta_init


@_run_once_only
def torch_fake_tensor_is_inf_patch():
    """Patch torch.isinf to work with FakeTensor.

    This patch provides a mock implementation of torch.isinf that returns
    a mock object compatible with FakeTensor operations.
    """
    from unittest.mock import Mock

    def _patched_isinf(x):
        obj = Mock()
        obj.dtype = torch.bool
        obj.shape = x.shape
        obj.any.return_value = False
        obj.all.return_value = False
        return obj

    torch.isinf = _patched_isinf


@_run_once_only
def gym_is_gymnasium_patch():
    """Monkey patch to make `import gym` equivalent to `import gymnasium as gym`.

    This patch is necessary because the original gym package is incompatible
    with numpy >= 2.0. It redirects gym imports to use gymnasium instead.
    """
    _g = importlib.import_module("gymnasium")
    sys.modules.setdefault("gym", _g)

    # This is a non-exhaustive list. More submodules may be added in the future as needed.
    # A more compressive solution would involve a lower-level approach using `finder`s and `loader`s.
    # See https://docs.python.org/3/reference/import.html#finders-and-loaders
    subpackages = [
        "spaces",
        "envs",
        "envs.classic_control",
        "envs.mujoco",
        "envs.toy_text",
        "wrappers",
        "vector",
        "vector.utils",
        "utils",
    ]

    for sub in subpackages:
        try:
            old_name = f"gym.{sub}"
            new_name = f"gymnasium.{sub}"
            if old_name in sys.modules:
                print(f"Module {old_name} already exists in sys.modules, skipping import of {new_name}")
            else:
                # Assuming importing the submodule has no side effects, which should be true for gymnasium
                sys.modules[old_name] = importlib.import_module(new_name)
        except (ImportError, ModuleNotFoundError):
            print("Failed to import gymnasium submodule:", sub, file=sys.stderr)


@_run_once_only
def torch_load_patch():
    """Override torch.load to handle weights_only argument.

    This patch ensures that torch.load properly handles the weights_only
    argument for PyTorch versions >= 2.6, setting it to False by default
    if not explicitly provided.
    """
    if torch.__version__ < "2.6":
        return

    original_load = torch.load

    def _patched_load(*args, weights_only=..., **kwargs):
        kwargs["weights_only"] = False if weights_only is ... else weights_only
        return original_load(*args, **kwargs)

    torch.load = _patched_load
