#!/usr/bin/env python

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
import os
import platform
import time
from functools import wraps

import numpy as np
import pytest
import torch

from opentau.utils.import_utils import is_package_available
from opentau.utils.utils import auto_torch_device


@pytest.fixture(scope="session")
def device():
    return os.environ.get("OPENTAU_TEST_DEVICE", auto_torch_device())


def require_x86_64_kernel(func):
    """
    Decorator that skips the test if plateform device is not an x86_64 cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if platform.machine() != "x86_64":
            pytest.skip("requires x86_64 plateform")
        return func(*args, **kwargs)

    return wrapper


def require_cpu(func):
    """
    Decorator that skips the test if device is not cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if device != "cpu":
            pytest.skip("requires cpu")
        return func(*args, **kwargs)

    return wrapper


def require_cuda(func):
    """
    Decorator that skips the test if cuda is not available.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("requires cuda")
        return func(*args, **kwargs)

    return wrapper


def cuda_total_vram_gib() -> float:
    """Report the total VRAM of the current CUDA device.

    Returns:
        Total device memory in GiB, or ``0.0`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3


def require_vram_gib(min_gib: float):
    """Build a decorator that skips a test unless the CUDA device is big enough.

    Deliberately a call-time decorator rather than a `pytest.mark.skipif`
    whose condition is evaluated at import: reading device properties
    initializes a CUDA context, and the gating CPU run (`-n auto`) imports
    every test module in every xdist worker. On a GPU box that would spawn one
    context per worker for tests that are deselected anyway.

    Args:
        min_gib: Total VRAM the card must have, in GiB. This is the card size
            required, not the test's peak allocation: the conftest
            `release_cuda_memory_after_gpu_test` fixture keeps peaks from
            accumulating across tests, but the floor still has to cover the
            CUDA context and allocator fragmentation on top of that peak.

    Returns:
        A decorator that skips the wrapped test when the device is smaller.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            available = cuda_total_vram_gib()
            if available < min_gib:
                pytest.skip(f"requires >= {min_gib} GiB VRAM (device has {available:.1f} GiB)")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def to_cuda_bf16(batch: dict, device: str = "cuda") -> dict:
    """Move a test batch to ``device``, casting only its float tensors to bfloat16.

    A blanket ``.to(dtype=torch.bfloat16)`` over the whole batch silently
    rewrites the dtype of everything a real dataloader emits as bool or int:
    ``*_is_pad`` masks become floats and the model fails on `~mask` (bitwise-not
    is integer/bool only), while integer fields get rounded — the dataloader
    emits ``speed``/``quality`` as ``torch.long`` and ``mistake`` as
    ``torch.bool``, so casting them changes what ``prepare_metadata``
    stringifies. Preserving every non-floating dtype keeps the batch shaped like
    the real thing and keeps a future token-id tensor safe from the same
    rounding.

    Args:
        batch: Batch dict; tensor values are moved and cast, everything else
            (prompt strings, metadata lists) passes through untouched.
        device: Target device for the tensor values.

    Returns:
        A new dict with floating-point tensors cast to bfloat16 on ``device``
        and every other tensor moved with its dtype intact.
    """
    return {
        key: value.to(device, non_blocking=True, dtype=torch.bfloat16 if value.is_floating_point() else None)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def require_env(func):
    """
    Decorator that skips the test if the required environment package is not installed.
    As it need 'env_name' in args, it also checks whether it is provided as an argument.
    If 'env_name' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'env_name' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "env_name" in arg_names:
            # Get the index of 'env_name' and retrieve the value from args
            index = arg_names.index("env_name")
            env_name = args[index] if len(args) > index else kwargs.get("env_name")
        else:
            raise ValueError("Function does not have 'env_name' as an argument.")

        # Perform the package check
        package_name = f"gym_{env_name}"
        if env_name is not None and not is_package_available(package_name):
            pytest.skip(f"gym-{env_name} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package_arg(func):
    """
    Decorator that skips the test if the required package is not installed.
    This is similar to `require_env` but more general in that it can check any package (not just environments).
    As it need 'required_packages' in args, it also checks whether it is provided as an argument.
    If 'required_packages' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'required_packages' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "required_packages" in arg_names:
            # Get the index of 'required_packages' and retrieve the value from args
            index = arg_names.index("required_packages")
            required_packages = args[index] if len(args) > index else kwargs.get("required_packages")
        else:
            raise ValueError("Function does not have 'required_packages' as an argument.")

        if required_packages is None:
            return func(*args, **kwargs)

        # Perform the package check
        for package in required_packages:
            if not is_package_available(package):
                pytest.skip(f"{package} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package(package_name):
    """
    Decorator that skips the test if the specified package is not installed.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_package_available(package_name):
                pytest.skip(f"{package_name} not installed")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_hf_flakiness(reruns: int = 2, delay: float = 10.0):
    """Retry a test on transient HF Hub server-side failures.

    Catches `HfHubHTTPError` with 5xx status codes (Hub outage) and
    `FileNotFoundError` whose path lives under the HF cache (downstream effect of
    `snapshot_download` falling back to an incomplete local_dir on Hub 5xx).
    Other exceptions propagate immediately so real test bugs still fail fast.
    """
    from huggingface_hub.errors import HfHubHTTPError

    def _is_hf_flaky(exc: BaseException) -> bool:
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            return status is not None and 500 <= status < 600
        if isinstance(exc, FileNotFoundError):
            path = str(exc.filename or exc)
            return "/.cache/huggingface/" in path or "/huggingface/" in path
        return False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(reruns + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt < reruns and _is_hf_flaky(exc):
                        time.sleep(delay)
                        continue
                    raise

        return wrapper

    return decorator


def generic_equal(obj1, obj2) -> bool:
    r"""Compare two objects for equality, handling torch tensors, numpy arrays, lists, tuples, and dictionaries."""
    if type(obj1) is not type(obj2):
        return False
    if isinstance(obj1, torch.Tensor):
        return torch.allclose(obj1, obj2)
    if isinstance(obj1, np.ndarray):
        return np.allclose(obj1, obj2, equal_nan=False)
    if isinstance(obj1, (list, tuple)):
        return len(obj1) == len(obj2) and all(
            generic_equal(o1, o2) for o1, o2 in zip(obj1, obj2, strict=False)
        )
    if isinstance(obj1, dict):
        return set(obj1) == set(obj2) and all(generic_equal(obj1[k], obj2[k]) for k in obj1)
    return obj1 == obj2
