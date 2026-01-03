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
"""Utilities for managing Accelerate accelerator instances.

This module provides functions for setting and getting a global accelerator
instance, which is useful for accessing accelerator state throughout the codebase.
"""

import warnings

from accelerate import Accelerator

_acc: Accelerator | None = None


def set_proc_accelerator(accelerator: Accelerator, allow_reset: bool = False) -> None:
    """Set the global accelerator instance for the current process.

    Args:
        accelerator: Accelerator instance to set.
        allow_reset: If True, allow resetting an already-set accelerator.
            Defaults to False.

    Raises:
        AssertionError: If accelerator is not an Accelerator instance.
        RuntimeError: If accelerator is already set and allow_reset is False.
    """
    global _acc

    assert isinstance(accelerator, Accelerator), (
        f"Expected an `Accelerator` got {type(accelerator)} with value {accelerator}."
    )
    if _acc is not None:
        if allow_reset:
            warnings.warn(
                "Resetting the accelerator. This could have unintended side effects.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise RuntimeError("Accelerator has already been set.")
    _acc = accelerator


def get_proc_accelerator() -> Accelerator:
    """Get the global accelerator instance for the current process.

    Returns:
        The accelerator instance, or None if not set.
    """
    return _acc


def acc_print(*args, **kwargs) -> None:
    """Print with process index prefix when using accelerate.

    If an accelerator is set, prints with a prefix showing the process index.
    Otherwise, prints normally.

    Args:
        *args: Positional arguments to pass to print.
        **kwargs: Keyword arguments to pass to print.
    """
    acc = get_proc_accelerator()
    if acc is None:
        print(*args, **kwargs)
    else:
        print(f"Acc[{acc.process_index} of {acc.num_processes}]", *args, **kwargs)
