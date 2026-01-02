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

import warnings

from accelerate import Accelerator

_acc: Accelerator | None = None


def set_proc_accelerator(accelerator: Accelerator, allow_reset=False) -> None:
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
    return _acc


def acc_print(*args, **kwargs):
    acc = get_proc_accelerator()
    if acc is None:
        print(*args, **kwargs)
    else:
        print(f"Acc[{acc.process_index} of {acc.num_processes}]", *args, **kwargs)
