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

from dataclasses import dataclass

from utils import grep_file

from opentau.configs.parser import wrap


@dataclass
class Arg:
    log_path: str
    expected_length: int
    re_pattern: str = r"accelerator\.sync_gradients=(True|False)"
    gradient_accumulation_steps: int = 2


@wrap()
def main(arg: Arg) -> None:
    sync_grads = grep_file(arg.log_path, arg.re_pattern, processor=bool)
    assert len(sync_grads) == arg.expected_length, (
        f"Expected {arg.expected_length} sync_gradients, found {len(sync_grads)} in {arg.log_path}."
    )
    assert all(sg == ((i + 1) % arg.gradient_accumulation_steps == 0) for i, sg in enumerate(sync_grads)), (
        f"Sync gradients should be set according to "
        f"gradient_accumulation_steps={arg.gradient_accumulation_steps}, "
        f"got {sync_grads}."
    )
