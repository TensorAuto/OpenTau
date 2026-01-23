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
    re_pattern: str = r"grad_norm:([0-9.eE+-]+)"


@wrap()
def main(arg: Arg) -> None:
    grad_norm = grep_file(arg.log_path, arg.re_pattern, processor=float)
    assert len(grad_norm) == arg.expected_length, (
        f"Expected {arg.expected_length} grad_norms, found {len(grad_norm)} in {arg.log_path}."
    )
    assert all(g > 0 for g in grad_norm), f"All grad_norms should be greater than zero, got {grad_norm}."
