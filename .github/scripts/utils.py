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

import re


def grep_file(file: str, pattern: str, processor=None) -> list:
    processor = processor or (lambda x: x)
    values = []
    with open(file) as f:
        for line in f:
            match = re.search(pattern, line)
            if not match:
                continue
            values.append(processor(match.group(1)))
    return values
