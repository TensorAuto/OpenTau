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

import pytest
import torch

from opentau.planner import Memory


@pytest.fixture(scope="session")
def mem(request):
    len = request.param
    if len is None:
        return {"object": Memory(), "len": 1000}
    else:
        return {"object": Memory(len=len), "len": len}


@pytest.fixture(scope="session")
def dummy_data_gpt_inference(request):
    no_of_images, task, mem = request.param

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dict = {
        f"camera{i}": torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16, device=device)
        for i in range(no_of_images)
    }

    return {"image_dict": image_dict, "task": task, "mem": mem}
