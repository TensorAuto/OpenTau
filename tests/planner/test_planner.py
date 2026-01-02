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

import pytest
from dotenv import load_dotenv
from openai._exceptions import OpenAIError

from opentau.planner import HighLevelPlanner, Memory

load_dotenv()


def test_set_openai_api_key():
    """
    checks if openai api key is set in the .env file , so it can he loaded while giving an api call
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    assert openai_key is not None


@pytest.mark.slow  # 5 sec
@pytest.mark.parametrize(
    "dummy_data_gpt_inference",
    [
        (1, "This is test task", Memory()),
        (3, "This is test task", Memory()),
        (3, "This is test task", None),
        (5, "This is test task", Memory(len=4)),
    ],
    indirect=True,
)
def test_gpt_inference(dummy_data_gpt_inference):
    """
    checks if api call to openai is made successfully and response is valid string object
    """
    planner = HighLevelPlanner()

    try:
        response = planner.gpt_inference(
            dummy_data_gpt_inference["image_dict"],
            dummy_data_gpt_inference["task"],
            dummy_data_gpt_inference["mem"],
        )
    except OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {str(e)}")

    assert response is not None
    assert isinstance(response, str)
