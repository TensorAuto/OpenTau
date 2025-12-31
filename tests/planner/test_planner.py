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
