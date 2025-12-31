import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from src.opentau.utils.hub import HubMixin

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")


class DummyHubMixin(HubMixin):
    """
    Creates a dummy class that inherits HubMixin class
    """

    def __init__(self):
        super().__init__()

    def _save_pretrained(self, save_directory: str):
        try:
            with open(os.path.join(save_directory, "test_file"), "w+") as f:
                f.write("this is test file")
        except Exception as e:
            raise Exception("Error occurred") from e

    @classmethod
    # @validate_hf_hub_args
    def from_pretrained(cls, pretrained_name_or_path: str):
        file_path = hf_hub_download(
            repo_id="bert-base-uncased", filename=pretrained_name_or_path, token=hf_token
        )

        print(file_path)


@pytest.fixture(scope="session")
def create_dummy_hubmixin():
    return DummyHubMixin()


@pytest.fixture(scope="session")
def create_dummy_hubmixin_class():
    return DummyHubMixin


@pytest.fixture(scope="session")
def create_hubmixin_instance(request):
    """
    Creates hubmixin instance and its parameters
    """
    instance, ground_truth, save_directory, repo_id, push_to_hub = request.param
    return {
        "instance": instance,
        "ground_truth": ground_truth,
        "save_directory": save_directory,
        "repo_id": repo_id,
        "push_to_hub": push_to_hub,
    }


@pytest.fixture(scope="session")
def dummyvideo(request):
    """
    Creates a dummy video
    """
    fps = request.param

    frames = []

    for _ in range(fps * 60):
        img = np.zeros((320, 320, 3), dtype=np.float32)
        frames.append(img)

    return {"frames": frames, "fps": fps}


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Fixture to ensure logging state is clean before and after each test."""
    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level
    yield
    logging.root.handlers = original_handlers
    logging.root.setLevel(original_level)


@pytest.fixture
def tmp_json_file(tmp_path: Path):
    """Writes `data` to a temporary JSON file and returns the file's path."""

    def _write(data: Any) -> Path:
        file_path = tmp_path / "data.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        return file_path

    return _write
