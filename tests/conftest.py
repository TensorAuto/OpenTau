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

import gc
import os

import pytest
import torch
from huggingface_hub import HfApi

from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from tests.utils import device

# Set OpenGL platform to EGL for headless environments
# This must be done before any OpenGL imports
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# Ensure the environment variable is set for subprocesses
os.environ["PYOPENGL_PLATFORM"] = "egl"


# Patch the problematic EGLDeviceEXT attribute that doesn't exist in some OpenGL versions
def patch_egl_device_ext():
    """Patch EGLDeviceEXT to avoid AttributeError in headless environments."""
    try:
        import ctypes

        from OpenGL import EGL

        if not hasattr(EGL, "EGLDeviceEXT"):
            # Create a mock EGLDeviceEXT ctypes type if it doesn't exist
            EGL.EGLDeviceEXT = ctypes.c_void_p
    except ImportError:
        pass


# Apply the patch before any imports
patch_egl_device_ext()

# Import fixture modules as plugins
pytest_plugins = [
    "tests.fixtures.dataset_factories",
    "tests.fixtures.files",
    "tests.fixtures.hub",
    "tests.fixtures.optimizers",
    "tests.fixtures.config_factory",
    "tests.fixtures.planner",
    "tests.fixtures.policies",
    "tests.fixtures.utils",
    "tests.fixtures.configs",
]


def pytest_collection_finish():
    print(f"\nTesting with {device=}")


@pytest.fixture
def patch_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)


@pytest.fixture(autouse=True)
def release_cuda_memory_after_gpu_test(request):
    """Drop a finished GPU test's CUDA memory before the next one allocates.

    `pytest -m gpu -n 0` runs every GPU test in one process. PyTorch frees its
    own cached blocks when an allocation fails, but it never runs a Python GC,
    so the reference cycles inside a just-finished test's `nn.Module` graph keep
    whole models resident. Peaks that should overlap add up instead. Measured:
    the four pi06/pi07 policy files OOM on a 24 GiB budget without this sweep
    and pass with it, though no single test in them exceeds 19.6 GiB.

    The marker check keeps this off the ~2600 CPU tests, where it would be pure
    overhead. On a failing test pytest still holds the frame for the traceback,
    so the sweep only reliably reclaims after a pass — which is the case that
    matters for the tests that follow.
    """
    yield
    if request.node.get_closest_marker("gpu") is None:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def get_huggingface_api():
    # Deliberately no whoami() call: its result (the username) is unused by the
    # only consumer (test_save_pretrained discards api/username, uses hf_token).
    # whoami() hits the same rate-limit-prone endpoint that has flaked CI at
    # setup, so we skip it; the token is validated implicitly when the push test
    # actually writes to the Hub. The middle tuple element stays for arity.
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=hf_token)

    return api, None, hf_token


@pytest.fixture(scope="session", autouse=True)
def add_mock_dataset_name_mapping():
    """
    Add a mock standard data format mapping for "mock_dataset" used by mock dataset configs
    """
    mock_name_mapping = {
        "camera0": "camera0",
        "camera1": "camera1",
        "state": "state",
        "actions": "actions",
        "prompt": "task",
        "response": "response",
    }

    DATA_FEATURES_NAME_MAPPING["mock_dataset"] = mock_name_mapping
