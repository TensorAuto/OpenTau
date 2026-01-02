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

"""Utility functions for policy implementations in OpenTau.

This module provides helper functions for managing data queues, inspecting model
properties (device, dtype), determining output shapes, and logging model loading
information.
"""

import logging
from collections import deque

import torch
from torch import nn


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
) -> dict[str, deque]:
    """Populates queues with batch data.

    If a queue is not full (e.g. at the start of an episode), it is filled by repeating
    the first observation. Otherwise, the latest observation is appended.

    Args:
        queues: A dictionary of deques to be populated.
        batch: A dictionary containing the data to add to the queues.
        exclude_keys: A list of keys to exclude from population. Defaults to None.

    Returns:
        dict[str, deque]: The updated dictionary of queues.
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note:
        Assumes that all parameters have the same device.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.device: The device of the module's parameters.
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note:
        Assumes that all parameters have the same dtype.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.dtype: The data type of the module's parameters.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module: A PyTorch module.
        input_shape: A tuple representing the input shape, e.g., (batch_size, channels, height, width).

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys: Keys that were expected but not found.
        unexpected_keys: Keys that were found but not expected.
    """
    if missing_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")
