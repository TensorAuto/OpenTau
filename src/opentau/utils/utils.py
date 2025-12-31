#!/usr/bin/env python
import enum
import inspect

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import platform
import warnings
from copy import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Callable

import accelerate
import numpy as np
import torch


def inside_slurm():
    """Check whether the python process was launched through slurm"""
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


def auto_torch_device():
    """Automatically select the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_safe_torch_device(try_device: str, log: bool = False, accelerator: Callable = None) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    match try_device:
        case "cuda":
            assert torch.cuda.is_available()
            device = accelerator.device if accelerator else torch.device("cuda")
        case "mps":
            assert torch.backends.mps.is_available()
            device = torch.device("mps")
        case "cpu":
            device = torch.device("cpu")
            if log:
                logging.warning("Using CPU, this will be slow.")
        case _:
            device = torch.device(try_device)
            if log:
                logging.warning(f"Using custom {try_device} device.")

    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device '{try_device}.")


def is_amp_available(device: str):
    if device in ["cuda", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


# Global variable to ensure logging is initialized only once
_logging_init_stack = ""


def _format_stack(stack: list[inspect.FrameInfo]) -> str:
    return "\n".join(
        f"  File '{frame.filename}', line {frame.lineno}, in {frame.function}"
        for frame in stack[1:]  # skip the current frame
    )


def init_logging(accelerator: accelerate.Accelerator | None = None, level=logging.INFO):
    global _logging_init_stack
    stack = inspect.stack()

    if _logging_init_stack:
        warnings.warn(
            f"""Logging was already initialized through the following stack:
            {_logging_init_stack}
            Not initializing again through the following stack:
            {_format_stack(stack)}""",
            stacklevel=2,
        )
    else:
        _logging_init_stack = _format_stack(stack)

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fnameline = f"{record.pathname}:{record.lineno}"
            return f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.getMessage()}"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(level=level, force=True, handlers=[console_handler])

    if accelerator and not accelerator.is_main_process:
        # Disable duplicate logging on non-main processes
        logging.info(f"Setting logging level on non-main process {accelerator.process_index} to WARNING.")
        logging.getLogger().setLevel(logging.WARNING)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
        if not blocking:
            cmd += " &"
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
        if blocking:
            cmd += "  --wait"
    elif platform.system() == "Windows":
        # TODO(rcadene): Make blocking option work for Windows
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    os.system(cmd)  # nosec: B605


def log_say(text, play_sounds, blocking=False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


def is_launched_with_accelerate() -> bool:
    return "ACCELERATE_MIXED_PRECISION" in os.environ


def attempt_torch_compile(fn: callable, device_hint=None) -> callable:
    r"""Attempt to compile a PyTorch function using `torch.compile`.
    The argument `device_hint` is used to check if torch.compile works reliably on the device.
    Compilation is skipped if the device is MPS (Metal Performance Shaders) as it is experimental.
    """
    if device_hint and "mps" in str(device_hint):
        logging.warning("torch.compile is experimental on MPS devices. Compilation skipped.")
        return fn

    if hasattr(torch, "compile"):
        logging.info("Attempting to compile the policy with torch.compile()...")
        try:
            # Other options: "default", "max-autotune" (longer compile time)
            fn = torch.compile(fn)
            logging.info("Policy compiled successfully.")
        except Exception as e:
            logging.warning(f"torch.compile failed with error: {e}. Proceeding without compilation.")
    else:
        logging.warning(
            "torch.compile is not available. Requires PyTorch 2.0+. Proceeding without compilation."
        )

    return fn


def create_dummy_observation(cfg, device, dtype=torch.bfloat16) -> dict:
    camera_observations = {
        f"camera{i}": torch.zeros((1, 3, *cfg.resolution), dtype=dtype, device=device)
        for i in range(cfg.num_cams)
    }
    return {
        **camera_observations,
        "state": torch.zeros((1, cfg.max_state_dim), dtype=dtype, device=device),
        "prompt": ["Pick up yellow lego block and put it in the bin"],
        "img_is_pad": torch.zeros((1, cfg.num_cams), dtype=torch.bool, device=device),
        "action_is_pad": torch.zeros((1, cfg.action_chunk), dtype=torch.bool, device=device),
    }


def encode_accelerator_state_dict(obj):
    """Encodes an object into a json/yaml-compatible primitive type."""
    if isinstance(obj, enum.Enum):
        return encode_accelerator_state_dict(obj.value)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [encode_accelerator_state_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key.replace(".", "_"): encode_accelerator_state_dict(value) for key, value in obj.items()}
    elif is_dataclass(obj):
        return {f.name: encode_accelerator_state_dict(getattr(obj, f.name)) for f in fields(obj)}
    else:
        return str(obj)  # Fallback to string representation for unsupported types


def on_accelerate_main_proc(*, local=False, _sync=False):
    r"""Returns a decorator to run a function only on the main process when using `accelerate`.
    If `local` is True (defaults to False), the function will run on the main process of each node
        (useful for multi-node setups).
    If `_sync` is True (defaults to False), the output of the function will be broadcasted to all processes.
    If `_sync` is True, you must ensure that all processes call the decorated function, otherwise it will deadlock.

    YOU SHOULD BE EXTREMELY CAREFUL WHEN USING THIS DECORATOR with _sync=True. Consider the following example

    >>> @on_accelerate_main_proc()
    ... def f():
    ...     return g()
    ...
    ... @on_accelerate_main_proc(_sync=True)
    ... def g():
    ...     return 42

    In this case, if f() is called on all processes, they will deadlock at g() because child processes don't even
    enter f(), hence never call g(), and thus won't reach the broadcast.

    Another example:
    >>> @on_accelerate_main_proc(_sync=cond())
    ... def f():
    ...     print("hi")

    In this case, if cond() is not the same on all processes, they may deadlock because one or more processes don't
    require a _sync, hence won't reach the broadcast, blocking the other processes.

    To prevent accidental misues, we set _sync to False by default.

    TODO: record list of processes that called the decorated function with _sync=True and only send result to them
    """

    def decorator(func):
        if local and _sync:
            warnings.warn(
                f"Using local=True with _sync=True when decorating {func.__qualname__} forces a wait_for_everyone. "
                "But the broadcast is not necessary returning the correct result on all nodes.",
                stacklevel=2,
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            state = accelerate.state.PartialState()
            if not is_launched_with_accelerate() or not state.use_distributed:
                return func(*args, **kwargs)

            output, exception = None, None
            flag = state.is_local_main_process if local else state.is_main_process
            if flag:
                try:
                    output = func(*args, **kwargs)
                except Exception as e:
                    exception = e

            if _sync:
                payload = [output, exception]
                accelerate.utils.broadcast_object_list(payload, from_process=0)
                output, exception = payload

            if exception is not None:
                raise RuntimeError("An exception occurred in the main process.") from exception
            return output

        return wrapper

    return decorator
