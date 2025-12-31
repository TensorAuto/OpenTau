import enum
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import patch

import pytest
import torch

from src.opentau.utils.hub import HubMixin
from src.opentau.utils.utils import (
    capture_timestamp_utc,
    encode_accelerator_state_dict,
    format_big_number,
    get_channel_first_image_shape,
    get_safe_dtype,
    get_safe_torch_device,
    has_method,
    init_logging,
    inside_slurm,
    is_amp_available,
    is_launched_with_accelerate,
    is_torch_device_available,
    is_valid_numpy_dtype_string,
    log_say,
    say,
)
from tests.fixtures.utils import DummyHubMixin
from tests.utils import require_cuda


class DummyAccelerator:
    is_main_process: bool
    process_index: int


class DummyEnum(enum.Enum):
    a = "a"
    b = "b"


@dataclass
class DummyDataClass:
    c: int
    d: int


@require_cuda
@pytest.mark.parametrize("device, log", [("cuda", False), ("cpu", False), ("abc", False), ("cpu", True)])
def test_get_safe_torch_device(device: str, log: bool):
    """
    Tests if the device is correctly getting converted into torch.device
    """
    try:
        torch_device = get_safe_torch_device(try_device=device, log=log)
        assert torch_device is not None

        assert isinstance(torch_device, torch.device)
    except Exception as e:
        if isinstance(e, RuntimeError):
            assert True
        else:
            pytest.fail(f"Error occurred is {e}")


@pytest.mark.parametrize(
    "dtype, device",
    [
        (torch.float32, torch.device("cpu")),
        (torch.float32, "cuda"),
        (torch.float64, "mps"),
        (torch.float64, "abc"),
    ],
)
def test_get_safe_dtype(dtype: torch.dtype, device: torch.device | str):
    """
    Tests if the dtype for device is correctly set
    """
    if device == torch.device:
        try:
            get_safe_dtype(dtype, device)
        except Exception as e:
            pytest.fail(f"Exception occurred is {e}")
    elif device == "mps" and dtype == torch.float64:
        final_dtype = get_safe_dtype(dtype, device)
        assert final_dtype == torch.float32
    else:
        final_dtype = get_safe_dtype(dtype, device)
        assert isinstance(final_dtype, torch.dtype)

        assert final_dtype == dtype


@pytest.mark.parametrize("device", [("cpu"), ("cuda"), ("mps"), ("abc")])
def test_is_torch_device_available(device: str):
    """
    Tests if the device is available or not. If not, raises ValurError
    """
    if device in ("cpu", "cuda", "mps"):
        torch_device = is_torch_device_available(device)
        assert isinstance(torch_device, bool)
    else:
        with pytest.raises(ValueError):
            is_torch_device_available(device)


@pytest.mark.parametrize("device", [("cpu"), ("cuda"), ("mps"), ("abc")])
def test_is_amp_available(device: str):
    """
    Tests if the amp is available or not. If not, ValueError is raised
    """
    if device in ("cuda", "cpu"):
        amp_exists = is_amp_available(device)
        assert amp_exists
    elif device == "mps":
        amp_exists = is_amp_available(device)
        assert not amp_exists
    else:
        with pytest.raises(ValueError):
            is_amp_available(device)


def test_capture_timestamp_utc():
    """
    Tests if the proper time is returned
    """
    time = capture_timestamp_utc()
    assert isinstance(time, datetime)


@pytest.mark.parametrize(
    "object1, method, ground_truth",
    [(DummyHubMixin(), "_save_pretrained", True), (DummyHubMixin(), "pretrained", False)],
)
def test_has_method(object1: HubMixin, method: str, ground_truth: bool):
    """
    Tests of the given class instance has a particular attribute
    """
    assert has_method(object1, method) == ground_truth


@pytest.mark.parametrize("string, ground_truth", [("float32", True), ("100", False)])
def test_is_valid_numpy_dtype_string(string: str, ground_truth: bool):
    """
    Tests of the valid dtype can be converted into numpy dtype
    """
    assert is_valid_numpy_dtype_string(string) == ground_truth


def test_inside_slurm_returns_true_when_in_job(monkeypatch):
    """
    GIVEN the 'SLURM_JOB_ID' environment variable is set
    WHEN inside_slurm() is called
    THEN it should return True
    """
    # Temporarily set the environment variable for the duration of this test.
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    assert inside_slurm() is True


def test_inside_slurm_returns_false_when_not_in_job(monkeypatch):
    """
    GIVEN the 'SLURM_JOB_ID' environment variable is NOT set
    WHEN inside_slurm() is called
    THEN it should return False
    """
    # Temporarily delete the environment variable for this test.
    # `raising=False` ensures no error is thrown if it wasn't set to begin with.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert inside_slurm() is False


def test_init_logging_clears_previous_handlers(reset_logging_state):
    """Tests that calling init_logging multiple times doesn't add duplicate handlers."""
    # Add a dummy handler first
    logging.getLogger().addHandler(logging.FileHandler("dummy.log"))
    assert len(logging.root.handlers) > 0

    init_logging()
    assert len(logging.root.handlers) == 1  # Should have cleared the dummy and added its own

    init_logging()
    assert len(logging.root.handlers) == 1  # Should still only have one handler


def test_format_big_number():
    assert format_big_number(4) == "4"
    assert format_big_number(4 * 1e3) == "4K"
    assert format_big_number(4 * 1e6) == "4M"
    assert format_big_number(4 * 1e9) == "4B"
    assert format_big_number(4 * 1e12) == "4T"
    assert format_big_number(4 * 1e15) == "4Q"
    assert format_big_number(4023 * 1e12, precision=3) == "4.023Q"
    assert format_big_number(4023 * 1e12, precision=2) == "4.02Q"
    assert (
        format_big_number(
            4023 * 1e15,
        )
        == 4.023
    )


def test_say_on_mac_non_blocking():
    """
    Tests the say function on macOS with non-blocking behavior.
    """

    with patch("platform.system", return_value="Darwin"), patch("os.system") as mock_os_system:
        say("hello")
        mock_os_system.assert_called_once_with('say "hello" &')


def test_say_on_mac_blocking():
    """
    Tests the say function on macOS with blocking behavior.
    """
    with patch("platform.system", return_value="Darwin"), patch("os.system") as mock_os_system:
        say("hello", blocking=True)
        mock_os_system.assert_called_once_with('say "hello"')


def test_say_on_linux_non_blocking():
    """
    Tests the say function on Linux with non-blocking behavior.
    """

    with patch("platform.system", return_value="Linux"), patch("os.system") as mock_os_system:
        say("hello")
        mock_os_system.assert_called_once_with('spd-say "hello"')


def test_say_on_linux_blocking():
    """
    Tests the say function on Linux with blocking behavior.
    """
    with patch("platform.system", return_value="Linux"), patch("os.system") as mock_os_system:
        say("hello", blocking=True)
        mock_os_system.assert_called_once_with('spd-say "hello"  --wait')


def test_say_on_windows():
    """
    Tests the say function on Windows.
    """
    with patch("platform.system", return_value="Windows"), patch("os.system") as mock_os_system:
        say("hello")
        expected_cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('hello')\""
        )
        mock_os_system.assert_called_once_with(expected_cmd)


def test_log_say_without_sound():
    """
    Tests log_say when sounds are disabled.
    """
    with patch("logging.info") as mock_logging_info, patch(__name__ + ".say") as mock_say:
        log_say("test message", play_sounds=False)
        mock_logging_info.assert_called_once_with("test message")
        mock_say.assert_not_called()


def test_is_launched_with_accelerate_when_variable_is_set():
    """
    Tests that the function returns True when the environment variable
    'ACCELERATE_MIXED_PRECISION' is present.
    """
    # Use mocker to temporarily set the environment variable for this test
    with patch.dict(os.environ, {"ACCELERATE_MIXED_PRECISION": "fp16"}):
        # Assert that the function returns True
        assert is_launched_with_accelerate() is True


def test_is_launched_with_accelerate_when_variable_is_not_set():
    """
    Tests that the function returns False when the environment variable
    'ACCELERATE_MIXED_PRECISION' is not present.
    """
    # Use mocker to ensure the environment is clear of the variable for this test
    with patch.dict(os.environ, clear=True):
        # Assert that the function returns False
        assert is_launched_with_accelerate() is False


def test_channels_last_to_channels_first():
    """
    Tests the conversion of a standard (h, w, c) shape.
    """
    assert get_channel_first_image_shape((224, 224, 3)) == (3, 224, 224)


def test_non_square_channels_last_to_channels_first():
    """
    Tests the conversion of a non-square (h, w, c) shape.
    """
    assert get_channel_first_image_shape((512, 256, 4)) == (4, 512, 256)


def test_channels_first_is_unchanged():
    """
    Tests that an already channels-first shape is returned without modification.
    """
    assert get_channel_first_image_shape((3, 224, 224)) == (3, 224, 224)


def test_another_invalid_shape_raises_value_error():
    """
    Tests that another ambiguous shape (where the first dimension is not
    the smallest) raises a ValueError.
    """
    with pytest.raises(ValueError):
        get_channel_first_image_shape((3, 3, 3))


@pytest.mark.parametrize(
    ["obj", "expected"],
    [
        [True, True],
        [False, False],
        [None, None],
        [3, 3],
        [4.0, 4.0],
        ["test string", "test string"],
        [torch.device("cpu"), "cpu"],
        [
            {"test.key": [1, 3, {}, torch.device("cpu")]},
            {"test_key": [1, 3, {}, "cpu"]},
        ],
        [
            (DummyDataClass(c=123, d=456), DummyEnum.a, DummyEnum.b),
            [{"c": 123, "d": 456}, "a", "b"],
        ],
    ],
)
def test_encode_accelerator_state_dict(obj, expected):
    output = encode_accelerator_state_dict(obj)
    assert output == expected, f"Expected {expected}, but got {output} for input"
