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
import json
import os
from pathlib import Path
from typing import Any

import pytest

from opentau.utils.io_utils import (
    deserialize_json_into_object,
    silence_output_unless_error,
    write_video,
)


def create_temp_json(tmp_path: Path, data: Any) -> Path:
    """Helper function to create a temporary JSON file."""
    fpath = os.path.join(tmp_path, "test.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return fpath


def test_simple_dict(tmp_json_file):
    data = {"name": "Alice", "age": 30}
    json_path = tmp_json_file(data)
    obj = {"name": "", "age": 0}
    assert deserialize_json_into_object(json_path, obj) == data


def test_nested_structure(tmp_json_file):
    data = {"items": [1, 2, 3], "info": {"active": True}}
    json_path = tmp_json_file(data)
    obj = {"items": [0, 0, 0], "info": {"active": False}}
    assert deserialize_json_into_object(json_path, obj) == data


def test_tuple_conversion(tmp_json_file):
    data = {"coords": [10.5, 20.5]}
    json_path = tmp_json_file(data)
    obj = {"coords": (0.0, 0.0)}
    result = deserialize_json_into_object(json_path, obj)
    assert result["coords"] == (10.5, 20.5)


def test_type_mismatch_raises(tmp_json_file):
    data = {"numbers": {"bad": "structure"}}
    json_path = tmp_json_file(data)
    obj = {"numbers": [0, 0]}
    with pytest.raises(TypeError):
        deserialize_json_into_object(json_path, obj)


def test_missing_key_raises(tmp_json_file):
    data = {"one": 1}
    json_path = tmp_json_file(data)
    obj = {"one": 0, "two": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_extra_key_raises(tmp_json_file):
    data = {"one": 1, "two": 2}
    json_path = tmp_json_file(data)
    obj = {"one": 0}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


def test_list_length_mismatch_raises(tmp_json_file):
    data = {"nums": [1, 2, 3]}
    json_path = tmp_json_file(data)
    obj = {"nums": [0, 0]}
    with pytest.raises(ValueError):
        deserialize_json_into_object(json_path, obj)


@pytest.mark.parametrize(
    "dummyvideo",
    [2],
    indirect=True,
)
def test_write_video(dummyvideo, tmp_path):
    """
    Tests write video function by creating dummy video and storing it in the artifacts file
    """

    video_path = tmp_path / "test_video.mp4"
    frames = dummyvideo["frames"]
    fps = dummyvideo["fps"]

    write_video(video_path, frames, fps)

    assert os.path.exists(video_path)


@pytest.mark.parametrize(
    "content, obj",
    [({"test": "test"}, ["test list"]), (("test1", "test2"), {"test": "test"})],
)
def test_deserialize_json_into_object(content, obj, tmp_path):
    """
    Tests desearlizing json object by passing various type like dict, str, tuple
    """
    with open(tmp_path / "test.json", "w+") as f:
        json.dump(content, f)

    with pytest.raises(TypeError):
        deserialize_json_into_object(tmp_path / "test.json", obj)

    assert os.path.exists(tmp_path / "test.json")


def test_error_on_type_mismatch_tuple(tmp_path):
    """Tests error when a tuple is expected but JSON has a non-list type."""
    template_obj = {"point": (1, 2)}
    json_data = {"point": {"x": 1, "y": 2}}  # dict instead of list

    fpath = create_temp_json(tmp_path, json_data)

    with pytest.raises(TypeError):
        deserialize_json_into_object(fpath, template_obj)


def test_error_on_tuple_length_mismatch(tmp_path):
    """Tests that an error is raised if tuple/list lengths don't match."""
    template_obj = {"point": (0, 0, 0)}
    json_data = {"point": [10, 20]}

    fpath = create_temp_json(tmp_path, json_data)

    with pytest.raises(ValueError, match="Tuple length mismatch"):
        deserialize_json_into_object(fpath, template_obj)


def test_error_on_int_float(tmp_path):
    template_obj = {"point": 0}
    json_data = {"point": "3.2"}

    fpath = create_temp_json(tmp_path, json_data)

    with pytest.raises(TypeError):
        deserialize_json_into_object(fpath, template_obj)


@pytest.mark.parametrize("value", [(2), (1.0), ("4"), (True)])
def test_success_on_int_float(value, tmp_path):
    template_obj = {"point": value}
    json_data = {"point": value}

    fpath = create_temp_json(tmp_path, json_data)

    source = deserialize_json_into_object(fpath, template_obj)

    assert source == json_data


# These exercise the file-descriptor-level redirect — the robosuite-logger / mujoco-C
# case the helper exists for — via os.write(1/2, ...). Note: pytest swaps
# sys.stdout/sys.stderr for objects NOT backed by fds 1/2, so a plain print() bypasses
# the fd redirect under pytest and is not a valid probe here; write to the fds directly.
def test_silence_output_unless_error_mutes_on_success(capfd):
    """On the success path, fd-level stdout and stderr writes are discarded."""
    with silence_output_unless_error():
        os.write(1, b"stdout banner noise\n")
        os.write(2, b"stderr banner noise\n")

    out, err = capfd.readouterr()
    assert "stdout banner noise" not in out
    assert "stderr banner noise" not in err


def test_silence_output_unless_error_replays_on_failure(capfd):
    """If the block raises, the captured fd output is replayed to stderr with the label."""

    def _emit_then_raise():
        # Factored into a helper so the failing action is a call rather than a literal
        # raise as the block's last statement, which keeps static analyzers from
        # wrongly treating the post-block assertions as unreachable code.
        os.write(1, b"stdout before crash\n")
        os.write(2, b"stderr before crash\n")
        raise ValueError("boom")

    with (
        pytest.raises(ValueError, match="boom"),
        silence_output_unless_error(label="task=CloseFridge idx=3"),
    ):
        _emit_then_raise()

    out, err = capfd.readouterr()
    # The exception still propagates (asserted by pytest.raises) and the otherwise-muted
    # output is replayed on stderr, tagged with the label, so a failing worker stays
    # debuggable.
    assert "task=CloseFridge idx=3" in err
    assert "stdout before crash" in err
    assert "stderr before crash" in err


def test_silence_output_unless_error_restores_streams(capfd):
    """fd-level stdout/stderr resume working after the block exits."""
    with silence_output_unless_error():
        os.write(1, b"hidden\n")
    os.write(1, b"visible again\n")

    out, err = capfd.readouterr()
    assert "hidden" not in out
    assert "visible again" in out
