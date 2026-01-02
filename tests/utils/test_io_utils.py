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

from opentau.utils.io_utils import deserialize_json_into_object, write_video


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
