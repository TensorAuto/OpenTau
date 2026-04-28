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
import json
import logging
import re
from copy import deepcopy
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from opentau import available_vqa_datasets
from opentau.datasets.factory import make_dataset
from opentau.datasets.image_writer import image_array_to_pil_image
from opentau.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from opentau.datasets.utils import (
    flatten_dict,
    unflatten_dict,
)
from tests.fixtures.constants import DUMMY_CHW, DUMMY_HWC, DUMMY_REPO_ID


@pytest.fixture
def image_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "image": {
            "dtype": "image",
            "shape": DUMMY_CHW,
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    }
    return empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)


def test_dataset_initialization(tmp_path, lerobot_dataset_factory):
    kwargs = {
        "repo_id": DUMMY_REPO_ID,
        "total_episodes": 10,
        "total_frames": 400,
        "episodes": [2, 5, 6],
    }
    dataset = lerobot_dataset_factory(root=tmp_path / "test", **kwargs)

    assert dataset.repo_id == kwargs["repo_id"]
    assert dataset.meta.total_episodes == kwargs["total_episodes"]
    assert dataset.meta.total_frames == kwargs["total_frames"]
    assert dataset.episodes == kwargs["episodes"]
    assert dataset.num_episodes == len(kwargs["episodes"])
    assert dataset.num_frames == len(dataset)


def test_add_frame_missing_task(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n"
    ):
        dataset.add_frame({"state": torch.randn(1)})


def test_add_frame_missing_feature(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nMissing features: {'state'}\n"
    ):
        dataset.add_frame({"task": "Dummy task"})


def test_add_frame_extra_feature(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="Feature mismatch in `frame` dictionary:\nExtra features: {'extra'}\n"
    ):
        dataset.add_frame({"state": torch.randn(1), "task": "Dummy task", "extra": "dummy_extra"})


def test_add_frame_wrong_type(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError, match="The feature 'state' of dtype 'float16' is not of the expected dtype 'float32'.\n"
    ):
        dataset.add_frame({"state": torch.randn(1, dtype=torch.float16), "task": "Dummy task"})


def test_add_frame_wrong_shape(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' of shape '(1,)' does not have the expected shape '(2,)'.\n"),
    ):
        dataset.add_frame({"state": torch.randn(1), "task": "Dummy task"})


def test_add_frame_wrong_shape_python_float(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' is expected to be a numpy array, but got 'float'.\n"),
    ):
        dataset.add_frame({"state": 1.0, "task": "Dummy task"})


def test_add_frame_wrong_shape_torch_ndim_0(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' of shape '()' does not have the expected shape '(1,)'.\n"),
    ):
        dataset.add_frame({"state": torch.tensor(1.0), "task": "Dummy task"})


def test_add_frame_wrong_shape_numpy_ndim_0(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features)
    with pytest.raises(
        ValueError,
        match=re.escape("The feature 'state' is expected to be a numpy array, but got 'float32'.\n"),
    ):
        dataset.add_frame({"state": np.float32(1.0), "task": "Dummy task"})


def test_add_frame(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(1), "task": "Dummy task", "actions": torch.randn(1)})
    dataset.save_episode()

    assert len(dataset) == 1
    assert dataset[0]["task"] == "Dummy task"
    assert dataset[0]["task_index"] == 0
    assert dataset[0]["state"].ndim == 0


def test_add_frame_state_1d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(2), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2])


def test_add_frame_state_2d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(2, 4), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4])


def test_add_frame_state_3d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(2, 4, 3), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3])


def test_add_frame_state_4d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3, 5), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(2, 4, 3, 5), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3, 5])


def test_add_frame_state_5d(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (2, 4, 3, 5, 1), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": torch.randn(2, 4, 3, 5, 1), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].shape == torch.Size([2, 4, 3, 5, 1])


def test_add_frame_state_numpy(tmp_path, empty_lerobot_dataset_factory):
    features = {"state": {"dtype": "float32", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"state": np.array([1], dtype=np.float32), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["state"].ndim == 0


def test_add_frame_string(tmp_path, empty_lerobot_dataset_factory):
    features = {"caption": {"dtype": "string", "shape": (1,), "names": None}}
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, standardize=False)
    dataset.add_frame({"caption": "Dummy caption", "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["caption"] == "Dummy caption"


def test_add_frame_image_wrong_shape(image_dataset):
    dataset = image_dataset
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The feature 'image' of shape '(3, 128, 96)' does not have the expected shape '(3, 96, 128)' or '(96, 128, 3)'.\n"
        ),
    ):
        c, h, w = DUMMY_CHW
        dataset.add_frame({"image": torch.randn(c, w, h), "task": "Dummy task"})


def test_add_frame_image_wrong_range(image_dataset):
    """This test will display the following error message from a thread:
    ```
    Error writing image ...test_add_frame_image_wrong_ran0/test/images/image/episode_000000/frame_000000.png:
    The image data type is float, which requires values in the range [0.0, 1.0]. However, the provided range is [0.009678772038470007, 254.9776492089887].
    Please adjust the range or provide a uint8 image with values in the range [0, 255]
    ```
    Hence the image won't be saved on disk and save_episode will raise `FileNotFoundError`.
    """
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_CHW) * 255, "task": "Dummy task"})
    with pytest.raises(FileNotFoundError):
        dataset.save_episode()


def test_add_frame_image(image_dataset):
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_CHW), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_h_w_c(image_dataset):
    dataset = image_dataset
    dataset.add_frame({"image": np.random.rand(*DUMMY_HWC), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_uint8(image_dataset):
    dataset = image_dataset
    image = np.random.randint(0, 256, DUMMY_HWC, dtype=np.uint8)
    dataset.add_frame({"image": image, "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_add_frame_image_pil(image_dataset):
    dataset = image_dataset
    image = np.random.randint(0, 256, DUMMY_HWC, dtype=np.uint8)
    dataset.add_frame({"image": Image.fromarray(image), "task": "Dummy task"})
    dataset.save_episode()

    assert dataset[0]["image"].shape == torch.Size(DUMMY_CHW)


def test_image_array_to_pil_image_wrong_range_float_0_255():
    image = np.random.rand(*DUMMY_HWC) * 255
    with pytest.raises(ValueError):
        image_array_to_pil_image(image)


# TODO(aliberts):
# - [ ] test various attributes & state from init and create
# - [ ] test init with episodes and check num_frames
# - [ ] test add_episode
# - [ ] test push_to_hub
# - [ ] test smaller methods


def check_standard_data_format(item, delta_timestamps_params, dataset, train_pipeline_config):
    n_obs = getattr(train_pipeline_config.dataset_mixture, "n_obs_history", None)
    # the keys in standard data format + tensor shape
    if n_obs is not None:
        state_shape = (n_obs, train_pipeline_config.max_state_dim)

        def cam_shape_fn(res):
            return (n_obs, 3, *res)

        obs_pad_shape = (n_obs,)
    else:
        state_shape = (train_pipeline_config.max_state_dim,)

        def cam_shape_fn(res):  # type: ignore[no-redef]
            return (3, *res)

        obs_pad_shape = (1,)

    keys_shape_required = [
        ("state", state_shape),
        ("actions", (train_pipeline_config.action_chunk, train_pipeline_config.max_action_dim)),
        ("prompt", None),
        ("response", None),
        ("img_is_pad", (train_pipeline_config.num_cams,)),
        ("action_is_pad", (train_pipeline_config.action_chunk,)),
        ("obs_history_is_pad", obs_pad_shape),
    ]
    for i in range(train_pipeline_config.num_cams):
        keys_shape_required.append((f"camera{i}", cam_shape_fn(train_pipeline_config.resolution)))

    # enforce standard data format
    for key, shape in keys_shape_required:
        if key not in item:
            raise ValueError(f'Missing key in dataset: "{key}" not in {dataset}.')

        if "camera" in key:
            assert item[key].dtype == torch.bfloat16, f"{key}"
            assert item[key].max() <= 1.0, f"{key}"
            assert item[key].min() >= 0.0, f"{key}"
            assert item[key].shape == shape, f"{key}"
        elif key == "state" or key == "actions":
            assert item[key].shape == shape, f"{key}"
        elif key == "prompt" or key == "response":
            assert type(item[key]) is str, f"{key}"
        elif key in ("img_is_pad", "action_is_pad", "obs_history_is_pad"):
            assert item[key].shape == shape, f"{key}"
            assert isinstance(item[key], torch.BoolTensor), f"{key}"

    # test delta_timestamps — per-feature keys
    dt_mean = delta_timestamps_params[0]
    expected_obs_len = n_obs if n_obs is not None else 1
    for key, val in dt_mean.items():
        if key == "action":
            assert val.shape == (train_pipeline_config.action_chunk,)
        else:
            assert val.shape == (expected_obs_len,), f"{key} has unexpected shape {val.shape}"


@pytest.mark.slow  # 3 sec
@pytest.mark.parametrize(
    "repo_id",
    [
        "lerobot/droid_100",
        "lerobot/aloha_mobile_cabinet",
        "danaaubakirova/koch_test",
    ],
)
def test_lerobot_dataset_factory(dataset_config, train_pipeline_config, repo_id):
    """
    Tests that:
        - we can create a dataset with the factory.
        - for a commonly used set of data keys, the data dimensions are correct.
    """
    dataset_config.vqa = None
    dataset_config.repo_id = repo_id
    dataset_config.root = None
    dataset_config.revision = None

    dataset = make_dataset(dataset_config, train_pipeline_config)
    delta_timestamps_params = dataset.delta_timestamps_params

    item = dataset[0]

    check_standard_data_format(item, delta_timestamps_params, dataset, train_pipeline_config)


@pytest.mark.slow  # 5 sec
@pytest.mark.parametrize(
    "repo_id",
    [
        "lerobot/droid_100",
        "lerobot/aloha_mobile_cabinet",
    ],
)
def test_do_not_use_imagenet_stats(dataset_config, train_pipeline_config, repo_id):
    """
    Tests that:
        - we can create a dataset with the factory.
        - for a commonly used set of data keys, the data dimensions are correct.
    """
    dataset_config.vqa = None
    dataset_config.repo_id = repo_id
    dataset_config.root = None
    dataset_config.revision = None
    dataset_config.use_imagenet_stats = False

    dataset = make_dataset(dataset_config, train_pipeline_config)
    delta_timestamps_params = dataset.delta_timestamps_params

    item = dataset[0]

    check_standard_data_format(item, delta_timestamps_params, dataset, train_pipeline_config)


# TODO(aliberts): Move to more appropriate location
def test_flatten_unflatten_dict():
    d = {
        "obs": {
            "min": 0,
            "max": 1,
            "mean": 2,
            "std": 3,
        },
        "action": {
            "min": 4,
            "max": 5,
            "mean": 6,
            "std": 7,
        },
    }

    original_d = deepcopy(d)
    d = unflatten_dict(flatten_dict(d))

    # test equality between nested dicts
    assert json.dumps(original_d, sort_keys=True) == json.dumps(d, sort_keys=True), f"{original_d} != {d}"


def test_dataset_feature_with_forward_slash_raises_error():
    # make sure dir does not exist
    from opentau.constants import HF_OPENTAU_HOME

    dataset_dir = HF_OPENTAU_HOME / "opentau/test/with/slash"
    # make sure does not exist
    if dataset_dir.exists():
        dataset_dir.rmdir()

    with pytest.raises(ValueError):
        LeRobotDataset.create(
            repo_id="opentau/test/with/slash",
            fps=30,
            features={"a/b": {"dtype": "float32", "shape": 2, "names": None}},
        )


def test_vqa_dataset_imports():
    for dataset in available_vqa_datasets:
        import_module(f"opentau.datasets.vqa.{dataset}")


def _make_dummy_mp4(path, fps=60, num_frames=100, width=128, height=96):
    """Create a minimal MP4 test video using ffmpeg with solid-color frames."""
    import shutil
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    duration = num_frames / fps
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={width}x{height}:r={fps}:d={duration:.6f}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-g",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _make_time_varying_mp4(path, fps=60, num_frames=300, width=128, height=96):
    """Create an MP4 whose content changes over time.

    Uses ffmpeg's ``testsrc2`` filter, which renders a test card with a
    changing timer — every frame is visibly different from the last.
    """
    import shutil
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    duration = num_frames / fps
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration:.6f}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-g",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_deferred_video_add_frame_and_save(tmp_path, empty_lerobot_dataset_factory):
    """Test that frames can be added without image data for deferred video keys."""
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": ["channels", "height", "width"],
            "info": None,
        },
    }
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "deferred_test",
        features=features,
        deferred_video_keys={"observation.images.top"},
    )

    # Add frames without video data
    for i in range(10):
        dataset.add_frame(
            {
                "state": np.array([float(i), float(i + 1)], dtype=np.float32),
                "task": "Dummy task",
            }
        )

    # save_episode should succeed without video data
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 10

    # The parquet file should exist
    parquet_path = dataset.root / dataset.meta.get_data_file_path(0)
    assert parquet_path.is_file()

    # The video file should NOT exist yet
    video_path = dataset.root / dataset.meta.get_video_file_path(0, "observation.images.top")
    assert not video_path.is_file()


def test_deferred_video_attach_video(tmp_path, empty_lerobot_dataset_factory):
    """Test full workflow: add frames, save episode, attach video."""
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": ["channels", "height", "width"],
            "info": None,
        },
    }
    target_fps = 10
    num_frames = 5
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=target_fps,
        root=tmp_path / "attach_test",
        features=features,
        deferred_video_keys={"observation.images.top"},
        standardize=False,
    )

    for i in range(num_frames):
        dataset.add_frame(
            {
                "state": np.array([float(i), float(i + 1)], dtype=np.float32),
                "task": "Dummy task",
            }
        )
    dataset.save_episode()

    # Create a source video at a different FPS (60fps, 100 frames = 1.67s)
    src_video = _make_dummy_mp4(tmp_path / "source.mp4", fps=60, num_frames=100, width=128, height=96)

    # Attach the video - it should be resampled to 10fps and trimmed to 5 frames
    result_path = dataset.attach_video(
        episode_index=0,
        video_key="observation.images.top",
        input_video_path=src_video,
        overwrite=True,
    )

    assert result_path.is_file()

    # Verify the output video exists at the expected dataset path
    expected_path = dataset.root / dataset.meta.get_video_file_path(0, "observation.images.top")
    assert expected_path.is_file()
    assert result_path == expected_path


def test_deferred_video_attach_video_start_time(tmp_path):
    """A non-zero ``start_time`` must shift the extracted segment."""
    from opentau.datasets.video_utils import decode_video_frames

    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": ["channels", "height", "width"],
            "info": None,
        },
    }
    target_fps = 10
    num_frames = 5

    def _build_dataset(root):
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=target_fps,
            root=root,
            features=features,
            deferred_video_keys={"observation.images.top"},
            standardize=False,
        )
        for i in range(num_frames):
            dataset.add_frame(
                {
                    "state": np.array([float(i), float(i + 1)], dtype=np.float32),
                    "task": "Dummy task",
                }
            )
        dataset.save_episode()
        return dataset

    # Source video has time-varying content so different offsets yield different pixels.
    src_video = _make_time_varying_mp4(tmp_path / "source.mp4", fps=60, num_frames=300, width=128, height=96)

    ds_no_offset = _build_dataset(tmp_path / "ds_no_offset")
    ds_no_offset.attach_video(
        episode_index=0,
        video_key="observation.images.top",
        input_video_path=src_video,
        overwrite=True,
    )

    ds_offset = _build_dataset(tmp_path / "ds_offset")
    ds_offset.attach_video(
        episode_index=0,
        video_key="observation.images.top",
        input_video_path=src_video,
        overwrite=True,
        start_time=2.0,
    )

    path_no_offset = ds_no_offset.root / ds_no_offset.meta.get_video_file_path(0, "observation.images.top")
    path_offset = ds_offset.root / ds_offset.meta.get_video_file_path(0, "observation.images.top")

    frames_no_offset = decode_video_frames(path_no_offset, [0.0], tolerance_s=0.1)
    frames_offset = decode_video_frames(path_offset, [0.0], tolerance_s=0.1)

    assert frames_no_offset.shape == frames_offset.shape
    # The two first frames should be clearly distinct (testsrc2 is different every frame).
    assert not torch.allclose(frames_no_offset, frames_offset, atol=1e-2)


def test_resample_and_trim_video_invalid_start_time(tmp_path):
    """``start_time`` must be finite and non-negative."""
    from opentau.datasets.video_utils import resample_and_trim_video

    src = _make_dummy_mp4(tmp_path / "src.mp4", fps=30, num_frames=30)
    out = tmp_path / "out.mp4"

    for bad in (-1.0, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="start_time"):
            resample_and_trim_video(
                input_path=src,
                output_path=out,
                target_fps=10,
                num_frames=5,
                start_time=bad,
            )


def test_deferred_video_multi_episode_hwc_convention(tmp_path, empty_lerobot_dataset_factory):
    """Saving two episodes in sequence with deferred video keys declared in
    LeRobot v2.1's (H, W, C) convention must not crash.

    Regression test for a bug in the placeholder-stats fallback: it assumed
    shape[0] was the channel axis, which is true for (C, H, W) but produces
    shape (H, 1, 1) for (H, W, C) — aggregate_stats then rejected it on the
    second save_episode because its shape-check requires (3, 1, 1) for
    features with 'image' in the key.
    """
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (96, 128, 3),
            "names": ["height", "width", "channel"],
            "info": None,
        },
    }
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "hwc_test",
        features=features,
        deferred_video_keys={"observation.images.top"},
    )
    for ep in range(2):
        for i in range(3):
            dataset.add_frame(
                {
                    "state": np.array([float(i), float(i + ep)], dtype=np.float32),
                    "task": "Dummy task",
                }
            )
        dataset.save_episode()

    assert dataset.meta.total_episodes == 2
    # Placeholder stats for the deferred key must use the channel axis (3),
    # not the height axis (96).
    stats = dataset.meta.stats["observation.images.top"]
    assert stats["min"].shape == (3, 1, 1)
    assert stats["max"].shape == (3, 1, 1)
    assert stats["mean"].shape == (3, 1, 1)
    assert stats["std"].shape == (3, 1, 1)


def test_deferred_video_invalid_key(tmp_path, empty_lerobot_dataset_factory):
    """Creating a dataset with invalid deferred video keys should raise ValueError."""
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
    }
    with pytest.raises(ValueError, match="not declared as video features"):
        empty_lerobot_dataset_factory(
            root=tmp_path / "invalid_deferred",
            features=features,
            deferred_video_keys={"nonexistent_camera"},
        )


def test_deferred_video_multi_episode(tmp_path, empty_lerobot_dataset_factory):
    """Test deferred video with multiple episodes."""
    features = {
        "state": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, 96, 128),
            "names": ["channels", "height", "width"],
            "info": None,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=10,
        root=tmp_path / "multi_ep_test",
        features=features,
        deferred_video_keys={"observation.images.top"},
        standardize=False,
    )

    # Episode 0: 5 frames
    for i in range(5):
        dataset.add_frame(
            {
                "state": np.array([float(i), 0.0], dtype=np.float32),
                "task": "Task A",
            }
        )
    dataset.save_episode()

    # Episode 1: 8 frames
    for i in range(8):
        dataset.add_frame(
            {
                "state": np.array([0.0, float(i)], dtype=np.float32),
                "task": "Task B",
            }
        )
    dataset.save_episode()

    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 13

    # Attach videos for both episodes
    src_video = _make_dummy_mp4(tmp_path / "source.mp4", fps=60, num_frames=300, width=128, height=96)

    for ep_idx in range(2):
        dataset.attach_video(
            episode_index=ep_idx,
            video_key="observation.images.top",
            input_video_path=src_video,
            overwrite=True,
        )

    # Verify both videos exist
    for ep_idx in range(2):
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, "observation.images.top")
        assert video_path.is_file()


def test_overlay_resolves_video_to_source_repo(tmp_path, monkeypatch, lerobot_dataset_factory, info_factory):
    """Overlay datasets resolve videos against the source repo, lazily caching
    them under HF_OPENTAU_HOME/<source_repo> and reusing the file on repeat
    calls. The path is formatted with `original_episode_index` from
    episodes.jsonl, not the dense partition-local `episode_index`."""
    src_repo = "src-org/source-dataset"
    monkeypatch.setattr("opentau.datasets.lerobot_dataset.HF_OPENTAU_HOME", tmp_path / "cache")
    src_root = tmp_path / "cache" / src_repo

    # Pre-populate the source info.json so __init__ skips the network fetch.
    src_info_path = src_root / "meta" / "info.json"
    src_info_path.parent.mkdir(parents=True, exist_ok=True)
    src_info_path.write_text(
        json.dumps(
            {
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "chunks_size": 1000,
            }
        )
    )

    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1)
    info["videos"] = {"source_repo": src_repo, "source_revision": "abc123"}

    task = "Perform action 0."
    eps = {
        0: {"episode_index": 0, "tasks": [task], "length": 50, "original_episode_index": 17},
        1: {"episode_index": 1, "tasks": [task], "length": 50, "original_episode_index": 42},
        2: {"episode_index": 2, "tasks": [task], "length": 50, "original_episode_index": 99},
    }

    with patch("opentau.datasets.lerobot_dataset.hf_hub_download") as mock_dl:
        ds = lerobot_dataset_factory(root=tmp_path / "ds", info=info, episode_dicts=eps)
        # Pre-populated info.json must short-circuit the source-info download.
        mock_dl.assert_not_called()

    vid_key = ds.meta.video_keys[0]
    expected = src_root / f"videos/chunk-000/{vid_key}/episode_000017.mp4"

    with patch("opentau.datasets.lerobot_dataset.hf_hub_download") as mock_dl:
        # Local file missing → download is triggered with source-repo args.
        result = ds._resolve_video_path(0, vid_key)
        assert result == expected
        mock_dl.assert_called_once_with(
            repo_id=src_repo,
            filename=f"videos/chunk-000/{vid_key}/episode_000017.mp4",
            repo_type="dataset",
            revision="abc123",
            local_dir=src_root,
        )

        # Place the file at the expected location → second call skips re-download
        # (this is what lets a future vanilla load of the source repo reuse it).
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()
        mock_dl.reset_mock()
        assert ds._resolve_video_path(0, vid_key) == expected
        mock_dl.assert_not_called()

    # Overlay datasets must not list video files for snapshot_download / local existence checks.
    assert all(not p.startswith("videos/") for p in ds.get_episodes_file_paths())


def test_control_mode_warning_emitted_once_per_repo(tmp_path, lerobot_dataset_factory, info_factory, caplog):
    """When info.json lacks `control_mode`, the loader warns exactly once per
    repo_id within a process even across multiple LeRobotDataset instances."""
    from opentau.datasets import lerobot_dataset as ld_mod

    test_repo = "warn-once/test-repo"
    ld_mod._CONTROL_MODE_WARNED.discard(test_repo)

    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1)  # no `control_mode` key

    with caplog.at_level(logging.WARNING):
        ds_a = lerobot_dataset_factory(root=tmp_path / "a", repo_id=test_repo, info=info)
        ds_b = lerobot_dataset_factory(root=tmp_path / "b", repo_id=test_repo, info=info)

    assert ds_a.control_mode == "unknown"
    assert ds_b.control_mode == "unknown"
    matching = [r for r in caplog.records if "control_mode" in r.getMessage() and r.args == (test_repo,)]
    assert len(matching) == 1
