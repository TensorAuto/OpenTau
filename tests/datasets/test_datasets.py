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
from unittest.mock import MagicMock, patch

import numpy as np
import packaging.version
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
from tests.utils import retry_on_hf_flakiness


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


def _assert_episode_row_alignment(dataset) -> None:
    """For each episode, verify hf_dataset rows in [from:to] all carry that episode_index.

    This catches the case where hf_dataset row order disagrees with the order
    used to build episode_data_index/epi2idx (e.g. unsorted self.episodes vs.
    sorted-by-filename row layout from Dataset.from_parquet).
    """
    # Reach into the underlying pa.Table to bypass the set_transform=hf_transform_to_torch
    # set in load_hf_dataset — calling `hf_dataset["episode_index"]` here would
    # route the column through the torch transform, which is wrong for a raw
    # episode_index check. Don't "simplify" this without removing the transform.
    ep_idx_col = dataset.hf_dataset.data.table.column("episode_index").to_pylist()
    for ep in dataset.episodes:
        pos = dataset.epi2idx[ep]
        start = int(dataset.episode_data_index["from"][pos])
        end = int(dataset.episode_data_index["to"][pos])
        assert end > start, f"Episode {ep} has empty row span [{start}:{end}]"
        rows = ep_idx_col[start:end]
        assert all(e == ep for e in rows), (
            f"Episode {ep} at hf_dataset[{start}:{end}] has mismatched episode_index values "
            f"(saw {sorted(set(rows))})"
        )


def test_dataset_unsorted_episodes_row_alignment(tmp_path, lerobot_dataset_factory):
    """Regression: passing an unsorted episodes list must still yield correct row indexing.

    Before the parquet-loader switch to Dataset.from_parquet, `data_files=[...]` produced
    rows in self.episodes order, so unsorted picks happened to work. With the
    glob-based loader, rows come back sorted by filename — so __init__ must sort
    self.episodes for episode_data_index/epi2idx to line up.
    """
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=10,
        total_frames=400,
        episodes=[6, 2, 5],
    )
    assert dataset.episodes == [2, 5, 6], "self.episodes should be sorted at the boundary"
    _assert_episode_row_alignment(dataset)


def test_dataset_sparse_episodes_row_alignment(tmp_path, lerobot_dataset_factory):
    """Sparse non-contiguous episodes filtered out of a larger corpus stay aligned."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=10,
        total_frames=400,
        episodes=[3, 7],
    )
    assert dataset.episodes == [3, 7]
    _assert_episode_row_alignment(dataset)
    # Confirm the filter actually dropped the unselected episodes.
    ep_idx_col = dataset.hf_dataset.data.table.column("episode_index").to_pylist()
    assert set(ep_idx_col) == {3, 7}


def test_dataset_no_episodes_loads_all(tmp_path, lerobot_dataset_factory):
    """episodes=None loads every episode and stays aligned with episode_data_index."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=200,
        episodes=None,
    )
    assert dataset.episodes == [0, 1, 2, 3]
    _assert_episode_row_alignment(dataset)


# ---------------------------------------------------------------------------
# excluded_episodes denylist + selected-episode normalization stats
# ---------------------------------------------------------------------------


def test_excluded_episodes_trims_from_full_set(tmp_path, lerobot_dataset_factory):
    """excluded_episodes with episodes=None drops the denied indices from the full list."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=250,
        excluded_episodes=[1, 3],
    )
    assert dataset.episodes == [0, 2, 4]
    # An explicit subset now exists, so subsequent downloads target only the kept set.
    assert dataset._episodes_were_specified is True
    _assert_episode_row_alignment(dataset)


def test_excluded_episodes_takes_precedence_over_episodes(tmp_path, lerobot_dataset_factory):
    """An index present in both `episodes` and `excluded_episodes` is excluded."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=250,
        episodes=[0, 1, 2],
        excluded_episodes=[1],
    )
    assert dataset.episodes == [0, 2]
    _assert_episode_row_alignment(dataset)


def test_excluded_episodes_none_is_noop(tmp_path, lerobot_dataset_factory):
    """excluded_episodes=None leaves the selection untouched."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=4,
        total_frames=200,
        episodes=[1, 2],
        excluded_episodes=None,
    )
    assert dataset.episodes == [1, 2]


def test_excluded_episodes_emptying_selection_raises(tmp_path, lerobot_dataset_factory):
    """Denylisting every selected episode is a config error, not a silent empty dataset."""
    with pytest.raises(ValueError, match="excluded_episodes removed every episode"):
        lerobot_dataset_factory(
            root=tmp_path / "test",
            repo_id=DUMMY_REPO_ID,
            total_episodes=4,
            total_frames=200,
            episodes=[0, 1],
            excluded_episodes=[0, 1],
        )


def _corrupt_state_std(episodes_stats: dict, ep_idx: int, value: float = 1000.0) -> None:
    """Inflate one episode's ``state`` std in-place to simulate a corrupt episode."""
    std = episodes_stats[ep_idx]["stats"]["state"]["std"]
    episodes_stats[ep_idx]["stats"]["state"]["std"] = [float(value)] * len(std)


def test_out_of_selection_corrupt_episode_excluded_from_norm_stats(
    tmp_path, info_factory, episodes_stats_factory, lerobot_dataset_factory
):
    """A corrupt episode OUTSIDE ``episodes`` must not poison the normalization stats.

    Regression for the norm-head bug: the policy normalizer pools ``ds.meta.stats``,
    which must reflect the SELECTED-episode aggregate, not the full on-disk set.
    """
    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1)
    episodes_stats = episodes_stats_factory(features=info["features"], total_episodes=3)
    _corrupt_state_std(episodes_stats, ep_idx=0)  # episode 0 is out of the selection below

    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=150,
        info=info,
        episodes_stats=episodes_stats,
        episodes=[1, 2],
    )

    # Selected-episode aggregate is clean (stats_factory uses std=0.25) ...
    assert np.allclose(dataset.meta.stats["state"]["std"], 0.25)
    # ... and that selected aggregate is the exact object the mixture will read.
    assert dataset.stats is dataset.meta.stats


def test_corrupt_episode_inside_selection_is_inflated_without_exclusion(
    tmp_path, info_factory, episodes_stats_factory, lerobot_dataset_factory
):
    """Negative control: a corrupt episode INSIDE the selection DOES inflate the std.

    Guards against a false pass in the test above by proving the fixture really
    injects a poisoned episode that aggregation would otherwise propagate.
    """
    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1)
    episodes_stats = episodes_stats_factory(features=info["features"], total_episodes=3)
    _corrupt_state_std(episodes_stats, ep_idx=0)

    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=150,
        info=info,
        episodes_stats=episodes_stats,
        episodes=[0, 1, 2],
    )
    assert np.max(dataset.meta.stats["state"]["std"]) > 1.0


def test_excluded_episodes_removes_corrupt_in_selection_from_norm_stats(
    tmp_path, info_factory, episodes_stats_factory, lerobot_dataset_factory
):
    """A corrupt episode inside ``episodes`` can be denylisted to clean the norm stats.

    This is the OpenTau-side mechanism for dropping corrupt-in-selection episodes
    without editing the data-side episode selection.
    """
    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1)
    episodes_stats = episodes_stats_factory(features=info["features"], total_episodes=3)
    _corrupt_state_std(episodes_stats, ep_idx=0)

    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=150,
        info=info,
        episodes_stats=episodes_stats,
        episodes=[0, 1, 2],
        excluded_episodes=[0],
    )
    assert dataset.episodes == [1, 2]
    assert np.allclose(dataset.meta.stats["state"]["std"], 0.25)


# --- shared selection helpers (also used by fit_fast_tokenizer / diagnose_norm_distribution) ---


def test_resolve_selected_episodes_precedence_and_none():
    from opentau.datasets.lerobot_dataset import resolve_selected_episodes

    # No selection at all -> None ("use everything").
    assert resolve_selected_episodes([0, 1, 2], None, None) is None
    # episodes only -> sorted copy.
    assert resolve_selected_episodes([0, 1, 2, 3], [3, 1], None) == [1, 3]
    # excluded_episodes only -> full set minus denylist.
    assert resolve_selected_episodes([0, 1, 2, 3], None, [1, 3]) == [0, 2]
    # both -> excluded wins on overlap.
    assert resolve_selected_episodes([0, 1, 2, 3], [0, 1, 2], [1]) == [0, 2]
    # excluding an index absent from the base is a no-op.
    assert resolve_selected_episodes([0, 1, 2], [0, 1], [99]) == [0, 1]
    # an empty denylist behaves like None (no-op), preserving the episodes arg.
    assert resolve_selected_episodes([0, 1, 2], [0, 1], []) == [0, 1]
    assert resolve_selected_episodes([0, 1, 2], None, []) is None


def test_resolve_selected_episodes_empty_raises():
    from opentau.datasets.lerobot_dataset import resolve_selected_episodes

    with pytest.raises(ValueError, match="excluded_episodes removed every episode"):
        resolve_selected_episodes([0, 1], [0, 1], [0, 1], repo_id="x/y")


class _StubMeta:
    """Minimal duck-typed stand-in for LeRobotDatasetMetadata."""

    def __init__(self, version: str, episodes_stats: dict, stats: dict, repo_id: str = "stub/repo"):
        self.repo_id = repo_id
        self.episodes_stats = episodes_stats
        self.episodes = list(episodes_stats)
        self.stats = stats
        self._version = packaging.version.parse(version)


def _state_stat(std: float, dim: int = 2) -> dict:
    return {
        "state": {
            "min": np.zeros(dim, dtype=np.float32),
            "max": np.ones(dim, dtype=np.float32),
            "mean": np.full(dim, 0.5, dtype=np.float32),
            "std": np.full(dim, float(std), dtype=np.float32),
            "count": np.array([10]),
        }
    }


def test_aggregate_selected_stats_v21_excludes_unselected():
    from opentau.datasets.lerobot_dataset import aggregate_selected_stats

    ep_stats = {0: _state_stat(1000.0), 1: _state_stat(0.25), 2: _state_stat(0.25)}
    meta = _StubMeta("v2.1", ep_stats, stats=_state_stat(999.0))  # global stats poisoned
    out = aggregate_selected_stats(meta, episodes=[1, 2])
    assert np.allclose(out["state"]["std"], 0.25)


def test_aggregate_selected_stats_no_selection_returns_meta_stats():
    from opentau.datasets.lerobot_dataset import aggregate_selected_stats

    sentinel = _state_stat(0.5)
    meta = _StubMeta("v2.1", {0: _state_stat(0.25)}, stats=sentinel)
    # No selection -> the global aggregate object is returned verbatim (no recompute).
    assert aggregate_selected_stats(meta, episodes=None, excluded_episodes=None) is sentinel


def test_aggregate_selected_stats_v20_falls_back_to_meta_stats():
    from opentau.datasets.lerobot_dataset import aggregate_selected_stats

    sentinel = _state_stat(0.5)
    # v2.0 has no usable per-episode stats; a selection cannot recompute -> fallback.
    meta = _StubMeta("v2.0", {0: _state_stat(1000.0), 1: _state_stat(0.25)}, stats=sentinel)
    assert aggregate_selected_stats(meta, episodes=[1]) is sentinel


def test_download_files_skips_present_files(tmp_path, lerobot_dataset_factory):
    """download_files must not call hf_hub_download for files already on disk.

    This is the core of the 429-avoidance fix: a pre-downloaded episode set
    should make download_files a no-op with zero Hub requests. Constructing
    the dataset already places every selected-episode file on disk, so a
    second download_files pass over the same paths must fetch nothing.
    """
    dataset = lerobot_dataset_factory(
        root=tmp_path / "test",
        repo_id=DUMMY_REPO_ID,
        total_episodes=10,
        total_frames=400,
        episodes=[2, 5, 6],
    )
    files = dataset.get_episodes_file_paths()
    assert files, "expected a non-empty file list for the test to be meaningful"
    assert all((dataset.root / f).is_file() for f in files), "fixture should pre-place all files"
    with patch("opentau.datasets.lerobot_dataset.hf_hub_download") as mock_hf_hub_download:
        dataset.download_files(files)
    mock_hf_hub_download.assert_not_called()


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
        ("real_action_dim", ()),
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
        elif key == "real_action_dim":
            assert item[key].shape == shape, f"{key}"
            assert item[key].dtype == torch.long, f"{key}"
            assert 0 < int(item[key].item()) <= train_pipeline_config.max_action_dim, f"{key}"

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
@retry_on_hf_flakiness()
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
@retry_on_hf_flakiness()
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


def test_skip_timestamp_check_bypasses_load_check(tmp_path, lerobot_dataset_factory):
    """`skip_timestamp_check=True` must suppress the load-time `check_timestamps_sync` call."""
    with patch("opentau.datasets.lerobot_dataset.check_timestamps_sync") as mock_check:
        dataset = lerobot_dataset_factory(
            root=tmp_path / "skip_check_test",
            skip_timestamp_check=True,
        )

    assert dataset.skip_timestamp_check is True
    # Only the load path is exercised here (no save_episode), so the call
    # count reflects whether the load-time gate was honored.
    assert mock_check.call_count == 0


def test_default_runs_load_check(tmp_path, lerobot_dataset_factory):
    """Without the flag, the load-time `check_timestamps_sync` runs exactly once."""
    with patch("opentau.datasets.lerobot_dataset.check_timestamps_sync") as mock_check:
        dataset = lerobot_dataset_factory(root=tmp_path / "default_check_test")

    assert dataset.skip_timestamp_check is False
    assert mock_check.call_count == 1


def _register_mapping(repo_id):
    """Temporarily install an empty entry into DATA_FEATURES_NAME_MAPPING.

    `factory.resolve_delta_timestamps` does a hard dict lookup on the mapping;
    callers that mock out `LeRobotDatasetMetadata` still need an entry to exist
    or they get a `KeyError`. Returns a no-arg cleanup callable.
    """
    from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

    sentinel = object()
    original = DATA_FEATURES_NAME_MAPPING.get(repo_id, sentinel)
    DATA_FEATURES_NAME_MAPPING[repo_id] = {}

    def _cleanup():
        if original is sentinel:
            DATA_FEATURES_NAME_MAPPING.pop(repo_id, None)
        else:
            DATA_FEATURES_NAME_MAPPING[repo_id] = original

    return _cleanup


def test_make_dataset_per_dataset_overrides_win(train_pipeline_config):
    """When set, `DatasetConfig.{tolerance_s,skip_timestamp_check}` win over mixture defaults."""
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        dataset_cfg.tolerance_s = 2e-3
        dataset_cfg.skip_timestamp_check = True
        train_pipeline_config.dataset_mixture.tolerance_s = 5e-4
        train_pipeline_config.dataset_mixture.skip_timestamp_check = False

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds_cls.return_value = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))

            make_dataset(dataset_cfg, train_pipeline_config)

        kwargs = mock_ds_cls.call_args.kwargs
        assert kwargs["tolerance_s"] == 2e-3
        assert kwargs["skip_timestamp_check"] is True
    finally:
        cleanup()


def test_make_dataset_inherits_mixture_defaults(train_pipeline_config):
    """When per-dataset overrides are None, the mixture-wide defaults flow through."""
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        # Per-dataset values left at None (the dataclass default).
        assert dataset_cfg.tolerance_s is None
        assert dataset_cfg.skip_timestamp_check is None
        train_pipeline_config.dataset_mixture.tolerance_s = 7e-3
        train_pipeline_config.dataset_mixture.skip_timestamp_check = True

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds_cls.return_value = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))

            make_dataset(dataset_cfg, train_pipeline_config)

        kwargs = mock_ds_cls.call_args.kwargs
        assert kwargs["tolerance_s"] == 7e-3
        assert kwargs["skip_timestamp_check"] is True
    finally:
        cleanup()


def test_make_dataset_per_dataset_skip_false_overrides_mixture_true(train_pipeline_config):
    """`DatasetConfig.skip_timestamp_check=False` must force the check on for one
    dataset even when the mixture default opts to skip — the override is true
    bidirectional, not a "set-only-when-True" sentinel.
    """
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        dataset_cfg.skip_timestamp_check = False
        train_pipeline_config.dataset_mixture.skip_timestamp_check = True

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds_cls.return_value = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))

            make_dataset(dataset_cfg, train_pipeline_config)

        assert mock_ds_cls.call_args.kwargs["skip_timestamp_check"] is False
    finally:
        cleanup()


def test_make_dataset_per_dataset_val_split_ratio_wins(train_pipeline_config):
    """A per-dataset `DatasetConfig.val_split_ratio` overrides the mixture default."""
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        dataset_cfg.val_split_ratio = 0.2
        train_pipeline_config.dataset_mixture.val_split_ratio = 0.05
        train_pipeline_config.val_freq = 1  # enable the train/val split branch

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))
            mock_ds.__len__.return_value = 100
            mock_ds.shallow_copy_with_dropout.return_value = mock_ds
            mock_ds_cls.return_value = mock_ds

            result = make_dataset(dataset_cfg, train_pipeline_config)

        assert isinstance(result, tuple)
        _, val_dataset = result
        # 0.2 (per-dataset) wins over 0.05 (mixture): int(100 * 0.2) == 20.
        assert len(val_dataset) == 20
    finally:
        cleanup()


def test_make_dataset_inherits_mixture_val_split_ratio(train_pipeline_config):
    """When the per-dataset `val_split_ratio` is None, the mixture default applies."""
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        assert dataset_cfg.val_split_ratio is None  # dataclass default (inherit)
        train_pipeline_config.dataset_mixture.val_split_ratio = 0.1
        train_pipeline_config.val_freq = 1

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))
            mock_ds.__len__.return_value = 100
            mock_ds.shallow_copy_with_dropout.return_value = mock_ds
            mock_ds_cls.return_value = mock_ds

            result = make_dataset(dataset_cfg, train_pipeline_config)

        assert isinstance(result, tuple)
        _, val_dataset = result
        # None (per-dataset) inherits mixture 0.1: int(100 * 0.1) == 10.
        assert len(val_dataset) == 10
    finally:
        cleanup()


def test_make_dataset_per_dataset_val_split_ratio_zero_opts_out(train_pipeline_config):
    """A per-dataset `val_split_ratio=0.0` opts that dataset out of validation.

    `make_dataset` still returns a `(train, val)` tuple (the branch is gated on
    `val_freq`, not the ratio), but the val half is empty and all samples stay
    in train. The empty val `Subset` is harmless in the val mixture: it carries
    no samples, and `WeightedDatasetMixture._calculate_sample_weights` skips
    length-0 members regardless of their weight (covered in
    `test_dataset_mixture.py::...skips_empty_member`).
    """
    dataset_cfg = train_pipeline_config.dataset_mixture.datasets[0]
    cleanup = _register_mapping(dataset_cfg.repo_id)
    try:
        dataset_cfg.val_split_ratio = 0.0  # opt this dataset out of validation
        train_pipeline_config.dataset_mixture.val_split_ratio = 0.05
        train_pipeline_config.val_freq = 1

        with (
            patch("opentau.datasets.factory.LeRobotDatasetMetadata") as mock_meta_cls,
            patch("opentau.datasets.factory.LeRobotDataset") as mock_ds_cls,
        ):
            mock_meta_cls.return_value = MagicMock(features=[])
            mock_ds = MagicMock(meta=MagicMock(info={}, stats={}, camera_keys=[]))
            mock_ds.__len__.return_value = 100
            mock_ds.shallow_copy_with_dropout.return_value = mock_ds
            mock_ds_cls.return_value = mock_ds

            result = make_dataset(dataset_cfg, train_pipeline_config)

        assert isinstance(result, tuple)
        train_dataset, val_dataset = result
        assert len(val_dataset) == 0  # int(100 * 0.0) == 0: empty val split
        assert len(train_dataset) == 100  # every sample remains for training
    finally:
        cleanup()


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


def test_skip_timestamp_warning_emitted_once_per_process(tmp_path, lerobot_dataset_factory, caplog):
    """`skip_timestamp_check=True` warns exactly once per process across multiple
    LeRobotDataset instances — locks in the `_SKIP_TIMESTAMP_WARNED` dedup so a
    heterogeneous mixture of N datasets emits 1 line, not N."""
    from opentau.datasets import lerobot_dataset as ld_mod

    original = ld_mod._SKIP_TIMESTAMP_WARNED
    ld_mod._SKIP_TIMESTAMP_WARNED = False
    try:
        with caplog.at_level(logging.WARNING):
            ds_a = lerobot_dataset_factory(
                root=tmp_path / "skip_warn_a",
                repo_id="warn-once/skip-ts-a",
                skip_timestamp_check=True,
            )
            ds_b = lerobot_dataset_factory(
                root=tmp_path / "skip_warn_b",
                repo_id="warn-once/skip-ts-b",
                skip_timestamp_check=True,
            )

        assert ds_a.skip_timestamp_check is True
        assert ds_b.skip_timestamp_check is True
        matching = [r for r in caplog.records if "skip_timestamp_check=True" in r.getMessage()]
        assert len(matching) == 1
    finally:
        ld_mod._SKIP_TIMESTAMP_WARNED = original


def test_robot_type_and_control_mode_in_meta_info(tmp_path, lerobot_dataset_factory, info_factory):
    """robot_type and control_mode are surfaced as optional fields in the
    standard data format. Verify the underlying meta/info.json values that
    BaseDataset._to_standard_data_format reads via ``self.meta.info.get(...)``.

    The unit tests in tests/datasets/test_optional_keys.py::TestRobotTypeControlMode
    cover the actual emission lines; this test just guards against a regression
    in how the factory threads info.json into ds.meta.info.
    """
    info_with = info_factory(
        robot_type="aloha",
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
    )
    info_with["control_mode"] = "joint"
    ds_with = lerobot_dataset_factory(root=tmp_path / "with-fields", info=info_with)
    assert ds_with.meta.info["robot_type"] == "aloha"
    assert ds_with.meta.info["control_mode"] == "joint"

    info_without = info_factory(
        robot_type=None,  # null robot_type is allowed by the type signature
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
    )  # no `control_mode` key — emulates pre-PR-#183 datasets
    ds_without = lerobot_dataset_factory(
        root=tmp_path / "without-fields",
        repo_id="missing-fields/test-repo",
        info=info_without,
    )
    assert ds_without.meta.info["robot_type"] is None
    assert "control_mode" not in ds_without.meta.info
