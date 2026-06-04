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
"""Read-only support for LeRobot v3.0 ("file consolidation") datasets.

v3.0 packs many episodes into a single parquet / mp4 instead of one file per
episode; the per-episode file + row/timestamp mapping lives in
``meta/episodes/**/*.parquet`` and tasks move to ``meta/tasks.parquet``. These
tests synthesize a tiny real on-disk v3.0 dataset (no Hub, no GPU, no real
codec) and check that metadata loads, ``episode_data_index`` stays aligned for
full and subset loads, ``__getitem__`` returns the right rows, and consolidated
video frames are queried at the correct (offset) timestamps.
"""

from pathlib import Path

import numpy as np
import packaging.version
import pandas as pd
import pytest
import torch
from huggingface_hub.errors import RevisionNotFoundError

from opentau.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from opentau.datasets.utils import (
    DEFAULT_FEATURES,
    V30_DATA_PATH,
    V30_EPISODES_PATH,
    V30_TASKS_PATH,
    V30_VIDEO_PATH,
    check_version_compatibility,
    flatten_dict,
    get_safe_version,
    load_episodes_stats_v30,
    load_episodes_v30,
    load_tasks_v30,
    write_json,
)

FPS = 10
STATE_DIM = 4
ACTION_DIM = 2
VIDEO_KEY = "observation.images.camera0"
DEFAULT_LENGTHS = (4, 3, 5)  # ep0+ep1 -> file-000, ep2 -> file-001


def _features(use_videos: bool) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    if use_videos:
        features[VIDEO_KEY] = {
            "dtype": "video",
            "shape": (3, 16, 16),
            "names": ["channels", "height", "width"],
            "video_info": {"video.fps": float(FPS), "video.codec": "h264"},
        }
    features.update({k: dict(v) for k, v in DEFAULT_FEATURES.items()})
    return features


# Features that get per-episode statistics. Kept to the 1-D vector modalities so
# the synthetic stats round-trip through parquet without nested-shape fuss; the
# loader path under test (unflatten + cast_stats_to_numpy + aggregate_stats) is
# identical regardless of how many features are present.
_STAT_FEATURES = {"observation.state": STATE_DIM, "action": ACTION_DIM}


def _episode_stats(length: int, use_videos: bool) -> dict:
    """Per-episode stats dict shaped like ``load_episodes_stats`` output."""
    stats = {}
    for feat, dim in _STAT_FEATURES.items():
        stats[feat] = {
            "min": np.zeros(dim, dtype=np.float32),
            "max": np.ones(dim, dtype=np.float32),
            "mean": np.full(dim, 0.5, dtype=np.float32),
            "std": np.full(dim, 0.25, dtype=np.float32),
            "count": np.array([length], dtype=np.int64),
        }
    if use_videos:
        # Image/video stats are (C,1,1). After a parquet round-trip they surface
        # as a (3,) object array of nested per-channel arrays, which must be
        # restored to (3,1,1) for aggregate_stats — the exact shape DROID hit.
        stats[VIDEO_KEY] = {
            "min": np.zeros((3, 1, 1), dtype=np.float32),
            "max": np.ones((3, 1, 1), dtype=np.float32),
            "mean": np.full((3, 1, 1), 0.5, dtype=np.float32),
            "std": np.full((3, 1, 1), 0.25, dtype=np.float32),
            "count": np.array([length], dtype=np.int64),
        }
    return stats


def _state_row(ep: int, frame: int) -> list[float]:
    # state[0] encodes (episode, frame) so any row mix-up is unambiguous.
    return [float(ep * 1000 + frame), 0.0, 0.0, 0.0]


def _action_row(ep: int, frame: int) -> list[float]:
    return [float(ep * 100 + frame), 0.0]


def _write_v30_dataset(
    root: Path,
    *,
    lengths=DEFAULT_LENGTHS,
    use_videos: bool = True,
    multi_task: bool = False,
    data_file_of_ep=None,
) -> None:
    """Write a complete, loadable v3.0 dataset under ``root``.

    ``data_file_of_ep`` maps episode_index -> (chunk_index, file_index) for the
    consolidated data parquet; default packs ep0,ep1 into file-000 and ep2 into
    file-001 to exercise the multi-file + subset paths. All episodes share one
    consolidated mp4 per camera (chunk-000/file-000).
    """
    root = Path(root)
    n_eps = len(lengths)
    if data_file_of_ep is None:
        data_file_of_ep = {ep: (0, 0 if ep < 2 else 1) for ep in range(n_eps)}

    features = _features(use_videos)
    tasks = ["Perform action 0.", "Perform action 1."] if multi_task else ["Perform action 0."]

    # --- per-frame data rows, grouped by their consolidated data file ----------
    files_rows: dict[tuple[int, int], list[dict]] = {}
    ep_meta_rows = []
    global_index = 0
    video_cursor = 0.0
    for ep in range(n_eps):
        length = lengths[ep]
        task_index = ep % len(tasks)
        dataset_from = global_index
        for frame in range(length):
            files_rows.setdefault(data_file_of_ep[ep], []).append(
                {
                    "observation.state": _state_row(ep, frame),
                    "action": _action_row(ep, frame),
                    "timestamp": np.float32(frame / FPS),
                    "frame_index": np.int64(frame),
                    "episode_index": np.int64(ep),
                    "index": np.int64(global_index),
                    "task_index": np.int64(task_index),
                }
            )
            global_index += 1
        dataset_to = global_index

        ep_chunk, ep_file = data_file_of_ep[ep]
        row = {
            "episode_index": np.int64(ep),
            "tasks": [tasks[task_index]],
            "length": np.int64(length),
            "data/chunk_index": np.int64(ep_chunk),
            "data/file_index": np.int64(ep_file),
            "dataset_from_index": np.int64(dataset_from),
            "dataset_to_index": np.int64(dataset_to),
        }
        if use_videos:
            ep_dur = length / FPS
            row.update(
                {
                    f"videos/{VIDEO_KEY}/chunk_index": np.int64(0),
                    f"videos/{VIDEO_KEY}/file_index": np.int64(0),
                    f"videos/{VIDEO_KEY}/from_timestamp": np.float64(video_cursor),
                    f"videos/{VIDEO_KEY}/to_timestamp": np.float64(video_cursor + ep_dur),
                }
            )
            video_cursor += ep_dur
        # flatten per-episode stats into stats/{feature}/{stat} columns, as
        # (possibly nested) Python lists so pandas/pyarrow stores them as list
        # columns — mirroring the real v3.0 converter (HF datasets), including
        # the nested (C,1,1) image stats that a parquet round-trip flattens.
        stat_flat = flatten_dict({"stats": _episode_stats(length, use_videos)})
        row.update({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stat_flat.items()})
        ep_meta_rows.append(row)

    # --- write consolidated data parquet shards --------------------------------
    for (chunk_idx, file_idx), rows in files_rows.items():
        fpath = root / V30_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(fpath, index=False)

    # --- meta/episodes/chunk-000/file-000.parquet ------------------------------
    ep_path = root / V30_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_meta_rows).to_parquet(ep_path, index=False)

    # --- meta/tasks.parquet (indexed by the task string, like upstream) --------
    tasks_path = root / V30_TASKS_PATH
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"task_index": list(range(len(tasks)))},
        index=pd.Index(tasks, name="task"),
    ).to_parquet(tasks_path)

    # --- meta/stats.json (aggregate; all episodes identical here) --------------
    agg = _episode_stats(sum(lengths), use_videos)
    write_json({k: {s: v.tolist() for s, v in fv.items()} for k, fv in agg.items()}, root / "meta/stats.json")

    # --- meta/info.json (v3.0; omits total_chunks/total_videos like upstream) --
    info = {
        "codebase_version": "v3.0",
        "robot_type": "dummy",
        "total_episodes": n_eps,
        "total_frames": sum(lengths),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": V30_DATA_PATH,
        "video_path": V30_VIDEO_PATH if use_videos else None,
        "features": features,
    }
    write_json(info, root / "meta/info.json")

    # Touch the consolidated mp4 so the on-disk file-existence assert passes;
    # actual decoding is monkeypatched in the video test.
    if use_videos:
        vpath = root / V30_VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        vpath.touch()


def _make_cfg(root: Path, repo_id: str, episodes):
    """Minimal TrainPipelineConfig, mirroring tests/fixtures/dataset_factories."""
    from dataclasses import dataclass

    from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
    from opentau.configs.policies import PreTrainedConfig
    from opentau.configs.train import TrainPipelineConfig

    @dataclass
    class DummyPolicyConfig(PreTrainedConfig):
        @property
        def observation_delta_indices(self):
            return None

        @property
        def action_delta_indices(self):
            return None

        @property
        def reward_delta_indices(self):
            return None

        def get_optimizer_preset(self):
            return None

        def get_scheduler_preset(self):
            return None

        def validate_features(self):
            pass

    dataset_cfg = DatasetConfig(repo_id=repo_id, root=str(root), episodes=episodes)
    mixture_cfg = DatasetMixtureConfig(
        datasets=[dataset_cfg],
        weights=[1.0],
        history_state_drop_prob=0.0,
        subgoal_drop_prob=0.0,
        subgoal_end_of_segment_prob=0.0,
        response_drop_prob=0.0,
        metadata_drop_all_prob=0.0,
        metadata_drop_each_prob=0.0,
    )
    return TrainPipelineConfig(dataset_mixture=mixture_cfg, policy=DummyPolicyConfig(), batch_size=8)


def _make_dataset(root: Path, *, episodes=None, repo_id="dummy/v30", **kwargs) -> LeRobotDataset:
    cfg = _make_cfg(root, repo_id, episodes)
    return LeRobotDataset(
        cfg=cfg,
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        standardize=False,
        **kwargs,
    )


def _episode_index_column(dataset: LeRobotDataset) -> np.ndarray:
    """Compacted (indices-map-respecting) episode_index column of the loaded rows.

    NOTE: do not use ``hf_dataset.data.table`` here — for a v3.0 subset load the
    rows are an ``.select()`` index-map over the full table, so the raw table is
    not physically compacted. Column access goes through the index map.
    """
    return np.asarray(dataset.hf_dataset.with_format("numpy")["episode_index"]).reshape(-1)


def _assert_row_alignment(dataset: LeRobotDataset) -> None:
    ep_col = _episode_index_column(dataset)
    for ep in dataset.episodes:
        pos = dataset.epi2idx[ep]
        start = int(dataset.episode_data_index["from"][pos])
        end = int(dataset.episode_data_index["to"][pos])
        assert end > start
        assert all(int(e) == ep for e in ep_col[start:end]), (
            f"episode {ep} rows [{start}:{end}] carry mismatched episode_index"
        )


# --------------------------------------------------------------------------- #
# Metadata loaders + version gating
# --------------------------------------------------------------------------- #
def test_v30_metadata_loads(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root)
    meta = LeRobotDatasetMetadata(repo_id="dummy/v30", root=root)

    assert meta._is_v30
    assert meta.total_episodes == 3
    assert meta.total_frames == sum(DEFAULT_LENGTHS)
    assert set(meta.episodes) == {0, 1, 2}
    assert meta.episodes[1]["length"] == DEFAULT_LENGTHS[1]
    assert meta.tasks[0] == "Perform action 0."
    # per-episode stats reconstructed from the flattened stats/* columns
    assert meta.episodes_stats[0]["observation.state"]["mean"].shape == (STATE_DIM,)
    np.testing.assert_allclose(meta.episodes_stats[2]["action"]["max"], np.ones(ACTION_DIM))
    # image/video stats must round-trip from parquet as (3,1,1) (not flattened to
    # (3,)) so aggregate_stats/_assert_type_and_shape accept them — the DROID bug.
    assert meta.episodes_stats[0][VIDEO_KEY]["min"].shape == (3, 1, 1)
    assert meta.episodes_stats[0][VIDEO_KEY]["count"].shape == (1,)
    # aggregated stats from meta/stats.json
    assert meta.stats["observation.state"]["mean"].shape == (STATE_DIM,)


def test_v30_path_accessors_use_consolidated_files(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root)
    meta = LeRobotDatasetMetadata(repo_id="dummy/v30", root=root)

    # ep0 and ep1 share file-000; ep2 is in file-001.
    assert meta.get_data_file_path(0) == Path("data/chunk-000/file-000.parquet")
    assert meta.get_data_file_path(1) == Path("data/chunk-000/file-000.parquet")
    assert meta.get_data_file_path(2) == Path("data/chunk-000/file-001.parquet")
    # all episodes share one consolidated mp4
    assert meta.get_video_file_path(2, VIDEO_KEY) == Path(f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4")


def test_v30_loaders_standalone(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, multi_task=True)
    tasks, task_to_idx = load_tasks_v30(root)
    assert tasks == {0: "Perform action 0.", 1: "Perform action 1."}
    assert task_to_idx["Perform action 1."] == 1

    episodes = load_episodes_v30(root)
    assert episodes[0]["dataset_from_index"] == 0
    assert episodes[0]["dataset_to_index"] == DEFAULT_LENGTHS[0]
    assert episodes[2]["data/file_index"] == 1

    stats = load_episodes_stats_v30(root)
    assert set(stats) == {0, 1, 2}
    assert stats[1]["observation.state"]["std"].shape == (STATE_DIM,)


def test_deep_float_array_coerces_object_nested_image_stat():
    # Reproduce the exact structure a real v3.0 image stat round-trips to from
    # parquet: a (3,) object array of (1,) object arrays of (1,) float arrays
    # (DROID). A single .tolist() leaves object dtype, breaking np.isfinite in
    # aggregate_stats; _deep_float_array must walk it down to clean float.
    from opentau.datasets.utils import _deep_float_array

    def _channel(v):
        return np.array([np.array([v], dtype=np.float64)], dtype=object)

    val = np.array([_channel(0.1), _channel(0.2), _channel(0.3)], dtype=object)
    out = _deep_float_array(val)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.reshape(-1), [0.1, 0.2, 0.3])


def test_check_version_compatibility_does_not_warn_on_v30(caplog):
    # The v2.0 global-stats warning must not fire for a v3.0 (different major)
    # dataset; v3.0 carries its own per-episode stats.
    import logging

    with caplog.at_level(logging.WARNING):
        check_version_compatibility("dummy/v30", "v3.0", "v2.1")
    assert "global stats" not in caplog.text.lower()


def test_get_safe_version_resolves_v30_only_repo(monkeypatch):
    # A repo published only as v3.0 must resolve (not ForwardCompatibilityError)
    # under the default v2.1 target, thanks to the read ceiling.
    monkeypatch.setattr(
        "opentau.datasets.utils.get_repo_versions", lambda repo_id: [packaging.version.parse("v3.0")]
    )
    assert get_safe_version("dummy/v30", "v2.1") == "v3.0"


def test_get_safe_version_v21_repo_unchanged(monkeypatch):
    monkeypatch.setattr(
        "opentau.datasets.utils.get_repo_versions", lambda repo_id: [packaging.version.parse("v2.1")]
    )
    assert get_safe_version("dummy/v21", "v2.1") == "v2.1"


def test_get_safe_version_untagged_repo_falls_back_to_main(monkeypatch):
    # An untagged repo (no version refs) resolves to the default branch when the
    # caller did not pin a revision, instead of raising RevisionNotFoundError.
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["main", "parquet"])
    assert get_safe_version("dummy/untagged", "v2.1", allow_branch_fallback=True) == "main"


def test_get_safe_version_untagged_repo_falls_back_to_master(monkeypatch):
    # With no `main`, the next default branch (`master`) is used.
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["master"])
    assert get_safe_version("dummy/untagged", "v2.1", allow_branch_fallback=True) == "master"


def test_get_safe_version_prefers_main_over_master(monkeypatch):
    # Order matters: `main` wins even when `master` is listed first.
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["master", "main"])
    assert get_safe_version("dummy/untagged", "v2.1", allow_branch_fallback=True) == "main"


def test_get_safe_version_untagged_repo_raises_without_fallback(monkeypatch):
    # Default behaviour (an explicitly requested revision): still hard-fails.
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["main"])
    with pytest.raises(RevisionNotFoundError):
        get_safe_version("dummy/untagged", "v2.1")


def test_get_safe_version_raises_when_no_default_branch(monkeypatch):
    # Fallback allowed but neither `main` nor `master` exists -> still raises.
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["dev"])
    with pytest.raises(RevisionNotFoundError):
        get_safe_version("dummy/untagged", "v2.1", allow_branch_fallback=True)


def test_metadata_untagged_repo_defaults_to_main(tmp_path, monkeypatch):
    # End-to-end: an untagged repo with no local cache resolves its revision to
    # the default branch (revision unset) and loads, rather than hitting the
    # strict get_safe_version error path. pull_from_repo is faked to materialize
    # the metadata the second load reads.
    root = tmp_path / "ds"
    monkeypatch.setattr("opentau.datasets.lerobot_dataset.get_proc_accelerator", lambda: None)
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["main"])

    def _fake_pull(self, allow_patterns=None, ignore_patterns=None):
        _write_v30_dataset(root)

    monkeypatch.setattr(LeRobotDatasetMetadata, "pull_from_repo", _fake_pull)
    meta = LeRobotDatasetMetadata(repo_id="dummy/untagged", root=root)  # revision unset
    assert meta.revision == "main"
    assert meta._is_v30


def test_metadata_untagged_repo_explicit_revision_still_raises(tmp_path, monkeypatch):
    # An explicitly pinned revision keeps the strict behaviour (no branch fallback).
    root = tmp_path / "ds"
    monkeypatch.setattr("opentau.datasets.lerobot_dataset.get_proc_accelerator", lambda: None)
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["main"])
    with pytest.raises(RevisionNotFoundError):
        LeRobotDatasetMetadata(repo_id="dummy/untagged", root=root, revision="v2.1")


def test_dataset_untagged_repo_loads_via_branch_fallback(tmp_path, monkeypatch):
    # Full LeRobotDataset path (the primary use case): an untagged repo with no
    # local cache must load by falling back to the default branch, not raise at
    # metadata construction. Regression test for forwarding the *original*
    # (uncoerced) revision from LeRobotDataset to its metadata constructor — with
    # the coerced `self.revision` the fallback was dead and this raised.
    root = tmp_path / "ds"
    monkeypatch.setattr("opentau.datasets.lerobot_dataset.get_proc_accelerator", lambda: None)
    monkeypatch.setattr("opentau.datasets.utils.get_repo_versions", lambda repo_id: [])
    monkeypatch.setattr("opentau.datasets.utils.get_repo_branches", lambda repo_id: ["main"])

    def _fake_pull(self, allow_patterns=None, ignore_patterns=None):
        _write_v30_dataset(root, use_videos=False)

    monkeypatch.setattr(LeRobotDatasetMetadata, "pull_from_repo", _fake_pull)
    dataset = _make_dataset(root)  # repo_id="dummy/v30", revision unset
    assert dataset.meta.revision == "main"
    assert dataset.episodes == [0, 1, 2]
    assert len(dataset) == sum(DEFAULT_LENGTHS)


# --------------------------------------------------------------------------- #
# Full LeRobotDataset: alignment + __getitem__ + video offset
# --------------------------------------------------------------------------- #
def test_v30_full_load_alignment(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=False)
    dataset = _make_dataset(root)

    assert dataset.episodes == [0, 1, 2]
    assert len(dataset) == sum(DEFAULT_LENGTHS)
    # episode_data_index matches the authored dataset_from/to (== length cumsum)
    assert dataset.episode_data_index["from"].tolist() == [0, 4, 7]
    assert dataset.episode_data_index["to"].tolist() == [4, 7, 12]
    _assert_row_alignment(dataset)


def test_v30_getitem_returns_aligned_rows(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=False)
    dataset = _make_dataset(root)

    # global row 7 is episode 2, frame 0  -> state[0] == 2*1000 + 0
    item = dataset[7]
    assert int(item["episode_index"].item()) == 2
    assert float(item["observation.state"][0].item()) == pytest.approx(2000.0)
    # last row of episode 1 (global row 6): frame 2 of ep1 -> 1*1000 + 2
    item = dataset[6]
    assert int(item["episode_index"].item()) == 1
    assert float(item["observation.state"][0].item()) == pytest.approx(1002.0)


def test_v30_subset_load_alignment(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=False)
    # episodes 0 and 2 share files with the unselected episode 1 (subset path).
    dataset = _make_dataset(root, episodes=[0, 2])

    assert dataset.episodes == [0, 2]
    assert len(dataset) == DEFAULT_LENGTHS[0] + DEFAULT_LENGTHS[2]
    # compacted: ep0 -> [0,4), ep2 -> [4,9)
    assert dataset.episode_data_index["from"].tolist() == [0, 4]
    assert dataset.episode_data_index["to"].tolist() == [4, 9]
    assert set(_episode_index_column(dataset).tolist()) == {0, 2}
    _assert_row_alignment(dataset)

    # compacted row 4 == episode 2, frame 0
    item = dataset[4]
    assert int(item["episode_index"].item()) == 2
    assert float(item["observation.state"][0].item()) == pytest.approx(2000.0)


def test_v30_video_query_uses_from_timestamp_offset(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=True)

    captured = []

    def _fake_decode(video_path, timestamps, tolerance_s, backend):
        captured.append((str(video_path), list(np.asarray(timestamps, dtype=np.float64))))
        return torch.zeros((len(timestamps), 3, 16, 16))

    monkeypatch.setattr("opentau.datasets.lerobot_dataset.decode_video_frames", _fake_decode)

    dataset = _make_dataset(root)  # nearest resample (default) -> one decode call

    # episode 0, frame 0: offset 0.0 -> queried at t=0.0
    captured.clear()
    _ = dataset[0]
    assert captured[-1][0].endswith(f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4")
    assert captured[-1][1] == pytest.approx([0.0])

    # episode 2, frame 0 (global row 7): cumulative offset 0.7s -> queried at 0.7
    captured.clear()
    _ = dataset[7]
    assert captured[-1][1] == pytest.approx([0.7])


def test_v30_no_video_dataset_loads(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=False)
    dataset = _make_dataset(root)
    assert dataset.meta.video_keys == []
    item = dataset[0]
    assert VIDEO_KEY not in item
    assert int(item["episode_index"].item()) == 0


def test_v30_multi_task_scalar_task_lookup(tmp_path):
    root = tmp_path / "ds"
    _write_v30_dataset(root, use_videos=False, multi_task=True)
    dataset = _make_dataset(root)
    # episode 1 -> task_index 1 -> "Perform action 1."
    item = dataset[4]  # global row 4 == episode 1, frame 0
    assert int(item["episode_index"].item()) == 1
    assert item["task"] == "Perform action 1."
