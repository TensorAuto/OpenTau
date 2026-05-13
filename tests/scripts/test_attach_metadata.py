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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from opentau.datasets.utils import EPISODES_PATH, INFO_PATH, load_jsonlines
from opentau.scripts.attach_metadata import (
    NEW_COLUMNS,
    _build_per_frame_columns,
    _load_and_validate_annotations,
    _update_episodes_jsonl,
    _update_info_json_features,
)
from tests.utils import retry_on_hf_flakiness

# Fixtures


def _minimal_meta(episode_lengths: dict[int, int]) -> SimpleNamespace:
    """A minimal stand-in for LeRobotDatasetMetadata for annotation validation."""
    return SimpleNamespace(
        episodes={ep: {"episode_index": ep, "length": length} for ep, length in episode_lengths.items()}
    )


@pytest.fixture()
def two_episode_meta() -> SimpleNamespace:
    return _minimal_meta({0: 50, 7: 30})


def _valid_annotations() -> list[dict]:
    return [
        {
            "episode_id": 0,
            "quality": 3,
            "segments": [
                {"start": 0, "subtask": "approach", "success": False},
                {"start": 20, "subtask": "grasp", "success": True},
            ],
        },
        {
            "episode_id": 7,
            "quality": 5,
            "segments": [{"start": 0, "subtask": "release", "success": True}],
        },
    ]


def _write_json(tmp_path: Path, payload) -> Path:
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# Validation tests


class TestValidation:
    def test_accepts_well_formed(self, tmp_path, two_episode_meta):
        ann = _load_and_validate_annotations(_write_json(tmp_path, _valid_annotations()), two_episode_meta)
        assert set(ann) == {0, 7}
        assert ann[0]["quality"] == 3
        assert len(ann[0]["segments"]) == 2

    def test_rejects_non_list_toplevel(self, tmp_path, two_episode_meta):
        with pytest.raises(ValueError, match="list"):
            _load_and_validate_annotations(_write_json(tmp_path, {"0": []}), two_episode_meta)

    def test_rejects_missing_episode(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()[:1]  # drop ep 7
        with pytest.raises(ValueError, match="Missing annotations"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_unknown_episode(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann.append({"episode_id": 999, "quality": 3, "segments": ann[0]["segments"]})
        with pytest.raises(ValueError, match="Unknown episodes"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_duplicate_episode(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann.append(dict(ann[0]))
        with pytest.raises(ValueError, match="Duplicate episode_id"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_empty_segments(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["segments"] = []
        with pytest.raises(ValueError, match="non-empty"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_nonzero_first_start(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["segments"][0]["start"] = 5
        with pytest.raises(ValueError, match="must start at 0"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_non_monotonic_starts(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        # [0, 20] -> [0, 20, 10] (non-monotonic third segment)
        ann[0]["segments"].append({"start": 10, "subtask": "x", "success": True})
        with pytest.raises(ValueError, match="strictly increase"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_start_past_end(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["segments"][1]["start"] = 50  # episode has length 50, valid range is 0..49
        with pytest.raises(ValueError, match=">= episode length"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_float_start(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["segments"][1]["start"] = 20.5
        with pytest.raises(ValueError, match="frame index, not a time"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_invalid_quality(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["quality"] = 6
        with pytest.raises(ValueError, match="quality must be in 1..5"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_nonbool_success(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        ann[0]["segments"][0]["success"] = "true"  # string, not bool
        with pytest.raises(ValueError, match="success must be bool"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)

    def test_rejects_missing_segment_key(self, tmp_path, two_episode_meta):
        ann = _valid_annotations()
        del ann[0]["segments"][0]["subtask"]
        with pytest.raises(ValueError, match="missing required key 'subtask'"):
            _load_and_validate_annotations(_write_json(tmp_path, ann), two_episode_meta)


# Per-frame column builders


class TestPerFrameColumns:
    def test_single_segment(self):
        segs = [{"start": 0, "subtask": "pick", "success": True, "memory": "M"}]
        r, m, k = _build_per_frame_columns(segs, ep_length=5, skip_memory=False)
        assert r == ["pick"] * 5
        assert m == ["M"] * 5
        assert k == [0] * 5

    def test_two_segments_half_and_half(self):
        segs = [
            {"start": 0, "subtask": "a", "success": False, "memory": "m0"},
            {"start": 5, "subtask": "b", "success": True, "memory": "m1"},
        ]
        r, m, k = _build_per_frame_columns(segs, ep_length=10, skip_memory=False)
        assert r == ["a"] * 5 + ["b"] * 5
        assert m == ["m0"] * 5 + ["m1"] * 5
        assert k == [1] * 5 + [0] * 5

    def test_three_uneven_segments(self):
        segs = [
            {"start": 0, "subtask": "a", "success": True, "memory": "m0"},
            {"start": 3, "subtask": "b", "success": False, "memory": "m1"},
            {"start": 7, "subtask": "c", "success": True, "memory": "m2"},
        ]
        r, m, k = _build_per_frame_columns(segs, ep_length=10, skip_memory=False)
        assert r == ["a", "a", "a", "b", "b", "b", "b", "c", "c", "c"]
        assert m == ["m0", "m0", "m0", "m1", "m1", "m1", "m1", "m2", "m2", "m2"]
        assert k == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

    def test_skip_memory_uses_placeholder(self):
        segs = [{"start": 0, "subtask": "s", "success": True}]
        _, m, _ = _build_per_frame_columns(segs, ep_length=4, skip_memory=True)
        assert m == ["memory0", "memory1", "memory2", "memory3"]


# info.json update


class TestInfoJsonUpdate:
    def _write_info(self, tmp_path: Path, features: dict) -> Path:
        root = tmp_path / "ds"
        (root / "meta").mkdir(parents=True)
        (root / INFO_PATH).write_text(json.dumps({"features": features}), encoding="utf-8")
        return root

    def test_adds_new_features(self, tmp_path):
        root = self._write_info(tmp_path, {"action": {"dtype": "float32", "shape": [7], "names": None}})
        _update_info_json_features(root)
        info = json.loads((root / INFO_PATH).read_text())
        for col in NEW_COLUMNS:
            assert col in info["features"]
            spec = info["features"][col]
            assert spec["shape"] == [1]
            assert spec["names"] is None
            assert spec["dtype"] in ("string", "int64")

    def test_idempotent(self, tmp_path):
        root = self._write_info(tmp_path, {})
        _update_info_json_features(root)
        first = (root / INFO_PATH).read_text()
        _update_info_json_features(root)
        assert (root / INFO_PATH).read_text() == first


# episodes.jsonl update


class TestEpisodesJsonlUpdate:
    def _write_episodes_jsonl(self, tmp_path: Path) -> Path:
        root = tmp_path / "ds"
        (root / "meta").mkdir(parents=True)
        (root / EPISODES_PATH).write_text(
            "\n".join(
                [
                    json.dumps({"episode_index": 0, "length": 50, "tasks": ["t0"]}),
                    json.dumps({"episode_index": 7, "length": 30, "tasks": ["t1"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return root

    def test_adds_quality_and_segments(self, tmp_path):
        root = self._write_episodes_jsonl(tmp_path)
        ann_by_ep = {
            0: {
                "quality": 3,
                "segments": [
                    {"start": 0, "subtask": "a", "success": False},
                    {"start": 20, "subtask": "b", "success": True},
                ],
            },
            7: {"quality": 5, "segments": [{"start": 0, "subtask": "c", "success": True}]},
        }
        _update_episodes_jsonl(root, ann_by_ep)
        records = load_jsonlines(root / EPISODES_PATH)
        by_ep = {r["episode_index"]: r for r in records}
        assert by_ep[0]["quality"] == 3
        assert by_ep[0]["segments"] == [0, 20]
        assert by_ep[0]["length"] == 50  # preserved
        assert by_ep[0]["tasks"] == ["t0"]  # preserved
        assert by_ep[7]["segments"] == [0]


# Integration (slow, real HF download of lerobot/droid_100 @ v2.1)


def _synthesize_annotations(episodes: dict[int, dict]) -> list[dict]:
    """Build a simple annotations payload covering every dataset episode.

    One or two segments per episode; quality alternates over 1..5 so tests see
    a range of values.
    """
    out = []
    for i, (ep_id, info) in enumerate(sorted(episodes.items())):
        length = int(info["length"])
        if length >= 4:
            half = length // 2
            segments = [
                {"start": 0, "subtask": f"ep{ep_id}_seg0", "success": False},
                {"start": half, "subtask": f"ep{ep_id}_seg1", "success": True},
            ]
        else:
            segments = [{"start": 0, "subtask": f"ep{ep_id}_seg0", "success": True}]
        out.append({"episode_id": ep_id, "quality": (i % 5) + 1, "segments": segments})
    return out


@pytest.mark.slow
@retry_on_hf_flakiness()
def test_attach_metadata_end_to_end_droid_100(tmp_path, dataset_config, train_pipeline_config):
    """Real-network integration test against lerobot/droid_100@v2.1.

    Uses the shared ``dataset_config`` / ``train_pipeline_config`` fixtures from
    ``tests/fixtures/config_factory.py`` so the loader sees a realistic
    delta-timestamps configuration (needed for ``_to_standard_data_format`` to
    emit ``action_is_pad``).
    """
    from opentau.datasets.factory import make_dataset

    # Make sure the mixture doesn't randomly mask optional keys while we assert shapes.
    mixture = train_pipeline_config.dataset_mixture
    for field in (
        "history_state_drop_prob",
        "subgoal_drop_prob",
        "subgoal_end_of_segment_prob",
        "response_drop_prob",
        "metadata_drop_all_prob",
        "metadata_drop_each_prob",
    ):
        setattr(mixture, field, 0.0)

    # --- Download the source dataset by constructing it via make_dataset. ---
    # Use the default HF cache root (same as the other droid_100 tests) rather
    # than a fresh tmp_path — writing into a fresh location triggers a cache
    # re-materialization that on some CI runners sees a different ``info.json``
    # fps and fails ``check_timestamps_sync``. We only ever READ from the source
    # root; the modified copy goes to ``tmp_path`` below.
    dataset_config.vqa = None
    dataset_config.repo_id = "lerobot/droid_100"
    dataset_config.root = None
    dataset_config.revision = "v2.1"
    dataset_config.episodes = None

    src_ds = make_dataset(dataset_config, train_pipeline_config)
    meta = src_ds.meta
    src_root = meta.root
    del src_ds

    # --- Build annotations and run the script. ---
    annotations = _synthesize_annotations(meta.episodes)
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")

    out_path = tmp_path / "droid_100_ann"

    from opentau.scripts.attach_metadata import attach_metadata

    attach_metadata(
        root=src_root,
        annotations_path=ann_path,
        copy_to=out_path,
        overwrite=False,
        skip_memory=True,
        model="gpt-4o-mini",
        delay_s=0.0,
    )

    # --- File-level assertions. ---
    info = json.loads((out_path / INFO_PATH).read_text())
    for col in NEW_COLUMNS:
        assert col in info["features"], col

    eps = load_jsonlines(out_path / EPISODES_PATH)
    for rec in eps:
        assert "quality" in rec
        assert "segments" in rec
        assert isinstance(rec["segments"], list)
        assert rec["segments"][0] == 0
        assert 1 <= rec["quality"] <= 5

    # episodes_stats.jsonl should be byte-identical to the source — we don't add
    # stats for the new columns.
    src_stats = (src_root / "meta/episodes_stats.jsonl").read_bytes()
    dst_stats = (out_path / "meta/episodes_stats.jsonl").read_bytes()
    assert src_stats == dst_stats

    # Parquet of the first episode: all new columns present; skip-memory
    # placeholder matches frame_index per row.
    first_ep = sorted(meta.episodes.keys())[0]
    rel = meta.get_data_file_path(first_ep)
    t = pq.read_table(out_path / rel)
    for col in NEW_COLUMNS:
        assert col in t.column_names, col
    frame_idx = t.column("frame_index").to_pylist()
    memory_col = t.column("memory").to_pylist()
    assert memory_col == [f"memory{i}" for i in frame_idx]

    # --- Load the annotated copy via make_dataset and verify optional keys. ---
    dataset_config.root = str(out_path)
    ds = make_dataset(dataset_config, train_pipeline_config)
    item = ds[0]

    # String keys use "" as the pad signal, so they have no companion flag.
    for k in ("memory", "next_memory"):
        assert k in item, f"missing {k}"
        assert f"{k}_is_pad" not in item
    for k in ("speed", "mistake", "quality"):
        assert k in item, f"missing {k}"
        assert f"{k}_is_pad" in item
    for k in range(ds.num_cams):
        assert f"subgoal{k}" in item
        assert item[f"subgoal{k}"].shape == (3, *ds.resolution)
    assert "subgoal_is_pad" in item
    speed_val = item["speed"].item()
    assert speed_val % 10 == 0
    assert 0 <= speed_val <= 100
    assert 1 <= item["quality"].item() <= 5
    # The memory column in the annotated dataset follows f"memory{frame_index}";
    # since __getitem__ already pulls the row, check memory_raw → memory.
    assert item["memory"].startswith("memory")
