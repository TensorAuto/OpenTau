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
"""Tests for config-time per-task prompt substitution.

Covers the ``_resolve_task`` unit behavior (always-swap, pass-through,
zero-RNG guarantee, determinism), the train/val gating through
``shallow_copy_with_dropout``, and the ``LeRobotDataset.__init__``
validation + end-to-end ``__getitem__`` integration via the dataset
fixtures.
"""

from types import SimpleNamespace

import pytest
import torch

import opentau.datasets.lerobot_dataset as _ld


def _make_dataset(substitutions_by_index: dict[int, list[str]], enable: bool = True):
    """Minimal LeRobotDataset carrying just the attrs ``_resolve_task`` reads."""
    ds = _ld.LeRobotDataset.__new__(_ld.LeRobotDataset)
    ds.enable_prompt_substitution = enable
    ds._prompt_substitutions_by_index = substitutions_by_index
    ds.meta = SimpleNamespace(tasks={0: "task zero", 1: "task one"})
    return ds


class TestResolveTask:
    def test_mapped_task_always_swaps(self):
        ds = _make_dataset({0: ["swapped"]})
        assert all(ds._resolve_task(0) == "swapped" for _ in range(20))

    def test_uniform_draws_cover_all_substitutes(self):
        """Every substitute appears; the original never does unless listed."""
        ds = _make_dataset({0: ["sub a", "sub b"]})
        torch.manual_seed(0)
        drawn = {ds._resolve_task(0) for _ in range(100)}
        assert drawn == {"sub a", "sub b"}

    def test_original_reappears_only_when_listed(self):
        ds = _make_dataset({0: ["task zero", "sub a"]})
        torch.manual_seed(0)
        drawn = {ds._resolve_task(0) for _ in range(100)}
        assert drawn == {"task zero", "sub a"}

    def test_unmapped_task_passes_through(self):
        ds = _make_dataset({0: ["swapped"]})
        assert ds._resolve_task(1) == "task one"

    @pytest.mark.parametrize(
        "substitutions, enable, task_idx",
        [
            ({}, True, 0),  # feature off (no mapping configured)
            ({0: ["swapped"]}, False, 0),  # gated off (val split)
            ({0: ["swapped"]}, True, 1),  # mapping present, this task misses
        ],
    )
    def test_no_rng_consumed_on_pass_through(self, substitutions, enable, task_idx):
        """Pins the determinism guarantee: pass-through paths must not touch
        the torch RNG stream, so configs without substitutions stay
        bit-identical to pre-feature runs."""
        ds = _make_dataset(substitutions, enable=enable)
        torch.manual_seed(0)
        state_before = torch.random.get_rng_state()
        ds._resolve_task(task_idx)
        assert torch.equal(state_before, torch.random.get_rng_state())

    def test_hit_is_deterministic_under_seed(self):
        ds = _make_dataset({0: ["sub a", "sub b", "sub c"]})
        torch.manual_seed(0)
        first = [ds._resolve_task(0) for _ in range(50)]
        torch.manual_seed(0)
        second = [ds._resolve_task(0) for _ in range(50)]
        assert first == second


class TestValGating:
    def test_val_clone_keeps_on_disk_prompt(self):
        ds = _make_dataset({0: ["swapped"]})
        clone = ds.shallow_copy_with_dropout(enable_dropout=False)
        assert clone._resolve_task(0) == "task zero"
        # The training instance is unaffected by the clone.
        assert ds._resolve_task(0) == "swapped"

    def test_val_opt_in_reenables_substitution(self):
        ds = _make_dataset({0: ["swapped"]})
        clone = ds.shallow_copy_with_dropout(enable_dropout=False, enable_prompt_substitution=True)
        assert clone._resolve_task(0) == "swapped"


class TestInitValidationAndGetItem:
    """Through the real ``__init__`` / ``__getitem__`` via the dataset
    fixtures (tasks are ``"Perform action {i}."``)."""

    def test_unknown_key_raises_with_near_miss(self, tmp_path, lerobot_dataset_factory):
        with pytest.raises(ValueError) as excinfo:
            lerobot_dataset_factory(
                root=tmp_path,
                total_tasks=1,
                prompt_substitutions={"Perform action 0.\n": ["anything"]},
            )
        msg = str(excinfo.value)
        assert "prompt_substitutions" in msg
        assert "Perform action 0.\\n" in msg  # repr() makes the whitespace visible
        assert "Whitespace near-misses" in msg
        assert "Perform action 0." in msg

    @pytest.mark.parametrize("bad_subs", [[], ["ok", ""]])
    def test_invalid_substitute_list_raises_at_init(self, tmp_path, lerobot_dataset_factory, bad_subs):
        """Direct constructions bypass DatasetConfig.__post_init__; init must
        still reject an empty list (would crash `torch.randint` at fetch time)
        and empty-string substitutes (would silently train matching samples on
        an empty prompt)."""
        with pytest.raises(ValueError, match="non-empty list of non-empty strings"):
            lerobot_dataset_factory(
                root=tmp_path,
                total_tasks=1,
                prompt_substitutions={"Perform action 0.": bad_subs},
            )

    def test_non_string_key_raises_designed_error(self, tmp_path, lerobot_dataset_factory):
        """A non-string key (e.g. keyed by task_index by mistake) must hit the
        designed ValueError, not an AttributeError inside the near-miss
        diagnostics."""
        with pytest.raises(ValueError, match="keys must be on-disk task strings"):
            lerobot_dataset_factory(
                root=tmp_path,
                total_tasks=1,
                prompt_substitutions={0: ["anything"]},
            )

    def test_duplicate_task_strings_map_every_index(self, tmp_path, lerobot_dataset_factory):
        """A v2.x tasks.jsonl may map the same string to several task_index
        values (`task_to_task_index` inverts last-wins); every duplicate index
        must substitute, not just the last one."""
        tasks = {
            0: {"task_index": 0, "task": "dup task"},
            1: {"task_index": 1, "task": "dup task"},
        }
        ds = lerobot_dataset_factory(
            root=tmp_path,
            tasks=tasks,
            standardize=False,
            prompt_substitutions={"dup task": ["swapped"]},
        )
        assert set(ds._prompt_substitutions_by_index) == {0, 1}
        assert ds._resolve_task(0) == "swapped"
        assert ds._resolve_task(1) == "swapped"

    @staticmethod
    def _stub_video_decode(monkeypatch):
        """The fixture declares video features but ships no mp4s; return
        zero frames so the real ``__getitem__`` can run end-to-end."""

        def _fake_query_videos(self, query_ts, ep_idx):
            return {key: torch.zeros((3, 8, 8)) for key in query_ts}

        monkeypatch.setattr(_ld.LeRobotDataset, "_query_videos", _fake_query_videos)

    def test_getitem_swaps_mapped_task(self, tmp_path, lerobot_dataset_factory, monkeypatch):
        self._stub_video_decode(monkeypatch)
        ds = lerobot_dataset_factory(
            root=tmp_path,
            total_tasks=1,
            standardize=False,
            prompt_substitutions={"Perform action 0.": ["Do the thing."]},
        )
        assert ds[0]["task"] == "Do the thing."

    def test_getitem_val_clone_keeps_on_disk_prompt(self, tmp_path, lerobot_dataset_factory, monkeypatch):
        self._stub_video_decode(monkeypatch)
        ds = lerobot_dataset_factory(
            root=tmp_path,
            total_tasks=1,
            standardize=False,
            prompt_substitutions={"Perform action 0.": ["Do the thing."]},
        )
        clone = ds.shallow_copy_with_dropout(enable_dropout=False)
        assert clone[0]["task"] == "Perform action 0."
        assert ds[0]["task"] == "Do the thing."
