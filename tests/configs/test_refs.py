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
from pathlib import Path

import pytest

from opentau.configs.refs import (
    REF_KEY,
    RefError,
    resolve_refs,
    resolve_refs_to_tempfile,
)


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_no_refs_passes_through(tmp_path):
    src = _write(tmp_path / "a.json", {"x": 1, "y": [1, 2, 3]})
    assert resolve_refs(src) == {"x": 1, "y": [1, 2, 3]}


def test_basic_ref_inlines_object(tmp_path):
    inner = _write(tmp_path / "inner.json", {"a": 1, "b": 2})
    outer = _write(tmp_path / "outer.json", {"nested": {REF_KEY: "inner.json"}})
    assert resolve_refs(outer) == {"nested": {"a": 1, "b": 2}}
    # Source files must not be mutated.
    assert json.loads(inner.read_text()) == {"a": 1, "b": 2}
    assert json.loads(outer.read_text()) == {"nested": {REF_KEY: "inner.json"}}


def test_ref_to_array(tmp_path):
    _write(tmp_path / "list.json", [{"id": 1}, {"id": 2}])
    outer = _write(tmp_path / "outer.json", {"items": {REF_KEY: "list.json"}})
    assert resolve_refs(outer) == {"items": [{"id": 1}, {"id": 2}]}


def test_ref_inside_list(tmp_path):
    _write(tmp_path / "d1.json", {"name": "first"})
    _write(tmp_path / "d2.json", {"name": "second"})
    outer = _write(
        tmp_path / "outer.json",
        {"datasets": [{REF_KEY: "d1.json"}, {REF_KEY: "d2.json"}, {"name": "third"}]},
    )
    assert resolve_refs(outer) == {"datasets": [{"name": "first"}, {"name": "second"}, {"name": "third"}]}


def test_sibling_keys_override_via_deep_merge(tmp_path):
    _write(tmp_path / "base.json", {"a": 1, "b": {"c": 2, "d": 3}})
    outer = _write(
        tmp_path / "outer.json",
        {REF_KEY: "base.json", "b": {"c": 99}, "extra": "kept"},
    )
    # Sibling `b.c` overrides; sibling `b.d` is not touched; sibling-only `extra` is added.
    assert resolve_refs(outer) == {"a": 1, "b": {"c": 99, "d": 3}, "extra": "kept"}


def test_sibling_list_replaces_base_list(tmp_path):
    _write(tmp_path / "base.json", {"items": [1, 2, 3]})
    outer = _write(tmp_path / "outer.json", {REF_KEY: "base.json", "items": [9]})
    assert resolve_refs(outer) == {"items": [9]}


def test_chained_refs(tmp_path):
    _write(tmp_path / "leaf.json", {"value": "leaf"})
    _write(tmp_path / "mid.json", {"wrapper": {REF_KEY: "leaf.json"}})
    outer = _write(tmp_path / "outer.json", {REF_KEY: "mid.json"})
    assert resolve_refs(outer) == {"wrapper": {"value": "leaf"}}


def test_relative_paths_resolve_against_including_file(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    # `sub/leaf.json` references `../shared.json` — must resolve relative to `sub/`.
    _write(tmp_path / "shared.json", {"shared": True})
    _write(sub / "leaf.json", {REF_KEY: "../shared.json"})
    outer = _write(tmp_path / "outer.json", {"x": {REF_KEY: "sub/leaf.json"}})
    assert resolve_refs(outer) == {"x": {"shared": True}}


def test_cycle_raises(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, {REF_KEY: "b.json"})
    _write(b, {REF_KEY: "a.json"})
    with pytest.raises(RefError, match="Cyclic"):
        resolve_refs(a)


def test_self_ref_raises(tmp_path):
    a = _write(tmp_path / "a.json", {REF_KEY: "a.json"})
    with pytest.raises(RefError, match="Cyclic"):
        resolve_refs(a)


def test_missing_ref_target_raises(tmp_path):
    outer = _write(tmp_path / "outer.json", {REF_KEY: "does_not_exist.json"})
    with pytest.raises(RefError, match="not found"):
        resolve_refs(outer)


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    outer = _write(tmp_path / "outer.json", {REF_KEY: "bad.json"})
    with pytest.raises(RefError, match="Invalid JSON"):
        resolve_refs(outer)


def test_non_string_ref_value_raises(tmp_path):
    outer = _write(tmp_path / "outer.json", {REF_KEY: 42})
    with pytest.raises(RefError, match="must be a string"):
        resolve_refs(outer)


def test_sibling_keys_with_non_object_target_raises(tmp_path):
    _write(tmp_path / "list.json", [1, 2, 3])
    outer = _write(
        tmp_path / "outer.json",
        {"items": {REF_KEY: "list.json", "extra": "not_allowed"}},
    )
    with pytest.raises(RefError, match="non-object content"):
        resolve_refs(outer)


def test_diamond_share_does_not_trigger_cycle(tmp_path):
    # Two separate refs to the same file are not a cycle.
    _write(tmp_path / "shared.json", {"v": 1})
    outer = _write(
        tmp_path / "outer.json",
        {"a": {REF_KEY: "shared.json"}, "b": {REF_KEY: "shared.json"}},
    )
    assert resolve_refs(outer) == {"a": {"v": 1}, "b": {"v": 1}}


def test_resolve_refs_to_tempfile_writes_resolved(tmp_path):
    _write(tmp_path / "inner.json", {"a": 1})
    outer = _write(tmp_path / "outer.json", {"x": {REF_KEY: "inner.json"}})
    out = resolve_refs_to_tempfile(outer)
    try:
        assert out.is_file()
        assert json.loads(out.read_text()) == {"x": {"a": 1}}
    finally:
        out.unlink(missing_ok=True)


def test_dataset_mixture_through_load_stripped(tmp_path):
    """End-to-end: an outer config that splits its dataset_mixture into a ref
    is correctly assembled by load_stripped_config_to_tempfile."""
    from opentau.configs.policies import load_stripped_config_to_tempfile

    _write(
        tmp_path / "mixture.json",
        {"datasets": [{"repo_id": "lerobot/droid_100"}], "weights": [1.0]},
    )
    outer = _write(
        tmp_path / "train.json",
        {
            "batch_size": 32,
            "dataset_mixture": {REF_KEY: "mixture.json"},
        },
    )
    tmp = load_stripped_config_to_tempfile(outer)
    try:
        merged = json.loads(tmp.read_text())
    finally:
        tmp.unlink(missing_ok=True)
    assert merged["batch_size"] == 32
    assert merged["dataset_mixture"]["datasets"] == [{"repo_id": "lerobot/droid_100"}]
    assert merged["dataset_mixture"]["weights"] == [1.0]


def test_train_pipeline_config_from_pretrained_with_ref(tmp_path):
    """End-to-end: TrainPipelineConfig.from_pretrained loads a config that
    factors out its dataset_mixture into a separate file via $ref."""
    from opentau.configs.train import TrainPipelineConfig

    artifact = Path("tests/artifacts/configs/train_config.json")
    full = json.loads(artifact.read_text())

    # Split the dataset_mixture into its own file and replace it with a $ref.
    mixture = full.pop("dataset_mixture")
    _write(tmp_path / "mixture.json", mixture)
    full["dataset_mixture"] = {REF_KEY: "mixture.json"}
    config_path = _write(tmp_path / "train_config.json", full)

    cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=config_path)
    # Round-tripping through $ref must reproduce the same dataset list.
    assert [d.repo_id or d.vqa for d in cfg.dataset_mixture.datasets] == [
        "lerobot/droid_100",
        "lerobot/aloha_mobile_cabinet",
        "dummy",
    ]
    assert list(cfg.dataset_mixture.weights) == [1.0, 1.0, 1.0]


def _decompose_into_refs(full: dict, sections: tuple[str, ...], out_dir: Path) -> dict:
    """Split each top-level ``section`` of ``full`` into its own file under
    ``out_dir`` and replace it in ``full`` with a ``$ref``."""
    decomposed = dict(full)
    for section in sections:
        _write(out_dir / f"{section}.json", full[section])
        decomposed[section] = {REF_KEY: f"{section}.json"}
    return decomposed


def test_train_config_round_trip_matches_baseline(tmp_path):
    """Regression: decomposing a real train config across multiple sub-files
    via $ref must yield a TrainPipelineConfig structurally identical to the
    one loaded from the original monolithic JSON.

    Covers the sections most likely to bloat a real config (datasets, policy,
    optimizer, scheduler, wandb), so the round-trip exercises ref resolution
    against the full draccus type system — not just a contrived dict.
    """
    from opentau.configs.train import TrainPipelineConfig

    artifact = Path("tests/artifacts/configs/train_config.json")
    full = json.loads(artifact.read_text())

    # Baseline: load the original monolithic config.
    baseline_path = _write(tmp_path / "baseline.json", full)
    baseline_cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=baseline_path)

    # Decomposed: split several large sections into their own files.
    decomposed_dir = tmp_path / "decomposed"
    decomposed_dir.mkdir()
    decomposed = _decompose_into_refs(
        full,
        sections=("dataset_mixture", "policy", "optimizer", "scheduler", "wandb"),
        out_dir=decomposed_dir,
    )
    decomposed_path = _write(decomposed_dir / "train_config.json", decomposed)
    decomposed_cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=decomposed_path)

    # Compare via draccus.encode — the canonical serialized form. Equality at
    # this level catches any field that drifted, not just spot-checks.
    import draccus

    assert draccus.encode(decomposed_cfg) == draccus.encode(baseline_cfg)


def test_train_config_chained_refs_match_baseline(tmp_path):
    """Regression: a $ref chain (parent → mid → leaf) for a real sub-config
    resolves to the same value as inlining the leaf directly."""
    from opentau.configs.train import TrainPipelineConfig

    artifact = Path("tests/artifacts/configs/train_config.json")
    full = json.loads(artifact.read_text())

    baseline_path = _write(tmp_path / "baseline.json", full)
    baseline_cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=baseline_path)

    # Build a chain: train_config.json → mixture_ref.json → mixture_leaf.json
    chain_dir = tmp_path / "chain"
    chain_dir.mkdir()
    _write(chain_dir / "mixture_leaf.json", full["dataset_mixture"])
    _write(chain_dir / "mixture_ref.json", {REF_KEY: "mixture_leaf.json"})
    chained = dict(full)
    chained["dataset_mixture"] = {REF_KEY: "mixture_ref.json"}
    chained_path = _write(chain_dir / "train_config.json", chained)

    chained_cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=chained_path)

    import draccus

    assert draccus.encode(chained_cfg) == draccus.encode(baseline_cfg)


def test_train_config_sibling_override_applies(tmp_path):
    """Regression: a sibling key alongside a $ref must override the matching
    field in the loaded content (deep-merge), not be silently ignored."""
    from opentau.configs.train import TrainPipelineConfig

    artifact = Path("tests/artifacts/configs/train_config.json")
    full = json.loads(artifact.read_text())

    # The artifact has weights = [1.0, 1.0, 1.0]; override the first weight to 7.0.
    _write(tmp_path / "mixture.json", full["dataset_mixture"])
    full["dataset_mixture"] = {REF_KEY: "mixture.json", "weights": [7.0, 1.0, 1.0]}
    config_path = _write(tmp_path / "train_config.json", full)

    cfg = TrainPipelineConfig.from_pretrained(pretrained_name_or_path=config_path)
    assert list(cfg.dataset_mixture.weights) == [7.0, 1.0, 1.0]
    # The non-overridden fields from the referenced file must still be present.
    assert [d.repo_id or d.vqa for d in cfg.dataset_mixture.datasets] == [
        "lerobot/droid_100",
        "lerobot/aloha_mobile_cabinet",
        "dummy",
    ]
