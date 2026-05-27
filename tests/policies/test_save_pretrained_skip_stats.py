#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""End-to-end test for `PreTrainedConfig.save_normalization_stats` + the method override."""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pretrained import NORM_BUFFER_PREFIXES, is_norm_buffer_key


class _FakePolicyConfig:
    """Minimal stand-in for PreTrainedConfig fields read by _save_pretrained.

    The full PreTrainedConfig is abstract (requires choice-type registration);
    constructing a real subclass would couple this test to a specific policy
    (pi0, pi05, etc.) which is overkill. We only need the flag this test
    exercises plus a no-op `_save_pretrained` for `config.json`.
    """

    def __init__(self, save_normalization_stats: bool):
        self.save_normalization_stats = save_normalization_stats
        self.dataset_names = ["fake"]

    def _save_pretrained(self, save_directory: Path) -> None:
        # No-op: this test only inspects the safetensors layout, not config.json.
        (save_directory / "config.json").write_text("{}")


class _FakePolicy(torch.nn.Module):
    """Bare-bones policy with a Normalize + Unnormalize module so the
    save path covers `normalize_inputs.buffer_*` / `unnormalize_outputs.buffer_*`.
    """

    def __init__(self, save_normalization_stats: bool):
        super().__init__()
        self.config = _FakePolicyConfig(save_normalization_stats)
        features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        }
        out_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        }
        norm_map = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
        per_dataset_stats = [
            {
                "observation.state": {
                    "mean": torch.zeros(4),
                    "std": torch.ones(4),
                },
                "action": {
                    "mean": torch.zeros(3),
                    "std": torch.ones(3),
                },
            }
        ]
        self.normalize_inputs = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)
        self.unnormalize_outputs = Unnormalize(out_features, norm_map, per_dataset_stats=per_dataset_stats)
        # Add one regular trainable param so the safetensors file is non-empty
        # in the stripped case too.
        self.linear = torch.nn.Linear(4, 3)


def _save_pretrained(policy: _FakePolicy, save_dir: Path, *, include_norm_stats=None) -> None:
    """Mirror `PreTrainedPolicy._save_pretrained` behaviour."""
    from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
    from safetensors.torch import save_file
    from safetensors.torch import save_model as save_model_as_safetensor

    policy.config._save_pretrained(save_dir)
    if include_norm_stats is None:
        include_norm_stats = policy.config.save_normalization_stats
    out_path = str(save_dir / SAFETENSORS_SINGLE_FILE)
    if include_norm_stats:
        save_model_as_safetensor(policy, out_path)
    else:
        sd = policy.state_dict()
        filtered = {k: v for k, v in sd.items() if not is_norm_buffer_key(k)}
        save_file(filtered, out_path)


def _file_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt") as f:
        return set(f.keys())


def test_save_with_stats_keeps_norm_buffers(tmp_path: Path):
    policy = _FakePolicy(save_normalization_stats=True)
    _save_pretrained(policy, tmp_path)
    keys = _file_keys(tmp_path / "model.safetensors")
    # Expect at least one normalize_inputs.buffer_* key in the safetensors.
    assert any(k.startswith("normalize_inputs.buffer_") for k in keys), keys
    assert any(k.startswith("unnormalize_outputs.buffer_") for k in keys), keys


def test_save_without_stats_strips_norm_buffers(tmp_path: Path):
    policy = _FakePolicy(save_normalization_stats=False)
    _save_pretrained(policy, tmp_path)
    keys = _file_keys(tmp_path / "model.safetensors")
    for prefix in NORM_BUFFER_PREFIXES:
        assert not any(k.startswith(prefix) for k in keys), (prefix, keys)
    # The non-norm parameters (e.g. the regular linear layer) are still saved.
    assert any(k.startswith("linear.") for k in keys), keys


def test_save_override_wins_over_config(tmp_path: Path):
    """include_norm_stats=False overrides save_normalization_stats=True (and vice versa)."""
    policy_keep = _FakePolicy(save_normalization_stats=True)
    override_false_dir = tmp_path / "override_false"
    override_false_dir.mkdir()
    _save_pretrained(policy_keep, override_false_dir, include_norm_stats=False)
    keys_override = _file_keys(override_false_dir / "model.safetensors")
    assert not any(k.startswith("normalize_inputs.buffer_") for k in keys_override)

    policy_strip = _FakePolicy(save_normalization_stats=False)
    override_true_dir = tmp_path / "override_true"
    override_true_dir.mkdir()
    _save_pretrained(policy_strip, override_true_dir, include_norm_stats=True)
    keys_override = _file_keys(override_true_dir / "model.safetensors")
    assert any(k.startswith("normalize_inputs.buffer_") for k in keys_override)


def test_is_norm_buffer_key_matches_known_prefixes():
    for prefix in NORM_BUFFER_PREFIXES:
        assert is_norm_buffer_key(prefix + "observation_state.mean")
    assert not is_norm_buffer_key("model.transformer.weight")
    assert not is_norm_buffer_key("linear.weight")


@pytest.mark.parametrize("strip", [False, True])
def test_state_dict_round_trip(tmp_path: Path, strip: bool):
    """Save then load (manually) — kept params survive, normalize params survive iff !strip."""
    from safetensors.torch import load_file

    policy = _FakePolicy(save_normalization_stats=not strip)
    _save_pretrained(policy, tmp_path)
    loaded = load_file(str(tmp_path / "model.safetensors"))

    # linear's params present either way
    assert "linear.weight" in loaded
    assert "linear.bias" in loaded

    has_norm = any(k.startswith("normalize_inputs.buffer_") for k in loaded)
    if strip:
        assert not has_norm
    else:
        assert has_norm
