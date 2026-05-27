#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""``action_dim`` emission from LeRobotDataset._to_standard_data_format.

The dataset records the real (pre-pad) trailing dim of ``actions`` so
per-policy MSE on the velocity field can skip the zero-pad columns
under heterogeneous co-training.
"""

from dataclasses import dataclass

import pytest
import torch

from opentau.configs.policies import PreTrainedConfig
from opentau.datasets.factory import resolve_delta_timestamps
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from tests.fixtures.constants import DUMMY_REPO_ID


@dataclass
class DummyPolicyConfig(PreTrainedConfig):
    chunk_size: int = 50

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


@pytest.fixture(autouse=True)
def _fix_dummy_mapping():
    original = DATA_FEATURES_NAME_MAPPING.get(DUMMY_REPO_ID, {}).copy()
    DATA_FEATURES_NAME_MAPPING[DUMMY_REPO_ID] = {
        "state": "state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    }
    yield
    DATA_FEATURES_NAME_MAPPING[DUMMY_REPO_ID] = original


def _make_dataset(
    lerobot_dataset_factory,
    info_factory,
    hf_dataset_factory,
    tasks_factory,
    episodes_factory,
    stats_factory,
    episodes_stats_factory,
    tmp_path,
    action_dim,
    suffix,
):
    """Build a minimal LeRobotDataset whose underlying parquet matches a
    custom ``action_dim``. We have to explicitly thread the same custom
    features through both ``info_factory`` and ``hf_dataset_factory`` —
    the default lerobot_dataset_factory hf_dataset path ignores info["features"].
    """
    motor_features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"a{i}" for i in range(action_dim)],
        },
        "state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"s{i}" for i in range(action_dim)],
        },
    }
    info = info_factory(
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
        camera_features={},
        motor_features=motor_features,
    )
    tasks = tasks_factory(total_tasks=1)
    episode_dicts = episodes_factory(total_episodes=3, total_frames=150, tasks=tasks)
    hf_dataset = hf_dataset_factory(
        features=info["features"], tasks=tasks, episodes=episode_dicts, fps=info["fps"]
    )
    stats = stats_factory(features=info["features"])
    episodes_stats = episodes_stats_factory(features=info["features"], total_episodes=3)
    dataset = lerobot_dataset_factory(
        root=tmp_path / f"action_dim_{action_dim}_{suffix}",
        repo_id=DUMMY_REPO_ID,
        total_episodes=3,
        total_frames=150,
        total_tasks=1,
        info=info,
        stats=stats,
        episodes_stats=episodes_stats,
        tasks=tasks,
        episode_dicts=episode_dicts,
        hf_dataset=hf_dataset,
    )
    dataset.cfg.policy = DummyPolicyConfig(chunk_size=dataset.cfg.action_chunk)
    dt_info = resolve_delta_timestamps(dataset.cfg, dataset.cfg.dataset_mixture.datasets[0], dataset.meta)
    dataset.delta_timestamps_params = LeRobotDataset.compute_delta_params(*dt_info)
    return dataset


def test_action_dim_reflects_pre_pad_shape(
    lerobot_dataset_factory,
    info_factory,
    hf_dataset_factory,
    tasks_factory,
    episodes_factory,
    stats_factory,
    episodes_stats_factory,
    tmp_path,
):
    """A dataset with a 4-DoF action emits ``action_dim==4`` per sample,
    while the padded ``actions`` tensor is still ``max_action_dim`` wide."""
    dataset = _make_dataset(
        lerobot_dataset_factory,
        info_factory,
        hf_dataset_factory,
        tasks_factory,
        episodes_factory,
        stats_factory,
        episodes_stats_factory,
        tmp_path,
        action_dim=4,
        suffix="single",
    )
    item = dataset[25]
    assert "action_dim" in item, "dataset must emit action_dim alongside the padded actions tensor"
    assert item["action_dim"].shape == (), f"action_dim must be scalar (got {item['action_dim'].shape})"
    assert item["action_dim"].dtype == torch.long
    assert int(item["action_dim"].item()) == 4
    assert item["actions"].shape == (dataset.cfg.action_chunk, dataset.cfg.max_action_dim)


def test_action_dim_default_dataloader_collation(
    lerobot_dataset_factory,
    info_factory,
    hf_dataset_factory,
    tasks_factory,
    episodes_factory,
    stats_factory,
    episodes_stats_factory,
    tmp_path,
):
    """Default PyTorch collate stacks the scalar ``action_dim`` into a (B,) long tensor."""
    dataset = _make_dataset(
        lerobot_dataset_factory,
        info_factory,
        hf_dataset_factory,
        tasks_factory,
        episodes_factory,
        stats_factory,
        episodes_stats_factory,
        tmp_path,
        action_dim=3,
        suffix="collate",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert "action_dim" in batch
    assert batch["action_dim"].shape == (4,)
    assert batch["action_dim"].dtype == torch.long
    assert (batch["action_dim"] == 3).all()


def test_two_datasets_with_different_action_dims(
    lerobot_dataset_factory,
    info_factory,
    hf_dataset_factory,
    tasks_factory,
    episodes_factory,
    stats_factory,
    episodes_stats_factory,
    tmp_path,
):
    """Heterogeneous mixture: a 4-DoF and a 9-DoF dataset must each emit
    the correct per-sample ``action_dim`` (4 and 9 respectively), independent
    of the shared ``max_action_dim``."""
    common = {
        "lerobot_dataset_factory": lerobot_dataset_factory,
        "info_factory": info_factory,
        "hf_dataset_factory": hf_dataset_factory,
        "tasks_factory": tasks_factory,
        "episodes_factory": episodes_factory,
        "stats_factory": stats_factory,
        "episodes_stats_factory": episodes_stats_factory,
        "tmp_path": tmp_path,
    }
    ds_a = _make_dataset(**common, action_dim=4, suffix="a")
    ds_b = _make_dataset(**common, action_dim=9, suffix="b")
    item_a = ds_a[10]
    item_b = ds_b[10]
    assert int(item_a["action_dim"].item()) == 4
    assert int(item_b["action_dim"].item()) == 9
    assert item_a["actions"].shape[-1] == ds_a.cfg.max_action_dim
    assert item_b["actions"].shape[-1] == ds_b.cfg.max_action_dim
    assert item_a["actions"].shape[-1] == item_b["actions"].shape[-1]
