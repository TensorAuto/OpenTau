#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the ``_TaggedDataset`` wrapper that injects per-sample dataset identity."""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from opentau.datasets.dataset_mixture import _TaggedDataset


class _FakeMeta:
    """Stand-in for ``BaseDataset.meta`` (any attribute object would do)."""


class _FakeDataset(Dataset):
    """Minimal stand-in for a ``BaseDataset`` for the wrapper's tagging path."""

    def __init__(self, length: int):
        self._length = length
        self.meta = _FakeMeta()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        return {
            "state": torch.tensor([float(idx)]),
            "action": torch.tensor([float(idx) * 2.0]),
        }


def test_tagged_dataset_injects_keys():
    """Each __getitem__ adds the expected `dataset_repo_id` and `dataset_index`."""
    base = _FakeDataset(length=3)
    tagged = _TaggedDataset(base, dataset_repo_id="foo/bar", dataset_index=2)
    for i in range(3):
        sample = tagged[i]
        assert sample["dataset_repo_id"] == "foo/bar"
        assert isinstance(sample["dataset_index"], torch.Tensor)
        assert sample["dataset_index"].dtype == torch.long
        assert int(sample["dataset_index"].item()) == 2
        # underlying keys preserved
        assert torch.equal(sample["state"], torch.tensor([float(i)]))


def test_tagged_dataset_preserves_meta():
    """`.meta` must round-trip through the wrapper (mixture validation relies on it)."""
    base = _FakeDataset(length=1)
    tagged = _TaggedDataset(base, dataset_repo_id="x", dataset_index=0)
    assert tagged.meta is base.meta


def test_tagged_dataset_len_delegates():
    base = _FakeDataset(length=7)
    tagged = _TaggedDataset(base, dataset_repo_id="x", dataset_index=0)
    assert len(tagged) == 7


def test_tagged_dataset_default_collate_batches_correctly():
    """Default collate must batch the str into a list[str] and the long scalar into (B,)."""
    base = _FakeDataset(length=4)
    tagged = _TaggedDataset(base, dataset_repo_id="foo/bar", dataset_index=3)
    samples = [tagged[i] for i in range(4)]
    batch = default_collate(samples)

    assert batch["dataset_repo_id"] == ["foo/bar"] * 4
    assert isinstance(batch["dataset_index"], torch.Tensor)
    assert batch["dataset_index"].dtype == torch.long
    assert tuple(batch["dataset_index"].shape) == (4,)
    assert torch.equal(batch["dataset_index"], torch.full((4,), 3, dtype=torch.long))


def test_tagged_dataset_dataloader_end_to_end():
    """Real DataLoader produces the right batched fields."""
    base = _FakeDataset(length=6)
    tagged = _TaggedDataset(base, dataset_repo_id="repo", dataset_index=1)
    loader = DataLoader(tagged, batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert batch["dataset_repo_id"] == ["repo"] * 3
        assert tuple(batch["dataset_index"].shape) == (3,)
        assert torch.equal(batch["dataset_index"], torch.ones(3, dtype=torch.long))
