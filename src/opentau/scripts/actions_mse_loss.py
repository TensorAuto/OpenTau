#!/usr/bin/env python

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

import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pprint import pformat
from typing import Any

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.action_indexing import subtract_chunk_start_state
from opentau.datasets.factory import make_dataset_mixture
from opentau.policies.candidates import refuse_candidates
from opentau.policies.factory import get_policy_class
from opentau.policies.utils import maybe_compile_sample_actions, to_dtype_preserving_siglip_float32
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    init_logging,
)


def dataset_delta_action_state_map(dataset: Dataset) -> dict[int, int] | None:
    """The post-index delta map ``dataset`` applies to its targets, or ``None`` for absolute ones.

    ``WeightedDatasetMixture`` wraps every entry in ``_TaggedDataset``, and a validation split adds
    a ``Subset`` beneath it; neither proxies attribute access.
    """
    ds = getattr(dataset, "_base", dataset)
    delta_map = getattr(ds, "delta_action_state_map", None)
    if delta_map is None and hasattr(ds, "dataset"):
        delta_map = getattr(ds.dataset, "delta_action_state_map", None)
    return delta_map or None


def first_action_pairs(
    sample_actions: Callable[[dict[str, Any]], torch.Tensor],
    dataloader: Iterable[dict[str, Any]],
    device: torch.device,
    delta_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and ground-truth first action of every chunk, each ``(N, action_dim)``.

    They are compared in the space the dataset's targets live in. A delta-action dataset stores
    each chunk relative to its chunk-start state, while the policy returns absolute targets (it
    adds that state back), so the prediction is re-expressed against the same batch state rather
    than the targets rebuilt as absolute. The re-anchoring state then cancels exactly — the
    dataset emits it rounded to bfloat16, so rebuilt "absolute" targets would not be the recorded
    ones — and R² is measured against the deltas' own variance; in absolute space it is dominated
    by the state the policy is given, and reads near 1 for almost any model.

    Args:
        sample_actions: The policy's ``sample_actions`` (possibly compiled).
        dataloader: Batches from one dataset of the mixture.
        device: Device the policy runs on.
        delta_map: That dataset's post-index ``delta_action_state_map``, or ``None``.

    Returns:
        ``(predicted, ground_truth)`` first actions, one row per sample, over the pre-pad dims.
    """
    pred, truth = [], []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        actions = sample_actions(batch).to(torch.float32)
        if delta_map:
            actions = subtract_chunk_start_state(actions, batch["state"], delta_map)
        # The dataset's emitted (post-index, pre-pad) width; the zero-padded tail is no robot dim.
        dof = int(batch["real_action_dim"][0])
        pred.append(actions[:, 0, :dof].cpu().numpy())
        truth.append(batch["actions"][:, 0, :dof].to(torch.float32).cpu().numpy())
    return np.concatenate(pred), np.concatenate(truth)


def action_fidelity(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension MSE and R² of ``pred`` against ``truth`` (both ``(N, action_dim)``)."""
    return np.mean((pred - truth) ** 2, axis=0), r2_score(truth, pred, multioutput="raw_values")


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # Unlike the other entry points this one compiles `policy.sample_actions` — the *policy*
    # level, not the inner sampler — so an armed critic would be traced into the graph.
    refuse_candidates(
        cfg,
        reason="this script torch.compiles the policy-level sample_actions, which would pull "
        "the critic and the candidate selection inside the traced region; the data-dependent "
        "argmax would graph-break or bake in a candidate count. It also measures fidelity "
        "against ground-truth actions, where a best-of-N pick is not the quantity being "
        "measured. Run it with policy.n_candidates=1.",
    )
    # build lerobot dataset and dataloader
    datasets = make_dataset_mixture(cfg)

    # load the trained or fine-tuned model (any batch size: every sample's first action is scored)

    device = auto_torch_device()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    # Preserve the float32-pinned SigLIP embeddings across the bf16 cast (openpi parity) — this
    # script measures action fidelity, so it must run the model as served, not a re-rounded one.
    policy = to_dtype_preserving_siglip_float32(policy, device=device, dtype=torch.bfloat16)
    policy.eval()
    policy_sample_actions = maybe_compile_sample_actions(policy, policy.sample_actions, device_hint=device)

    # Always reset policy before episode to clear out action cache.
    policy.reset()

    for dataset in datasets.datasets:
        print(f"The batch size is {cfg.batch_size}")
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size)
        delta_map = dataset_delta_action_state_map(dataset)
        if delta_map:
            print("delta-action targets: comparing in delta space (prediction minus chunk-start state)")

        with torch.inference_mode():
            pred, truth = first_action_pairs(policy_sample_actions, dataloader, device, delta_map)
        mse, r2 = action_fidelity(pred, truth)

        print(f"the mean squared error loss per dimension is {mse}")

        print(f"the r2 score per dimension is {r2}")
    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
