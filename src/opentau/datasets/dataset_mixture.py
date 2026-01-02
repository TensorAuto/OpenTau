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
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.compute_stats import aggregate_stats
from opentau.datasets.lerobot_dataset import BaseDataset, DatasetMetadata
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING


def pad_vector(vector, new_dim):
    """Only the last dimension of the vector is padded to 'new_dim' with zeros."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype)
    new_vector[..., :current_dim] = vector
    return new_vector


class DatasetMixtureMetadata:
    """
    A class to hold metadata for a mixture of datasets.
    This is used to aggregate metadata from multiple datasets into a single object.
    """

    def __init__(
        self, cfg: TrainPipelineConfig, metadatas: List[DatasetMetadata], dataset_weights: List[float]
    ):
        self.cfg = cfg

        # convert each metadata stats to the standard data format
        for metadata in metadatas:
            metadata.stats = self._to_standard_data_format(metadata.repo_id, metadata.stats)

        self.stats = aggregate_stats([metadata.stats for metadata in metadatas], weights=dataset_weights)

    def _to_standard_data_format(
        self, repo_id: str, stats: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Converts stats to the standard data format."""
        name_map = DATA_FEATURES_NAME_MAPPING[repo_id]
        features_without_stats = ["prompt", "response", "advantage"]

        standard_stats = {}
        for new_key, key in name_map.items():
            if new_key in features_without_stats:
                # skip features that do not have stats
                continue

            # ensure only the first num_cams is used
            if new_key.startswith("camera"):
                cam_idx = int(new_key[len("camera") :])
                if cam_idx >= self.cfg.num_cams:
                    continue
            if key in stats:
                standard_stats[new_key] = stats[key]
            else:
                raise KeyError(f"Key '{key}' not found in stats. Available keys: {list(stats.keys())}")

        # pad state and action vectors
        for stat in standard_stats["state"]:
            if stat in ["mean", "std", "min", "max"]:
                standard_stats["state"][stat] = pad_vector(
                    standard_stats["state"][stat], self.cfg.max_state_dim
                )
                standard_stats["actions"][stat] = pad_vector(
                    standard_stats["actions"][stat], self.cfg.max_action_dim
                )

        # pad missing cameras
        for cam_idx in range(self.cfg.num_cams):
            if f"camera{cam_idx}" in standard_stats:
                continue
            standard_stats[f"camera{cam_idx}"] = {
                "min": np.zeros((3, 1, 1), dtype=np.float32),
                "max": np.ones((3, 1, 1), dtype=np.float32),
                "mean": np.zeros((3, 1, 1), dtype=np.float32),
                "std": np.zeros((3, 1, 1), dtype=np.float32),
                "count": np.array(
                    standard_stats["state"]["count"]
                ),  # create a copy in case this gets modified
            }

        # check for missing keys
        for data in standard_stats:
            missing_keys = {"mean", "std", "min", "max"} - standard_stats[data].keys()
            if missing_keys:
                raise KeyError(
                    f"The dataset {repo_id} is missing required statistics: {', '.join(sorted(missing_keys))}"
                )

        return standard_stats

    @property
    def features(self) -> dict[str, dict]:
        """Return standard data format"""
        features = {
            "state": {
                "shape": (self.cfg.max_state_dim,),
                "dtype": "float32",
            },
            "actions": {
                "shape": (self.cfg.max_action_dim,),
                "dtype": "float32",
            },
        }
        # add camera features
        for i in range(self.cfg.num_cams):
            features[f"camera{i}"] = {
                "shape": (3, self.cfg.resolution[0], self.cfg.resolution[1]),
                "dtype": "image",
            }
        return features


class HierarchicalSampler(Sampler[int]):
    r"""With-replacement sampler for a ConcatDataset that first samples a dataset according to `dataset_probs`, and then
    samples uniformly within that dataset. This avoids multinomial over a huge number of categories (over 2^24)
    by operating at the dataset level.
    """

    def __init__(
        self,
        dataset_lengths: List[int],
        dataset_probs: List[float],
        num_samples: int,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 262144,
    ):
        super().__init__()

        if len(dataset_lengths) != len(dataset_probs):
            raise ValueError("dataset_lengths and dataset_probs must have the same length.")
        self.num_samples = int(num_samples)
        self.chunk_size = int(chunk_size)

        lens = torch.as_tensor(dataset_lengths, dtype=torch.long)
        probs = torch.as_tensor(dataset_probs, dtype=torch.double)

        if (lens < 0).any():
            raise ValueError("dataset_lengths must be non-negative.")

        # Offsets for mapping local indices to global ConcatDataset indices
        self._full_offsets = torch.zeros(len(lens), dtype=torch.long)
        if len(lens) > 0:
            self._full_offsets[1:] = lens.cumsum(0)[:-1]

        # Keep only non-empty datasets with positive probability
        valid_mask = (lens > 0) & (probs > 0)
        if not bool(valid_mask.any()):
            raise ValueError("All datasets are empty or have zero probability.")

        self._valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
        self._valid_lens = lens[self._valid_ids]
        valid_probs = probs[self._valid_ids]
        self._valid_probs = (valid_probs / valid_probs.sum()).to(dtype=torch.double)

        self._num_valid = int(self._valid_ids.numel())
        self._gen = generator if generator is not None else torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        # Generate indices in memory-friendly chunks
        total = self.num_samples
        cs = self.chunk_size
        for start in range(0, total, cs):
            m = min(cs, total - start)

            # Choose dataset ids according to probs (over valid ids only)
            ds_choices_valid = torch.multinomial(self._valid_probs, m, replacement=True, generator=self._gen)

            # For each chosen dataset, draw uniform local indices and map to global indices
            out = torch.empty(m, dtype=torch.long)
            for k in range(self._num_valid):
                mask = ds_choices_valid == k
                k_count = int(mask.sum().item())
                if k_count == 0:
                    continue
                local_idx = torch.randint(0, int(self._valid_lens[k].item()), (k_count,), generator=self._gen)
                orig_ds_id = int(self._valid_ids[k].item())
                out[mask] = local_idx + self._full_offsets[orig_ds_id]

            # Yield one by one to conform to Sampler API
            for idx in out.tolist():
                yield int(idx)


class WeightedDatasetMixture:
    """
    A class to combine multiple PyTorch Datasets and create a DataLoader
    that samples from them according to specified weightings.
    """

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        datasets: List[BaseDataset],
        dataset_weights: List[float],
        action_freq: float,
    ):
        """
        Initializes the WeightedDatasetMixture.

        Args:
            cfg (TrainPipelineConfig): Configuration for the training pipeline.
            datasets (List[Dataset]): A list of PyTorch Dataset objects.
            dataset_weights (List[float]): A list of weights corresponding to each dataset.
                                          These determine the relative sampling frequency.
        """
        if not datasets:
            raise ValueError("The list of datasets cannot be empty.")
        if len(datasets) != len(dataset_weights):
            raise ValueError("The number of datasets must match the number of dataset_weights.")
        if any(w < 0 for w in dataset_weights):
            raise ValueError("Dataset weights must be non-negative.")
        if sum(dataset_weights) == 0 and any(len(ds) > 0 for ds in datasets):
            # If all weights are zero, but there's data, sampler will fail.
            # If all datasets are empty, sum of weights being zero is fine.
            logging.warning(
                "Warning: All dataset weights are zero. The sampler might not behave as expected if datasets have samples."
            )

        self.cfg = cfg
        self.datasets = datasets
        self.dataset_weights = dataset_weights
        self.action_freq = action_freq  # Frequency used for resampling action output
        self.dataset_names = [type(ds).__name__ + f"_{i}" for i, ds in enumerate(datasets)]  # For logging

        logging.info("Initializing WeightedDatasetMixture...")
        self._log_dataset_info()

        self.concatenated_dataset: ConcatDataset = ConcatDataset(datasets)
        logging.info(f"Total length of concatenated dataset: {len(self.concatenated_dataset)}")

        self.sample_weights: torch.Tensor = self._calculate_sample_weights()
        if self.sample_weights is None and len(self.concatenated_dataset) > 0:
            raise ValueError("Sample weights could not be calculated, but concatenated dataset is not empty.")
        elif self.sample_weights is not None and len(self.sample_weights) != len(self.concatenated_dataset):
            raise ValueError(
                f"Length of sample_weights ({len(self.sample_weights)}) "
                f"must match concatenated_dataset length ({len(self.concatenated_dataset)})."
            )
        logging.info("-" * 30)

        # aggregate metadata
        if not all(hasattr(ds, "meta") and ds.meta is not None for ds in datasets):
            raise ValueError("All datasets must have a 'meta' attribute with valid metadata.")
        self.meta = DatasetMixtureMetadata(cfg, [ds.meta for ds in datasets], dataset_weights)

    def _log_dataset_info(self):
        logging.info("Dataset information:")
        for i, ds in enumerate(self.datasets):
            logging.info(f"  - {self.dataset_names[i]}: Length={len(ds)}, Weight={self.dataset_weights[i]}")
        logging.info("-" * 30)

    def _calculate_sample_weights(self) -> Optional[torch.Tensor]:
        """
        Calculates the weight for each individual sample in the concatenated dataset.
        Samples from datasets with higher weights or smaller sizes (for a given weight)
        will have higher individual sample weights.
        """
        if not self.concatenated_dataset:  # Handles case where all input datasets are empty
            logging.warning("Warning: Concatenated dataset is empty. No sample weights to calculate.")
            return None

        logging.info("Calculating per-sample weights...")
        all_sample_weights: List[float] = []
        dataset_lengths = [len(ds) for ds in self.datasets]

        for i, length in enumerate(dataset_lengths):
            dataset_name = self.dataset_names[i]
            current_dataset_weight = self.dataset_weights[i]

            if length == 0:
                logging.info(f"  Skipping {dataset_name} (length 0).")
                continue  # Skip empty datasets

            if current_dataset_weight == 0:
                # Assign zero weight to all samples in this dataset
                weight_per_sample = 0.0
                logging.info(
                    f"  Weight for each sample in {dataset_name} (size {length}): {weight_per_sample:.10f} (dataset weight is 0)"
                )
            else:
                # Standard calculation: dataset_weight / num_samples_in_dataset
                weight_per_sample = current_dataset_weight / length
                logging.info(
                    f"  Weight for each sample in {dataset_name} (size {length}): {weight_per_sample:.10f}"
                )

            all_sample_weights.extend([weight_per_sample] * length)

        if not all_sample_weights:  # All datasets were empty or had 0 weight
            if len(self.concatenated_dataset) > 0:  # Should not happen if logic is correct
                raise RuntimeError(
                    "Mismatch: concatenated_dataset has samples but all_sample_weights is empty."
                )
            logging.warning(
                "Warning: All datasets are effectively empty or have zero weight. Sample weights list is empty."
            )
            return None  # No samples to weight

        return torch.DoubleTensor(all_sample_weights)

    def get_dataloader(self) -> DataLoader:
        """
        Creates and returns a PyTorch DataLoader with weighted sampling.

        Returns:
            DataLoader: A DataLoader instance configured for weighted sampling.
        """
        if len(self.concatenated_dataset) == 0:
            logging.warning("Warning: Concatenated dataset is empty. DataLoader will produce no batches.")
            # Return an empty dataloader or raise error, depending on desired behavior.
            # For now, let it create an empty dataloader.
            return DataLoader(
                self.concatenated_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers
            )

        # Validate there is at least one non-empty dataset with positive weight
        if not any(len(ds) > 0 and w > 0 for ds, w in zip(self.datasets, self.dataset_weights, strict=True)):
            logging.error("Error: No non-empty dataset has a positive sampling weight.")
            raise ValueError("No non-empty dataset has a positive sampling weight.")

        num_samples_per_epoch = len(self.concatenated_dataset)
        logging.info("\nCreating DataLoader...")
        logging.info(f"  Batch size: {self.cfg.batch_size}")
        logging.info(f"  Samples per epoch (num_samples for sampler): {num_samples_per_epoch}")

        # Hierarchical sampling: choose dataset by weight, then uniform within it (both with replacement)
        ds_lengths = [len(ds) for ds in self.datasets]
        sampler = HierarchicalSampler(
            dataset_lengths=ds_lengths,
            dataset_probs=self.dataset_weights,
            num_samples=num_samples_per_epoch,
        )

        dataloader = DataLoader(
            self.concatenated_dataset,
            batch_size=self.cfg.dataloader_batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            prefetch_factor=self.cfg.prefetch_factor,
        )
        logging.info("DataLoader created successfully.")
        logging.info("-" * 30)
        return dataloader
