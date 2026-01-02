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
import math
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.policies.factory import get_policy_class
from opentau.policies.pi0.modeling_pi0 import PI0Policy
from opentau.policies.pi05.modeling_pi05 import PI05Policy


@dataclass
class Args:
    target_cfg: str  # path to the training configuration file
    keys: tuple[str] = (
        "response",
        "prompt",
    )  # keys to compute max token length for, e.g. ["response", "prompt"]
    num_workers: int | None = None
    chunk_size: int = 1000
    output_path: str | None = None


def get_tokenizer(cfg: TrainPipelineConfig) -> callable:
    r"""Returns a tokenizer function based on the policy type in the configuration."""
    policy_class = get_policy_class(cfg.policy.type)

    # TODO: Add `elif` for other policy types if needed
    if issubclass(policy_class, PI0Policy) or issubclass(policy_class, PI05Policy):
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    raise ValueError(f"Unsupported policy type: {cfg.policy.type}")


def chunked(dataset: BaseDataset, key: str, chunk_size: int):
    n = len(dataset)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        yield [dataset[i][key] for i in range(start, end)]


def worker_fn(chunk, tokenizer: PreTrainedTokenizer):
    return Counter(len(tokenizer(s)["input_ids"]) for s in chunk)


def to_percentile(counter: Counter) -> dict[int, float]:
    r"""Convert counter to a dictionary with token lengths as keys and their percentile as values."""
    total = counter.total()
    sorted_keys = sorted(counter.keys())
    values = accumulate(counter[k] for k in sorted_keys)
    return {k: v / total for k, v in reversed(list(zip(sorted_keys, values, strict=False)))}


@parser.wrap()
def main(args: Args):
    cfg = TrainPipelineConfig.from_pretrained(args.target_cfg)
    datasets = make_dataset_mixture(cfg).datasets
    tokenizer = get_tokenizer(cfg)
    worker = partial(worker_fn, tokenizer=tokenizer)

    output = {}
    for key in args.keys:
        counter = Counter()
        for ds in datasets:
            tasks = tqdm(
                chunked(ds, key, args.chunk_size),
                total=math.ceil(len(ds) / args.chunk_size),
                desc=f"Processing {key} in {ds._get_feature_mapping_key()}",
            )
            # TODO: multiprocessing doesn't seem to speed things up. debug why.
            with Pool(args.num_workers or cpu_count()) as pool:
                parts = pool.imap_unordered(worker, tasks)
                counter = sum(parts, start=counter)

        output[key] = to_percentile(counter)

    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
