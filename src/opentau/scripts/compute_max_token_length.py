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

from lerobot.common.datasets.factory import make_dataset_mixture
from lerobot.common.datasets.lerobot_dataset import BaseDataset
from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


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
    if issubclass(policy_class, TAU0Policy):
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
