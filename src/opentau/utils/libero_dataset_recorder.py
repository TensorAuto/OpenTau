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

"""Utilities for recording and converting LIBERO datasets.

This module provides functions for converting LIBERO episode data into the standard
dataset format, including consolidation of task results, aggregation of multiple
results, and generation of dataset metadata.
"""

import datetime
import io
import json
import logging
import math
import shutil
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

LIBERO_TASKS = [
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove and put the moka pot on it",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both moka pots on the stove",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "pick up the book and place it in the back compartment of the caddy",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
    "open the top drawer and put the bowl inside",
    "put the cream cheese in the bowl",
    "put the wine bottle on top of the cabinet",
    "push the plate to the front of the stove",
    "turn on the stove",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "open the middle drawer of the cabinet",
    "pick up the orange juice and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]

LIBERO_TASK_TO_IDX = {task: idx for idx, task in enumerate(LIBERO_TASKS)}
LIBERO_FPS = 10
LIBERO_CHUNK_SIZE = 1000
LIBERO_DEFAULT_STATS = {
    "image": {
        "mean": [
            [[0.48068472743034363]],
            [[0.4485854208469391]],
            [[0.4106878638267517]],
        ],
        "std": [
            [[0.229267880320549]],
            [[0.22267605364322662]],
            [[0.21524138748645782]],
        ],
        "max": [
            [[1.0]],
            [[1.0]],
            [[1.0]],
        ],
        "min": [
            [[0.0]],
            [[0.0]],
            [[0.0]],
        ],
    },
    "wrist_image": {
        "mean": [
            [[0.5057708024978638]],
            [[0.46449801325798035]],
            [[0.42354270815849304]],
        ],
        "std": [
            [[0.2672027349472046]],
            [[0.25637131929397583]],
            [[0.24345873296260834]],
        ],
        "max": [
            [[1.0]],
            [[1.0]],
            [[1.0]],
        ],
        "min": [
            [[0.0]],
            [[0.0]],
            [[0.0]],
        ],
    },
    "state": {
        "mean": [
            -0.04651879519224167,
            0.03440921753644943,
            0.7645525336265564,
            2.972202777862549,
            -0.22047005593776703,
            -0.1255796253681183,
            0.026914266869425774,
            -0.02719070389866829,
        ],
        "std": [
            0.10494378954172134,
            0.15176637470722198,
            0.3785160183906555,
            0.34427398443222046,
            0.9069469571113586,
            0.3253920078277588,
            0.014175914227962494,
            0.014058894477784634,
        ],
        "max": [
            0.21031762659549713,
            0.39128610491752625,
            1.3660105466842651,
            3.6714255809783936,
            3.560650587081909,
            1.386339545249939,
            0.04233968257904053,
            0.0013633022317662835,
        ],
        "min": [
            -0.4828203022480011,
            -0.3255046010017395,
            0.008128180168569088,
            0.35277295112609863,
            -3.641430377960205,
            -1.842738389968872,
            -0.0013586411951109767,
            -0.042040832340717316,
        ],
    },
    "actions": {
        "mean": [
            0.06278137117624283,
            0.0868409126996994,
            -0.09037282317876816,
            0.0005407406715676188,
            0.005643361248075962,
            -0.005229088477790356,
            -0.04964079707860947,
        ],
        "std": [
            0.3355240225791931,
            0.3784470558166504,
            0.44472837448120117,
            0.03924351558089256,
            0.06339313089847565,
            0.07797032594680786,
            0.9987710118293762,
        ],
        "max": [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375, 1.0],
        "min": [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.3675000071525574, -1.0],
    },
}


def make_readme() -> str:
    """Generate a README string for automatically generated datasets.

    Returns:
        String containing information about when and how the dataset was generated.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return f"This dataset was automatically generated by {__file__} on {now_utc.isoformat()}"


def make_libero_info(total_episodes: int, total_frames: int, total_chunks: int) -> dict:
    """Generate metadata dictionary for LIBERO dataset.

    Args:
        total_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        total_chunks: Total number of data chunks.

    Returns:
        Dictionary containing dataset metadata including codebase version, robot
        type, episode/frame counts, task information, and feature specifications.
    """
    return {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(LIBERO_TASKS),
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": LIBERO_CHUNK_SIZE,
        "fps": LIBERO_FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }


def uint8_arr_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a uint8 image array to PNG bytes.

    Args:
        arr: Image array in CHW format (channels, height, width) with uint8 values.

    Returns:
        PNG-encoded image bytes.
    """
    arr_hwc = np.transpose(arr, (1, 2, 0))  # shape: (244, 224, 3)
    img = Image.fromarray(arr_hwc, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def consolidate_task_result(task_result: dict, output_dir: str | Path, allow_overwrite: bool = False) -> None:
    """Consolidate LIBERO task results into a standardized dataset format.

    Args:
        task_result: Dictionary containing episode data with observations, actions,
            and metadata.
        output_dir: Directory path where the consolidated dataset will be saved.
        allow_overwrite: If True, overwrite existing output directory. Defaults to False.

    Raises:
        FileExistsError: If output_dir already exists and allow_overwrite is False.
    """
    output_dir = Path(output_dir)
    if allow_overwrite and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=False)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for idx, task in enumerate(LIBERO_TASKS):
            task_entry = json.dumps({"task_index": idx, "task": task})
            f.write(task_entry + "\n")

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(LIBERO_DEFAULT_STATS, f, indent=4)

    all_lengths = [d["done_index"] + 1 for d in task_result["per_episode"]]
    all_successes = [d["success"] for d in task_result["per_episode"]]
    n_episodes = len(all_lengths)

    with open(meta_dir / "info.json", "w") as f:
        json.dump(
            make_libero_info(
                total_episodes=n_episodes,
                total_frames=sum(all_lengths).item(),
                total_chunks=int(math.ceil(n_episodes / 1000)),
            ),
            f,
            indent=4,
        )

    with open(output_dir / "README.md", "w") as f:
        f.write(make_readme())

    global_idx_start = 0
    for ep_idx, (ep_len, success) in enumerate(zip(all_lengths, all_successes, strict=False)):
        ep_actions = task_result["episodes"]["action"][ep_idx][:ep_len]
        ep_actions = ep_actions.to(torch.float32).numpy(force=True)
        ep_images = task_result["episodes"]["observation"]["camera0"][ep_idx][:ep_len]
        ep_images = (ep_images * 255).to(torch.uint8).numpy(force=True)
        ep_wrist_images = task_result["episodes"]["observation"]["camera1"][ep_idx][:ep_len]
        ep_wrist_images = (ep_wrist_images * 255).to(torch.uint8).numpy(force=True)
        ep_states = task_result["episodes"]["observation"]["state"][ep_idx][:ep_len]
        ep_states = ep_states.to(torch.float32).numpy(force=True)

        prompt = task_result["episodes"]["observation"]["prompt"][ep_idx][0]
        task_idx = LIBERO_TASK_TO_IDX[prompt]
        df = pd.DataFrame(
            {
                "image": list(ep_images),
                "wrist_image": list(ep_wrist_images),
                "state": list(ep_states),
                "actions": list(ep_actions),
                "timestamp": np.arange(ep_len, dtype=np.float32) / np.float32(LIBERO_FPS),
                "frame_index": np.arange(ep_len, dtype=int),
                "episode_index": np.ones((ep_len,), dtype=int) * ep_idx,
                "index": global_idx_start + np.arange(ep_len, dtype=int),
                "task_index": [task_idx] * len(ep_actions),
            }
        )
        global_idx_start += ep_len

        cnt = count()
        df.image = df.image.apply(
            lambda x, counter=cnt: {
                "bytes": uint8_arr_to_png_bytes(x),
                "path": f"frame_{next(counter):06d}.png",
            }
        )

        cnt = count()
        df.wrist_image = df.wrist_image.apply(
            lambda x, counter=cnt: {
                "bytes": uint8_arr_to_png_bytes(x),
                "path": f"frame_{next(counter):06d}.png",
            }
        )

        df.actions = df.actions.apply(lambda x: x[:7])  # only keep first 7 dimensions of action
        df.state = df.state.apply(lambda x: x[:8])  # only keep first 8 dimensions of state

        chunk_idx = ep_idx // LIBERO_CHUNK_SIZE
        parquet_path = output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert the pandas DataFrame to a PyArrow Table without the pandas index
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Inject Hugging Face-style metadata so `datasets` will reconstruct the right features.
        # Target:
        # {b'huggingface': b'{"info": {"features": {"image": {"_type": "Image"}, ...}}}'}
        hf_features = {
            "image": {"_type": "Image"},
            "wrist_image": {"_type": "Image"},
            "state": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 8,
                "_type": "Sequence",
            },
            "actions": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 7,
                "_type": "Sequence",
            },
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }

        hf_meta = {"info": {"features": hf_features}}

        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[b"huggingface"] = json.dumps(hf_meta).encode("utf-8")
        table = table.replace_schema_metadata(schema_metadata)

        # Finally, write the Parquet file with the patched schema metadata
        pq.write_table(table, parquet_path)

        with open(meta_dir / "episodes.jsonl", "a") as f:
            ep_entry = json.dumps(
                {
                    "episode_index": ep_idx,
                    "tasks": [prompt],
                    "length": ep_len.item(),
                    "success": success,
                }
            )
            f.write(ep_entry + "\n")


def aggregate_task_results(results: list[dict]) -> dict:
    """Aggregate multiple LIBERO task results into a single result dictionary.

    Args:
        results: List of task result dictionaries to aggregate.

    Returns:
        Dictionary containing aggregated episode data with all observations,
        actions, and per-episode metadata combined.
    """
    logging.info(f"Aggregating {len(results)} result(s)")
    ret = {
        "per_episode": [],
        "episodes": {
            "observation": {
                "camera0": [],
                "camera1": [],
                "state": [],
                "prompt": [],
            },
            "action": [],
        },
    }

    for task_result in results:
        ret["per_episode"].extend(task_result["per_episode"])
        ret["episodes"]["observation"]["camera0"].extend(task_result["episodes"]["observation"]["camera0"])
        ret["episodes"]["observation"]["camera1"].extend(task_result["episodes"]["observation"]["camera1"])
        ret["episodes"]["observation"]["state"].extend(task_result["episodes"]["observation"]["state"])
        ret["episodes"]["observation"]["prompt"].extend(task_result["episodes"]["observation"]["prompt"])
        ret["episodes"]["action"].extend(task_result["episodes"]["action"])

    return ret
