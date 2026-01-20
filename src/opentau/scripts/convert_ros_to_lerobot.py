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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rosbags.highlevel import AnyReader

from opentau.configs import parser
from opentau.configs.ros2lerobot import RosToLeRobotConfig
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.utils.ros2lerobot import EXTRACTORS


def get_sec_from_timestamp(timestamp: Any) -> float:
    """Converts a ROS timestamp object to seconds.

    Args:
        timestamp: A ROS timestamp object containing 'sec' and 'nanosec' attributes.

    Returns:
        float: Time in seconds.
    """
    return timestamp.sec + timestamp.nanosec / 1e9


def synchronize_sensor_data(data: list[tuple[Any, Any]], fps: int, start_time: float) -> list[Any]:
    """Synchronize sensor data timestamps to a common FPS.

    Assumes the stream is recorded in increasing order of timestamps.

    Args:
        data (list): List of (timestamp, value) tuples.
        fps (int): Frames per second to target for synchronization.
        start_time (float): Start time of the bag in seconds.
    Returns:
        list: List of synchronized values.
    """
    # assuming the stream is recorded in increasing order of timestamps

    if data == []:
        return []
    sync = []
    final_timestamp = get_sec_from_timestamp(data[-1][0])
    sync.append(data[0][1])

    idx = 0
    total_frames = int((final_timestamp - start_time) * fps)

    for frame_idx in range(1, total_frames):
        current_timestamp = start_time + frame_idx / fps
        if idx >= len(data) - 1:
            sync.append(data[-1][1])
            continue
        while idx < len(data) - 1:
            if abs(current_timestamp - get_sec_from_timestamp(data[idx][0])) > abs(
                current_timestamp - get_sec_from_timestamp(data[idx + 1][0])
            ):
                idx += 1
            else:
                break
        sync.append(data[idx][1])

    return sync


def batch_synchronize_sensor_data(
    topic_data: dict[tuple[str, str], list], fps: int, start_time: float
) -> dict[tuple[str, str], list]:
    """Synchronize sensor data timestamps to a common FPS. Batch version.

    Args:
        topic_data (dict): Dictionary mapping topic names to lists of (timestamp, value) tuples.
        fps (int): Frames per second to target for synchronization.
        start_time (float): Start time of the bag in seconds.
    Returns:
        dict: Dictionary mapping topic names to lists of synchronized values.
    """
    sync_data = {}
    for topic, data in topic_data.items():
        sync_data[topic] = synchronize_sensor_data(data, fps, start_time)
    return sync_data


def extract_topics_from_mcap(cfg: RosToLeRobotConfig, mcap_path: Path) -> dict[tuple[str, str], list] | None:
    """Reads a ROS 2 MCAP bag and converts /joint_states to a dictionary format.

    Suitable for loading into LeRobot (or converting to HF Dataset).

    Args:
        cfg (RosToLeRobotConfig): Configuration object containing joint ordering.
        mcap_path (str | Path): Path to the input MCAP file.

    Returns:
        dict: Dictionary mapping topics to lists of (timestamp, value) tuples.
    """
    mcap_path = Path(mcap_path)
    logging.info(f"Scanning {mcap_path}...")

    # Data buffers
    # Stores lists of (timestamp, value) tuples for each topic
    topic_data = defaultdict(list)

    required_topics = defaultdict(list)
    required_topics_enum = defaultdict(str)
    for _, v in cfg.dataset_features.items():
        required_topics[v.ros_topic].append(v.topic_attribute)
        required_topics_enum[v.topic_attribute] = v.enum_values

    # 1. Setup Reader
    with AnyReader([mcap_path]) as reader:
        connections = reader.connections
        if not connections:
            logging.info("No connections found in bag!")
            return

        # Initialize extractors
        extractors = {k: v(cfg) for k, v in EXTRACTORS.items()}

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Try to get the header timestamp, otherwise fall back to bag timestamp
            ts = msg.header.stamp if hasattr(msg, "header") and hasattr(msg.header, "stamp") else timestamp

            if connection.topic in required_topics:
                attributes = required_topics[connection.topic]

                for attribute in attributes:
                    if required_topics_enum[attribute] in extractors:
                        extracted_data = extractors[required_topics_enum[attribute]](
                            msg, connection.topic, attribute
                        )
                        if extracted_data:
                            topic_data[(connection.topic, attribute)].append((ts, extracted_data))
                    else:
                        logging.exception(f"Extractor for {required_topics_enum[attribute]} not implemented")

    return topic_data


@parser.wrap()
def batch_convert_ros_bags(cfg: RosToLeRobotConfig) -> None:
    """Batch convert ROS bags to LeRobot dataset.

    Iterates through a directory of ROS bags, extracts necessary features, synchronizes them,
    and creates a LeRobot dataset with the given features.

    Args:
        cfg (RosToLeRobotConfig): Configuration object specifying input/output paths,
            FPS, joint order, and dataset features.
    """
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)

    dataset = LeRobotDataset.create(
        repo_id=str(output_path.name),
        fps=cfg.fps,
        root=output_path,
        robot_type="unknown",  # You might want to make this configurable
        features={
            k: {
                "dtype": v.dtype,
                "shape": tuple(v.shape),
                "names": cfg.joint_order if "state" in k or "action" in k else None,
            }
            for k, v in cfg.dataset_features.items()
        },
        use_videos=cfg.use_videos,  # Set to True if you are extracting images
    )

    # Iterate over all ros bags in the input path and store them as a dataset
    for bag_path in [p for p in input_path.iterdir() if p.is_dir()]:
        logging.info(f"Processing bag: {bag_path}")
        try:
            topic_data = extract_topics_from_mcap(cfg, bag_path)
            with open(bag_path / "metadata.yaml") as f:
                metadata = yaml.safe_load(f)
            task = metadata.get("task", "task not defined")
            start_time = (
                metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"] / 1e9
            )

            if topic_data is None:
                continue

            sync_data = batch_synchronize_sensor_data(topic_data, fps=cfg.fps, start_time=start_time)

            # Assuming '/joint_states' is the primary data source
            # We need to iterate through the synchronized data and add frames
            num_frames = float("inf")
            for _, data in sync_data.items():
                num_frames = min(num_frames, len(data))

            for i in range(num_frames):
                frame = {
                    k: np.array(sync_data[(v.ros_topic, v.topic_attribute)][i], dtype=v.dtype)
                    for k, v in cfg.dataset_features.items()
                }
                frame.update(
                    {
                        "task": task,  # Required field
                    }
                )

                dataset.add_frame(frame)

            dataset.save_episode()
            logging.info(f"Episode saved for {bag_path.name}")

        except Exception as e:
            logging.exception(f"Failed to convert {bag_path}: {e}")

    logging.info(f"Batch conversion complete. Saved to {output_path}")
    if cfg.use_videos:
        dataset.encode_videos()  # If videos were used


if __name__ == "__main__":
    batch_convert_ros_bags()
