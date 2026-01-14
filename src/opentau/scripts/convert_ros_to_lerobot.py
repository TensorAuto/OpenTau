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

# create config to specific metadata for conversion (features, naming, frequency)
# take in a directory of ros bags (each is an episode) - done
# each the entire rosbag into lists for each feature (use msg.header.stamp instead of timestamp) - done
# synchronize timesteps (nearest neighbor) - done
# create lerobot dataset - done
# add frames  - done
# save episode - done
# lerobot.finalize() (look into this from upstream lerobot)


from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from opentau.configs import parser
from opentau.configs.ros2lerobot import RosToLeRobotConfig
from opentau.datasets.lerobot_dataset import LeRobotDataset


def get_sec_from_timestamp(timestamp):
    """Converts a ROS timestamp object to seconds.

    Args:
        timestamp: A ROS timestamp object containing 'sec' and 'nanosec' attributes.

    Returns:
        float: Time in seconds.
    """
    return timestamp.sec + timestamp.nanosec / 1e9


def synchronize_sensor_data(data, fps):
    """Synchronize sensor data timestamps to a common FPS.

    Assumes the stream is recorded in increasing order of timestamps.

    Args:
        data (list): List of (timestamp, value) tuples.
        fps (int): Frames per second to target for synchronization.

    Returns:
        list: List of synchronized values.
    """
    # assuming the stream is recorded in increasing order of timestamps

    sync = []
    initial_timestamp = get_sec_from_timestamp(data[0][0])
    final_timestamp = get_sec_from_timestamp(data[-1][0])
    sync.append(data[0][1])

    idx = 0
    total_frames = int((final_timestamp - initial_timestamp) * fps)

    for frame_idx in range(1, total_frames):
        current_timestamp = initial_timestamp + frame_idx / fps
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


def batch_synchronize_sensor_data(topic_data, fps):
    """Synchronize sensor data timestamps to a common FPS. Batch version.

    Args:
        topic_data (dict): Dictionary mapping topic names to lists of (timestamp, value) tuples.
        fps (int): Frames per second to target for synchronization.

    Returns:
        dict: Dictionary mapping topic names to lists of synchronized values.
    """
    sync_data = {}
    for topic, data in topic_data.items():
        sync_data[topic] = synchronize_sensor_data(data, fps)
    return sync_data


def read_mcap_to_lerobot(cfg, mcap_path, output_path):
    """Reads a ROS 2 MCAP bag and converts /joint_states to a dictionary format.

    Suitable for loading into LeRobot (or converting to HF Dataset).

    Args:
        cfg (RosToLeRobotConfig): Configuration object containing joint ordering.
        mcap_path (str | Path): Path to the input MCAP file.
        output_path (str | Path): Path for output (currently unused).

    Returns:
        dict: Dictionary mapping topics to lists of (timestamp, value) tuples.
    """
    mcap_path = Path(mcap_path)
    print(f"Scanning {mcap_path}...")

    # Data buffers
    # Stores lists of (timestamp, value) tuples for each topic
    topic_data = defaultdict(list)

    required_topics = defaultdict(list)
    for _, v in cfg.dataset_features.items():
        required_topics[v.ros_topic].append(v.topic_attribute)

    # 1. Setup Reader
    with AnyReader([mcap_path]) as reader:
        connections = reader.connections
        if not connections:
            print("No connections found in bag!")
            return

        # Initialize joint state tracking
        joint_order = cfg.joint_order
        first_joint_msg = True

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Try to get the header timestamp, otherwise fall back to bag timestamp
            ts = msg.header.stamp if hasattr(msg, "header") and hasattr(msg.header, "stamp") else timestamp

            # Extract values based on topic type/name
            if connection.topic == "/joint_states" and connection.topic in required_topics:
                # --- Handle Joint Ordering ---
                if first_joint_msg:
                    if not joint_order:
                        joint_order = msg.name
                        print(f"Auto-detected joint order ({len(joint_order)} joints): {joint_order}")
                    first_joint_msg = False

                # Create a map for this message {name: index}
                current_map = {name: i for i, name in enumerate(msg.name)}

                # Extract data in the correct fixed order
                extracted_data = {}
                attributes = required_topics[connection.topic]
                for attribute in attributes:
                    extracted_data[attribute] = []

                try:
                    for j_name in joint_order:
                        idx = current_map[j_name]

                        # Safe extraction
                        for attribute in attributes:
                            extracted_data[attribute].append(
                                getattr(msg, attribute)[idx] if len(getattr(msg, attribute)) > idx else 0.0
                            )

                    # Store extracted values instead of raw message

                    for attribute in attributes:
                        topic_data[(connection.topic, attribute)].append((ts, extracted_data[attribute]))

                except KeyError:
                    pass

            # Add handlers for other topics here (e.g. images)
            elif "image" in connection.topic and connection.topic in required_topics:
                # Placeholder for image handling
                topic_data[connection.topic].append((ts, msg))  # Keep msg for now if extraction not defined
            else:
                topic_data[connection.topic].append((ts, msg))

    return topic_data


@parser.wrap()
def batch_convert_ros_bags(cfg: RosToLeRobotConfig):
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
                "names": cfg.joint_order if k == "observation.state" else None,
            }
            for k, v in cfg.dataset_features.items()
        },
        use_videos=False,  # Set to True if you are extracting images
    )

    # Iterate over all ros bags in the input path and store them as a dataset
    for bag_path in [p for p in input_path.iterdir() if p.is_dir()]:
        print(f"Processing bag: {bag_path}")
        try:
            topic_data = read_mcap_to_lerobot(cfg, bag_path, None)

            if topic_data is None:
                continue

            sync_data = batch_synchronize_sensor_data(topic_data, fps=cfg.fps)

            # Assuming '/joint_states' is the primary data source
            # We need to iterate through the synchronized data and add frames
            for _, data in sync_data.items():
                num_frames = len(data)

            for i in range(num_frames):
                frame = {
                    k: np.array(sync_data[(v.ros_topic, v.topic_attribute)][i], dtype=v.dtype)
                    for k, v in cfg.dataset_features.items()
                }
                frame.update(
                    {
                        "task": "dummy_task",  # Required field
                    }
                )

                dataset.add_frame(frame)

            dataset.save_episode()
            print(f"Episode saved for {bag_path.name}")

        except Exception as e:
            print(f"Failed to convert {bag_path}: {e}")
            import traceback

            traceback.print_exc()

    print(f"Batch conversion complete. Saved to {output_path}")
    dataset.encode_videos()  # If videos were used


if __name__ == "__main__":
    batch_convert_ros_bags()
