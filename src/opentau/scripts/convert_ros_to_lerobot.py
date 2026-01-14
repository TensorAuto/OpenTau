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


from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from opentau.configs import parser
from opentau.configs.ros2lerobot import RosToLeRobotConfig
from opentau.datasets.lerobot_dataset import LeRobotDataset


def get_sec_from_timestamp(timestamp):
    return timestamp.sec + timestamp.nanosec / 1e9


def synchronize_sensor_data(data, fps):
    """
    Synchronize sensor data timestamps to a common FPS.
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
    """
    Synchronize sensor data timestamps to a common FPS. Batch version.
    """
    sync_data = {}
    for topic, data in topic_data.items():
        sync_data[topic] = synchronize_sensor_data(data, fps)
    return sync_data


def read_mcap_to_lerobot(cfg, mcap_path, output_path):
    """
    Reads a ROS 2 MCAP bag and converts /joint_states to a dictionary format
    suitable for loading into LeRobot (or converting to HF Dataset).
    """

    mcap_path = Path(mcap_path)
    print(f"Scanning {mcap_path}...")

    # Data buffers
    # Stores lists of (timestamp, value) tuples for each topic
    topic_data = {}

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

            if connection.topic not in topic_data:
                topic_data[connection.topic] = []

            # Extract values based on topic type/name
            if connection.topic == "/joint_states":
                # --- Handle Joint Ordering ---
                if first_joint_msg:
                    if not joint_order:
                        joint_order = msg.name
                        print(f"Auto-detected joint order ({len(joint_order)} joints): {joint_order}")
                    first_joint_msg = False

                # Create a map for this message {name: index}
                current_map = {name: i for i, name in enumerate(msg.name)}

                # Extract data in the correct fixed order
                pos_vec = []

                try:
                    for j_name in joint_order:
                        idx = current_map[j_name]

                        # Safe extraction
                        p = msg.position[idx] if len(msg.position) > idx else 0.0

                        pos_vec.append(p)

                    # Store extracted values instead of raw message
                    topic_data[connection.topic].append((ts, pos_vec))

                except KeyError:
                    pass

            # Add handlers for other topics here (e.g. images)
            elif "image" in connection.topic:
                # Placeholder for image handling
                topic_data[connection.topic].append((ts, msg))  # Keep msg for now if extraction not defined
            else:
                topic_data[connection.topic].append((ts, msg))

    return topic_data


@parser.wrap()
def batch_convert_ros_bags(cfg: RosToLeRobotConfig):
    """
    Batch convert ROS bags to LeRobot dataset.
    """
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)

    dataset = LeRobotDataset.create(
        repo_id=str(output_path.name),
        fps=cfg.fps,
        root=output_path,
        robot_type="unknown",  # You might want to make this configurable
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (25,),  # Based on your joint_order list length (25 joints)
                "names": cfg.joint_order,
            },
        },
        use_videos=False,  # Set to True if you are extracting images
    )

    # Iterate over all subdirectories in the input path
    for bag_path in [p for p in input_path.iterdir() if p.is_dir()]:
        print(f"Processing bag: {bag_path}")
        try:
            # We don't need output_path here anymore for individual pickles,
            # we just need the data to add to the dataset
            topic_data = read_mcap_to_lerobot(cfg, bag_path, None)

            if topic_data is None:
                continue

            sync_data = batch_synchronize_sensor_data(topic_data, fps=cfg.fps)

            # Assuming '/joint_states' is the primary data source
            # We need to iterate through the synchronized data and add frames
            if "/joint_states" in sync_data:
                joint_states = sync_data["/joint_states"]
                num_frames = len(joint_states)

                print(f"Adding {num_frames} frames to dataset...")

                for i in range(num_frames):
                    # Prepare frame dictionary
                    # sync_data['/joint_states'][i] is just the value list/array now,
                    # because synchronize_sensor_data returns list of values.
                    # Wait, read_mcap_to_lerobot returns (ts, pos_vec) for joint_states.
                    # synchronize_sensor_data returns a list of values (pos_vecs).

                    # We need to reconstruct the frame dict expected by LeRobotDataset

                    frame = {
                        "observation.state": np.array(joint_states[i], dtype=np.float32),
                        "task": "dummy_task",  # Required field
                    }

                    dataset.add_frame(frame)

                dataset.save_episode()
                print(f"Episode saved for {bag_path.name}")

        except Exception as e:
            print(f"Failed to convert {bag_path}: {e}")
            import traceback

            traceback.print_exc()

    print(f"Batch conversion complete. Saved to {output_path}")
    dataset.encode_videos()  # If videos were used
    # Finalize/push if needed, but save_episode writes to disk.


if __name__ == "__main__":
    batch_convert_ros_bags()
