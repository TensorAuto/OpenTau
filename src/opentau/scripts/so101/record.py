# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Record a teleoperated SO-100/SO-101 dataset in OpenTau's native LeRobotDataset format.

The dataset is written with :class:`opentau.datasets.lerobot_dataset.LeRobotDataset`
(codebase v2.1), so it can be trained on directly by OpenTau.

Keyboard controls during recording:
    right arrow  -> end current episode early (proceed to reset/save)
    left arrow   -> re-record current episode
    escape       -> stop recording after this episode

Example:
    python -m opentau.scripts.so101.record \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
        --robot.cameras='{"front": {"type": "opencv", "index_or_path": "/dev/video0",
            "width": 640, "height": 480, "fps": 30}}' \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader \
        --dataset.repo_id=TensorAuto/my_task --dataset.single_task="do the task" \
        --dataset.num_episodes=50
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus
import numpy as np

from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.scripts.so101.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from opentau.scripts.so101.constants import ACTION, OBS_IMAGES, OBS_STATE
from opentau.scripts.so101.robots import (  # noqa: F401  (registers draccus choices)
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from opentau.scripts.so101.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
)
from opentau.scripts.so101.utils import init_logging, log_say, precise_sleep


@dataclass
class DatasetRecordConfig:
    # Dataset identifier e.g. `TensorAuto/so101_chip_pick_place_3`.
    repo_id: str = ""
    # Natural-language task description, e.g. "put the red chip bag into the white bin".
    single_task: str = ""
    # Root directory for the dataset (default: ~/.cache/huggingface/lerobot/{repo_id}).
    root: str | None = None
    fps: int = 30
    episode_time_s: float = 60
    reset_time_s: float = 10
    num_episodes: int = 50
    # Encode camera frames as videos (v2.1 video features) rather than image files.
    video: bool = True
    push_to_hub: bool = False
    private: bool = True
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self):
        if not self.repo_id or not self.single_task:
            raise ValueError("--dataset.repo_id and --dataset.single_task are required.")


@dataclass
class RecordConfig:
    robot: RobotConfig = field(default_factory=lambda: None)  # type: ignore[assignment]
    teleop: TeleoperatorConfig = field(default_factory=lambda: None)  # type: ignore[assignment]
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    # Resume recording into an existing local dataset.
    resume: bool = False
    play_sounds: bool = True

    def __post_init__(self):
        if self.robot is None or self.teleop is None:
            raise ValueError("Both --robot and --teleop must be provided.")


def open_dataset_for_append(repo_id: str, root: str | None) -> LeRobotDataset:
    """Open an existing local v2.1 dataset for recording more episodes.

    OpenTau's ``LeRobotDataset.__init__`` is training-oriented (requires a
    ``TrainPipelineConfig``), so this mirrors the ``create()`` classmethod's
    ``__new__`` assembly against the existing on-disk metadata instead.
    """
    from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from opentau.datasets.utils import get_episode_data_index
    from opentau.datasets.video_utils import get_safe_default_codec

    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = LeRobotDatasetMetadata(repo_id, root=root)
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.deferred_video_keys = set()
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps_params = obj.compute_delta_params(None, None, None, None)
    obj.episode_data_index = None
    obj.enable_prompt_substitution = True
    obj._prompt_substitutions_by_index = {}
    obj.video_backend = get_safe_default_codec()
    obj.image_resample_strategy = "nearest"
    obj.vector_resample_strategy = "nearest"
    obj.standardize = True
    obj.skip_video_stats = False
    obj._overlay = None  # video-overlay feature of the full constructor; unused when recording
    obj.episode_data_index, obj.epi2idx = get_episode_data_index(obj.meta.episodes, obj.episodes)
    return obj


def build_dataset_features(robot, use_videos: bool) -> dict[str, dict]:
    """Map the robot's observation/action features to v2.1 dataset features."""
    motor_keys = list(robot.action_features)
    cam_shapes = {k: v for k, v in robot.observation_features.items() if isinstance(v, tuple)}

    features: dict[str, dict] = {
        ACTION: {"dtype": "float32", "shape": (len(motor_keys),), "names": motor_keys},
        OBS_STATE: {"dtype": "float32", "shape": (len(motor_keys),), "names": motor_keys},
    }
    for cam, (h, w, c) in cam_shapes.items():
        features[f"{OBS_IMAGES}.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }
    return features


def init_keyboard_listener(events: dict):
    """Arrow-key episode controls via pynput; no-op when headless."""
    try:
        from pynput import keyboard
    except Exception:
        logging.warning("pynput unavailable (headless?); keyboard controls disabled.")
        return None

    def on_press(key):
        if key == keyboard.Key.right:
            print("Right arrow: ending episode early...")
            events["exit_early"] = True
        elif key == keyboard.Key.left:
            print("Left arrow: re-recording episode...")
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif key == keyboard.Key.esc:
            print("Escape: stopping after this episode...")
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def record_episode(robot, teleop, dataset, events: dict, task: str, episode_time_s: float, fps: int):
    motor_keys = list(robot.action_features)
    # Camera keys must come from observation_features (like build_dataset_features), NOT
    # robot.cameras: the bi-arm robot prefixes observation keys with left_/right_, so its
    # merged robot.cameras dict (raw names) matches neither the observation dict nor the
    # dataset feature names. observation_features keys match both for single- and bi-arm.
    cam_keys = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]
    start_t = time.perf_counter()
    while time.perf_counter() - start_t < episode_time_s and not events["exit_early"]:
        loop_start = time.perf_counter()

        observation = robot.get_observation()
        action = teleop.get_action()
        sent_action = robot.send_action(action)

        frame = {
            OBS_STATE: np.array([observation[k] for k in motor_keys], dtype=np.float32),
            ACTION: np.array([sent_action[k] for k in motor_keys], dtype=np.float32),
            "task": task,
        }
        for cam in cam_keys:
            frame[f"{OBS_IMAGES}.{cam}"] = observation[cam]
        dataset.add_frame(frame)

        precise_sleep(1 / fps - (time.perf_counter() - loop_start))
    events["exit_early"] = False


@draccus.wrap()
def record(cfg: RecordConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    # Count cameras on the instantiated robot, not the config: bi-arm configs
    # nest cameras per arm, but the robot exposes the merged dict.
    num_cams = len(getattr(robot, "cameras", {}) or {})
    features = None

    if cfg.resume:
        dataset = open_dataset_for_append(cfg.dataset.repo_id, cfg.dataset.root)
        if num_cams > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cams,
            )
    else:
        features = build_dataset_features(robot, cfg.dataset.video)  # shapes come from config only
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * max(num_cams, 1),
        )
        # SO-101 arms are joint-position controlled; the mixture sampler reads
        # this from meta/info.json to resolve norm heads (robot_type::control_mode).
        from opentau.datasets.utils import write_info

        dataset.meta.info["control_mode"] = "joint"
        write_info(dataset.meta.info, dataset.meta.root)

    robot.connect()
    teleop.connect()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = init_keyboard_listener(events)

    try:
        # `recorded` counts episodes already in the dataset, so num_episodes is the
        # dataset TOTAL: a fresh dataset starts at 0; --resume continues where it
        # left off (e.g. 9 done, num_episodes=50 -> records episodes 10..50).
        recorded = dataset.num_episodes
        if recorded >= cfg.dataset.num_episodes:
            logging.warning(
                "Dataset already has %d episode(s) >= --dataset.num_episodes=%d; nothing to record.",
                recorded,
                cfg.dataset.num_episodes,
            )
        while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded + 1} of {cfg.dataset.num_episodes}", cfg.play_sounds)
            record_episode(
                robot,
                teleop,
                dataset,
                events,
                cfg.dataset.single_task,
                cfg.dataset.episode_time_s,
                cfg.dataset.fps,
            )

            if events["rerecord_episode"]:
                log_say("Re-recording episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                # Drain the async image writer before deleting the episode's image
                # dir: frames still in its queue land AFTER rmtree starts, leaving
                # "Directory not empty" crashes and orphaned files.
                if dataset.image_writer is not None:
                    dataset.image_writer.wait_until_done()
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded += 1

            if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say("Reset the environment", cfg.play_sounds)
                reset_start = time.perf_counter()
                while (
                    time.perf_counter() - reset_start < cfg.dataset.reset_time_s and not events["exit_early"]
                ):
                    time.sleep(0.1)
                events["exit_early"] = False
    except KeyboardInterrupt:
        logging.info("Interrupted — discarding the partial episode and stopping.")
        # Drop the partially recorded episode cleanly: drain the async image
        # writer, then delete its buffered frames + image dir so the dataset is
        # left resume-ready (no orphaned episode_XXXXXX image folders).
        try:
            if dataset.image_writer is not None:
                dataset.image_writer.wait_until_done()
            if dataset.episode_buffer is not None and dataset.episode_buffer["size"] > 0:
                dataset.clear_episode_buffer()
        except Exception:
            logging.warning("Partial-episode cleanup failed:", exc_info=True)
    finally:
        robot.disconnect()
        teleop.disconnect()
        if listener is not None:
            listener.stop()

    log_say("Recording finished", cfg.play_sounds)
    logging.info("Recorded %d episode(s) at %s", dataset.num_episodes, dataset.root)

    if cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to the Hub as %s", cfg.dataset.repo_id)
        dataset.push_to_hub(private=cfg.dataset.private)


def main() -> None:
    record()


if __name__ == "__main__":
    main()
