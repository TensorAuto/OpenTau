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
"""Build a LeRobot dataset from a human video using MediaPipe pose or hand landmarks.

Reads an input MP4, runs pose (exo) or hand (ego) detection, and writes a LeRobot
dataset with: video frames as observation.images.camera, 3D landmarks as
observation.state, and action = subsequent state. The user provides a task prompt.
Optionally writes a landmark-overlay video.

Example:
    # LeRobot dataset only (exo = pose)
    python human_video_to_lerobot.py input.mp4 ./my_dataset --prompt "Pick up the cup" --mode exo

    # Output at 10 FPS regardless of video FPS
    python human_video_to_lerobot.py input.mp4 ./my_dataset --prompt "Pick up the cup" --fps 10

    # With overlay video
    python human_video_to_lerobot.py input.mp4 ./my_dataset --prompt "Pick up the cup" --overlay overlay.mp4

    # Ego (hand landmarks)
    python human_video_to_lerobot.py ego_demo.mp4 ./hand_dataset --prompt "Open the drawer" --mode ego
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from opentau.datasets.lerobot_dataset import LeRobotDataset

# Pose landmarker model (exo / 3rd person) – heavy variant only
# See https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Hand landmarker model (ego / 1st person)
# See https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Landmark counts for state dimension
POSE_NUM_LANDMARKS = 33
POSE_STATE_DIM = POSE_NUM_LANDMARKS * 3
HAND_NUM_LANDMARKS = 21
HAND_STATE_DIM_PER_HAND = HAND_NUM_LANDMARKS * 3


def get_pose_model_path(cache_dir: Path) -> Path:
    """Return path to pose landmarker model (heavy), downloading if needed."""
    path = cache_dir / "pose_landmarker_heavy.task"
    if path.is_file():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model (heavy) to {path}...")
    urllib.request.urlretrieve(POSE_MODEL_URL, path)  # nosec B310 - URL is trusted HTTPS constant
    return path


def get_hand_model_path(cache_dir: Path) -> Path:
    """Return path to hand landmarker model, downloading if needed."""
    path = cache_dir / "hand_landmarker.task"
    if path.is_file():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {path}...")
    urllib.request.urlretrieve(HAND_MODEL_URL, path)  # nosec B310 - URL is trusted HTTPS constant
    return path


def pose_world_landmarks_to_state(results) -> np.ndarray:
    """Extract 3D pose world landmarks as a flat float32 vector. Returns zeros if none."""
    out = np.zeros(POSE_STATE_DIM, dtype=np.float32)
    if not results.pose_world_landmarks or len(results.pose_world_landmarks) == 0:
        return out
    # Use first detected pose
    landmarks = results.pose_world_landmarks[0]
    for i, lm in enumerate(landmarks):
        if i >= POSE_NUM_LANDMARKS:
            break
        out[i * 3] = lm.x
        out[i * 3 + 1] = lm.y
        out[i * 3 + 2] = lm.z
    return out


def hand_world_landmarks_to_state(results, num_hands: int) -> np.ndarray:
    """Extract 3D hand world landmarks as a flat float32 vector. Returns zeros if none."""
    state_dim = num_hands * HAND_STATE_DIM_PER_HAND
    out = np.zeros(state_dim, dtype=np.float32)
    if not results.hand_world_landmarks or len(results.hand_world_landmarks) == 0:
        return out
    for idx, hand_landmarks in enumerate(results.hand_world_landmarks):
        if idx >= num_hands:
            break
        for j, lm in enumerate(hand_landmarks):
            if j >= HAND_NUM_LANDMARKS:
                break
            base = idx * HAND_STATE_DIM_PER_HAND + j * 3
            out[base] = lm.x
            out[base + 1] = lm.y
            out[base + 2] = lm.z
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LeRobot dataset from a human video using pose or hand landmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input MP4 file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output LeRobot dataset root (directory must not exist).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Task description (prompt) for the episode.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="If set, also write a landmark-overlay MP4 to this path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output dataset FPS. If not set, use the input video's FPS. Frames are sampled to match.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ego", "exo"],
        default="exo",
        help="Video type: exo = 3rd person (pose landmarks), ego = 1st person (hand landmarks).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for pose/hand detection.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for pose/hand tracking.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Max number of hands to detect (ego only).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "mediapipe",
        help="Directory to cache downloaded models.",
    )
    return parser.parse_args()


def run_detection_and_collect_states(
    input_path: Path,
    args: argparse.Namespace,
    landmarker,  # PoseLandmarker | HandLandmarker
    is_exo: bool,
    overlay_writer: cv2.VideoWriter | None,
    target_fps: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float, int, int, float, int]:
    """Run detection on video; return states, actions, target_fps, width, height, video_fps, total_frames."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_fps <= 0:
        video_fps = 30.0
    if total_frames <= 0:
        total_frames = 1

    num_output_frames = max(1, int(round(total_frames * target_fps / video_fps)))
    # For each output frame index i (at target_fps), pick the video frame at time i/target_fps
    # seconds: frame_index = round(i * video_fps / target_fps), clamped to the last frame.
    sampled_frame_indices = [
        min(int(round(i * video_fps / target_fps)), total_frames - 1) for i in range(num_output_frames)
    ]
    states: list[np.ndarray] = []

    if overlay_writer:
        # Overlay gets every frame at video_fps; dataset state only for sampled frames
        next_sampled_idx = 0
        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(1000 * frame_index / video_fps) if video_fps > 0 else frame_index
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if is_exo:
                state = pose_world_landmarks_to_state(results)
                if results.pose_landmarks:
                    pose_style = drawing_styles.get_default_pose_landmarks_style()
                    conn_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                    for pose_landmarks in results.pose_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            pose_landmarks,
                            vision.PoseLandmarksConnections.POSE_LANDMARKS,
                            landmark_drawing_spec=pose_style,
                            connection_drawing_spec=conn_style,
                        )
            else:
                state = hand_world_landmarks_to_state(results, args.num_hands)
                if results.hand_landmarks:
                    hand_style = drawing_styles.get_default_hand_landmarks_style()
                    conn_style = drawing_styles.get_default_hand_connections_style()
                    for hand_landmarks in results.hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,
                            landmark_drawing_spec=hand_style,
                            connection_drawing_spec=conn_style,
                        )

            overlay_writer.write(frame)
            if (
                next_sampled_idx < len(sampled_frame_indices)
                and frame_index == sampled_frame_indices[next_sampled_idx]
            ):
                states.append(state)
                next_sampled_idx += 1
    else:
        # No overlay: only process sampled frames
        for i in range(num_output_frames):
            frame_idx = sampled_frame_indices[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(1000 * i / target_fps) if target_fps > 0 else i
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if is_exo:
                state = pose_world_landmarks_to_state(results)
            else:
                state = hand_world_landmarks_to_state(results, args.num_hands)
            states.append(state)

    cap.release()
    if overlay_writer:
        overlay_writer.release()

    # Actions: next state; last frame action = last state
    actions = [states[i + 1].copy() for i in range(len(states) - 1)] + [states[-1].copy()]

    return states, actions, target_fps, width, height, video_fps, total_frames


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path.exists():
        raise FileExistsError(
            f"Output path must not exist (LeRobot dataset will be created there): {output_path}"
        )

    overlay_path = args.overlay.resolve() if args.overlay else None
    if overlay_path:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

    is_exo = args.mode == "exo"
    if is_exo:
        state_dim = POSE_STATE_DIM
        model_path = get_pose_model_path(args.model_cache_dir)
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
    else:
        state_dim = args.num_hands * HAND_STATE_DIM_PER_HAND
        model_path = get_hand_model_path(args.model_cache_dir)
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_detection_confidence,
            min_hand_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

    cap_preview = cv2.VideoCapture(str(input_path))
    video_fps = cap_preview.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_preview.release()

    target_fps = args.fps if args.fps is not None else video_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Overlay video always uses the input video's FPS; --fps only affects the LeRobot dataset
    overlay_writer = (
        cv2.VideoWriter(str(overlay_path), fourcc, video_fps, (width, height)) if overlay_path else None
    )

    try:
        states, actions, target_fps, width, height, video_fps, total_frames = (
            run_detection_and_collect_states(
                input_path=input_path,
                args=args,
                landmarker=landmarker,
                is_exo=is_exo,
                overlay_writer=overlay_writer,
                target_fps=target_fps,
            )
        )
    finally:
        landmarker.close()

    num_frames = len(states)
    if num_frames == 0:
        raise RuntimeError("No frames read from video.")

    # Build LeRobot dataset features: one camera video, state, action
    image_key = "observation.images.camera"
    state_key = "observation.state"
    action_key = "action"
    fps_int = int(round(target_fps)) or 30
    features = {
        image_key: {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        state_key: {"dtype": "float32", "shape": (state_dim,), "names": None},
        action_key: {"dtype": "float32", "shape": (state_dim,), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=output_path.name,
        fps=fps_int,
        root=output_path,
        robot_type="human",
        features=features,
        use_videos=True,
    )

    # Second pass: read the same sampled frames and add to dataset
    cap = cv2.VideoCapture(str(input_path))
    for i in range(num_frames):
        frame_idx = min(int(round(i * video_fps / target_fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_dict = {
            image_key: frame_rgb,
            state_key: states[i].astype(np.float32),
            action_key: actions[i].astype(np.float32),
            "task": args.prompt,
        }
        dataset.add_frame(frame_dict)
    cap.release()

    dataset.save_episode()

    print(f"LeRobot dataset written to {output_path} ({num_frames} frames, 1 episode).")
    if overlay_path:
        print(f"Overlay video written to {overlay_path}.")


if __name__ == "__main__":
    main()
