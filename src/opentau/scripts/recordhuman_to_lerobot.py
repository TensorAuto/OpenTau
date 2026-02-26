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

"""Convert Pico headset recordings (video + hand/head pose JSON) to a LeRobot dataset.

State vector (364-D): camera-space pose of each hand joint (26 joints × 7 floats
[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]) for left hand (182)
concatenated with right hand (182). Zeros when a hand is not tracked.

Action vector (371-D): next frame's state (364) concatenated with the delta head
pose (7 = delta_pos(3) + delta_rot(4)). Delta position is pos_{t+1} - pos_{t}.
Delta rotation is the relative quaternion q_{t+1} * q_t^{-1}. The last frame
uses zero delta position and identity quaternion [0,0,0,1].

Video frames are stored as observation.images.camera.

Optionally generates a skeleton overlay video with --overlay.

Example:
    python src/opentau/scripts/recordhuman_to_lerobot.py \\
        --video recording.mp4 --poses recording.json \\
        --output ./my_dataset --prompt "Pick up the snack bag"

    # With overlay video
    python src/opentau/scripts/recordhuman_to_lerobot.py \\
        --video recording.mp4 --poses recording.json \\
        --output ./my_dataset --prompt "Pick up the snack bag" \\
        --overlay overlay.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from opentau.datasets.lerobot_dataset import LeRobotDataset

JOINTS_PER_HAND = 26
FLOATS_PER_JOINT = 7  # pos(3) + rot(4)
STATE_DIM_PER_HAND = JOINTS_PER_HAND * FLOATS_PER_JOINT  # 182
STATE_DIM = STATE_DIM_PER_HAND * 2  # 364: left(182) + right(182)
HEAD_POSE_DIM = 7  # pos(3) + rot(4)
ACTION_DIM = STATE_DIM + HEAD_POSE_DIM  # 371: next-state(364) + delta head pose(7)

FINGERTIP_INDICES = {5, 10, 15, 20, 25}

SKELETON_BONES = [
    (1, 0),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
]

LEFT_HAND_COLOR = (255, 150, 50)
RIGHT_HAND_COLOR = (50, 150, 255)
HEAD_AXIS_COLORS = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
JOINT_RADIUS = 4
BONE_THICKNESS = 2
HEAD_AXIS_DISPLAY_LEN = 40


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def quat_to_rotation_matrix(q):
    """Convert an (x, y, z, w) quaternion to a 3x3 rotation matrix.

    The returned matrix R rotates column vectors from the local frame into the
    world frame: ``p_world = R @ p_local``.

    Args:
        q: Quaternion as a 4-element array-like in (x, y, z, w) order.
            Normalized internally.

    Returns:
        A (3, 3) numpy rotation matrix.
    """
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_matrix_to_quat(R):
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion.

    Selects the numerically stable branch
    based on the matrix diagonal.

    Note:
        The sign of the returned quaternion is arbitrary (q and -q represent
        the same rotation). Avoid round-tripping through this function when
        sign consistency matters; use ``quat_inverse`` directly instead.

    Args:
        R: A (3, 3) rotation matrix.

    Returns:
        A length-4 numpy array ``[x, y, z, w]``.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def quat_inverse(q):
    """Return the inverse of a unit quaternion (its conjugate).

    Args:
        q: Unit quaternion as a 4-element array-like in (x, y, z, w) order.

    Returns:
        A length-4 numpy array ``[-x, -y, -z, w]``.
    """
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_multiply(q1, q2):
    """Compute the Hamilton product of two quaternions.

    The result represents rotation q1 applied *after* q2:
    ``q_out = q1 * q2`` means "first rotate by q2, then by q1".

    Args:
        q1: Left quaternion, (x, y, z, w).
        q2: Right quaternion, (x, y, z, w).

    Returns:
        A length-4 numpy array ``[x, y, z, w]``.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def world_to_camera(points_tracking, head_pos_tracking, head_rot_q, eye_offset=None):
    """Transform 3-D points from tracking (world) space to camera (head-local) space.

    Computes ``p_cam = R_world_to_cam @ (p_world - eye_pos)`` using row-vector
    convention internally (``(N,3) @ (3,3)``).

    Args:
        points_tracking: (N, 3) array of points in tracking/world coordinates.
        head_pos_tracking: (3,) head position in tracking space (after origin
            subtraction).
        head_rot_q: (4,) head orientation quaternion (x, y, z, w) mapping local
            to world.
        eye_offset: Optional (3,) offset from head center to the eye in head-
            local coordinates. Applied after rotating by head orientation.

    Returns:
        (N, 3) array of points in camera space.
    """
    R_local_to_world = quat_to_rotation_matrix(head_rot_q)
    R_world_to_local = R_local_to_world.T
    eye_pos = head_pos_tracking
    if eye_offset is not None:
        eye_pos = head_pos_tracking + R_local_to_world @ eye_offset
    return (points_tracking - eye_pos) @ R_world_to_local.T


def project_camera_to_pixel(points_cam, fx, fy, cx, cy):
    """Project camera-space 3-D points to 2-D pixel coordinates (pinhole model).

    Y is negated so that camera-up maps to image-down (standard image coords).
    Points behind the camera (Z <= 0.01) are marked invalid and set to NaN.

    Args:
        points_cam: (N, 3) array in camera coordinates (Z forward, Y up).
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).

    Returns:
        Tuple of ``(px, py, valid)`` where ``px`` and ``py`` are (N,) float
        arrays (NaN for invalid points), and ``valid`` is a boolean mask.
    """
    valid = points_cam[:, 2] > 0.01
    px = np.full(len(points_cam), np.nan)
    py = np.full(len(points_cam), np.nan)
    z = points_cam[valid, 2]
    px[valid] = fx * points_cam[valid, 0] / z + cx
    py[valid] = -fy * points_cam[valid, 1] / z + cy
    return px, py, valid


def parse_hand_joints(flat_array):
    """Parse a flat joint array into a structured (26, 7) array.

    Args:
        flat_array: Flat sequence of 182 floats (26 joints x 7 values per
            joint: pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w).

    Returns:
        (26, 7) float64 numpy array.
    """
    return np.array(flat_array, dtype=np.float64).reshape(JOINTS_PER_HAND, FLOATS_PER_JOINT)


def parse_hand_positions(flat_array):
    """Parse a flat joint array and return only the 3-D positions.

    Args:
        flat_array: Flat sequence of 182 floats (see ``parse_hand_joints``).

    Returns:
        (26, 3) float64 numpy array of joint positions.
    """
    return parse_hand_joints(flat_array)[:, :3]


# ---------------------------------------------------------------------------
# Drawing helpers (used only when --overlay is set)
# ---------------------------------------------------------------------------


def draw_hand(frame, positions_2d, valid, color):
    """Draw a hand skeleton (bones and joints) onto an image.

    Args:
        frame: BGR image (H, W, 3), modified in place.
        positions_2d: (26, 2) array of pixel coordinates per joint.
        valid: (26,) boolean mask indicating which joints are visible.
        color: BGR tuple for the skeleton color.
    """
    h, w = frame.shape[:2]
    for a, b in SKELETON_BONES:
        if not (valid[a] and valid[b]):
            continue
        p1 = (int(round(positions_2d[a, 0])), int(round(positions_2d[a, 1])))
        p2 = (int(round(positions_2d[b, 0])), int(round(positions_2d[b, 1])))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, color, BONE_THICKNESS, cv2.LINE_AA)
    for i in range(JOINTS_PER_HAND):
        if not valid[i]:
            continue
        pt = (int(round(positions_2d[i, 0])), int(round(positions_2d[i, 1])))
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue
        r = JOINT_RADIUS + 2 if i in FINGERTIP_INDICES else JOINT_RADIUS
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (255, 255, 255), 1, cv2.LINE_AA)


def draw_head_gizmo(frame, head_rot_q):
    """Draw a 3-axis orientation gizmo showing how world axes appear in camera view.

    Renders X (red), Y (green), Z (blue) arrows in the upper-left corner of
    the frame. Each arrow shows the direction of the corresponding world axis
    as seen from the head's local frame.

    Args:
        frame: BGR image (H, W, 3), modified in place.
        head_rot_q: (4,) head orientation quaternion (x, y, z, w), local-to-world.
    """
    R = quat_to_rotation_matrix(head_rot_q)
    Rw2l = R.T
    ox, oy = 60, 60
    for axis_name, color in HEAD_AXIS_COLORS.items():
        idx = {"x": 0, "y": 1, "z": 2}[axis_name]
        axis = np.zeros(3)
        axis[idx] = 1.0
        d = Rw2l @ axis
        ex, ey = int(ox + d[0] * HEAD_AXIS_DISPLAY_LEN), int(oy - d[1] * HEAD_AXIS_DISPLAY_LEN)
        cv2.arrowedLine(frame, (ox, oy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(
            frame, axis_name.upper(), (ex + 4, ey + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    cv2.putText(
        frame,
        "Head",
        (ox - 20, oy + HEAD_AXIS_DISPLAY_LEN + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, video_time, pose_time, frame_idx, total_frames, left_tracked, right_tracked):
    """Draw a heads-up display with timing and hand-tracking status.

    Args:
        frame: BGR image (H, W, 3), modified in place.
        video_time: Current video timestamp in seconds.
        pose_time: Matched pose-data timestamp in seconds.
        frame_idx: Current video frame index.
        total_frames: Total number of video frames.
        left_tracked: Whether the left hand is currently tracked.
        right_tracked: Whether the right hand is currently tracked.
    """
    h, w = frame.shape[:2]
    info = f"t={video_time:.2f}s  pose_t={pose_time:.2f}s  frame {frame_idx}/{total_frames}"
    cv2.putText(frame, info, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    lt, rt = "ON" if left_tracked else "off", "ON" if right_tracked else "off"
    sc = (
        (0, 255, 0)
        if (left_tracked and right_tracked)
        else (0, 200, 255)
        if (left_tracked or right_tracked)
        else (0, 0, 255)
    )
    cv2.putText(frame, f"L:{lt}  R:{rt}", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core: process one frame's pose data into camera-space state
# ---------------------------------------------------------------------------


def compute_frame_state(pf, head_pos_tracking, head_rot, eye_offset):
    """Build the 364-D camera-space state vector for one pose frame.

    Each hand contributes 182 floats (26 joints x 7: pos(3) + quat(4)), all
    expressed in head/camera space. An untracked hand is left as zeros.

    Joint positions are transformed via ``world_to_camera``. Joint quaternions
    are rotated by ``q_head^{-1}`` (directly, without a matrix round-trip) to
    convert from world orientation to camera-local orientation.

    Note:
        ``head_pos_tracking`` is expected to already have the XR tracking
        origin subtracted. The raw joint positions from ``pf`` are used
        as-is because the Pico recording format stores them in a coordinate
        system that does not require the same origin correction.

    Args:
        pf: Single frame dict from the pose JSON (must contain ``left_joints``
            / ``right_joints`` and ``left_tracked`` / ``right_tracked``).
        head_pos_tracking: (3,) head position in tracking space (origin-
            subtracted).
        head_rot: (4,) head orientation quaternion (x, y, z, w), local-to-world.
        eye_offset: Optional (3,) eye offset in head-local coords, or ``None``.

    Returns:
        Tuple of ``(state, left_cam_pos, right_cam_pos)`` where ``state`` is a
        float32 array of shape ``(364,)``, and each ``*_cam_pos`` is either a
        ``(26, 3)`` float64 array of camera-space joint positions or ``None``
        if that hand is untracked.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    left_cam_pos = right_cam_pos = None

    q_head_inv = quat_inverse(head_rot)

    for hand_idx, (jkey, tkey) in enumerate(
        [
            ("left_joints", "left_tracked"),
            ("right_joints", "right_tracked"),
        ]
    ):
        if not pf.get(tkey, False):
            continue
        joints = parse_hand_joints(pf[jkey])
        pos_tracking = joints[:, :3]
        quat = joints[:, 3:]

        cam_pos = world_to_camera(pos_tracking, head_pos_tracking, head_rot, eye_offset)

        cam_quat = np.array([quat_multiply(q_head_inv, quat[i]) for i in range(len(quat))])

        hand_state = np.hstack([cam_pos, cam_quat])  # (26, 7)
        offset = hand_idx * STATE_DIM_PER_HAND
        state[offset : offset + STATE_DIM_PER_HAND] = hand_state.flatten().astype(np.float32)

        if hand_idx == 0:
            left_cam_pos = cam_pos
        else:
            right_cam_pos = cam_pos

    return state, left_cam_pos, right_cam_pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Entry point: parse CLI args, process video + poses, write LeRobot dataset."""
    parser = argparse.ArgumentParser(
        description="Convert Pico recordings to a LeRobot dataset (with optional overlay video).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--poses", required=True, help="Path to JSON pose data file")
    parser.add_argument(
        "--output", required=True, type=Path, help="LeRobot dataset output directory (must not exist)"
    )
    parser.add_argument("--prompt", required=True, help="Task description for the episode")
    parser.add_argument(
        "--overlay", type=Path, default=None, help="If set, write a skeleton overlay video to this path"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Vertical FOV in degrees for projection (overlay only, default: 90)",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Seconds added to video timestamps for pose alignment (default: 0)",
    )
    parser.add_argument(
        "--tracking-origin",
        type=float,
        nargs=3,
        default=[-0.32, 2.0276, 0.0],
        metavar=("X", "Y", "Z"),
        help="XR Origin offset subtracted from head_pos (default: -0.32 2.0276 0)",
    )
    parser.add_argument(
        "--eye-offset", type=float, default=0.06, help="Horizontal eye offset in meters (default: 0.06)"
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Output dataset FPS. Defaults to the video's FPS."
    )
    parser.add_argument(
        "--overlay-codec", default="mp4v", help="FourCC codec for overlay video (default: mp4v)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    poses_path = Path(args.poses)
    output_path = args.output.resolve()

    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not poses_path.exists():
        print(f"Error: pose file not found: {poses_path}", file=sys.stderr)
        sys.exit(1)
    if output_path.exists():
        print(f"Error: output path already exists: {output_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load data ---
    with open(poses_path) as f:
        pose_data = json.load(f)
    frames_data = pose_data["frames"]
    pose_timestamps = np.array([f["t"] for f in frames_data])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    target_fps = args.fps if args.fps is not None else video_fps
    fps_int = int(round(target_fps)) or 30

    tracking_origin = np.array(args.tracking_origin)
    eye_offset = np.array([args.eye_offset, 0.0, 0.0]) if args.eye_offset != 0 else None

    # Camera intrinsics (for overlay)
    fov_rad = np.radians(args.fov)
    fy = (video_h / 2) / np.tan(fov_rad / 2)
    fx = fy
    cx, cy = video_w / 2, video_h / 2

    # Frame sampling: map output frame indices to video frame indices
    num_output_frames = max(1, int(round(total_video_frames * target_fps / video_fps)))
    sampled_indices = [
        min(int(round(i * video_fps / target_fps)), total_video_frames - 1) for i in range(num_output_frames)
    ]

    # --- Pass 1: compute states (and optionally write overlay) ---
    overlay_writer = None
    overlay_path = args.overlay.resolve() if args.overlay else None
    if overlay_path:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*args.overlay_codec)
        overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, video_fps, (video_w, video_h))

    states: list[np.ndarray] = []
    head_poses: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))

    if overlay_writer:
        next_sample = 0
        for frame_idx in range(total_video_frames):
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / video_fps + args.time_offset
            pidx = int(np.clip(np.searchsorted(pose_timestamps, video_time), 0, len(pose_timestamps) - 1))
            if pidx > 0 and abs(pose_timestamps[pidx - 1] - video_time) < abs(
                pose_timestamps[pidx] - video_time
            ):
                pidx -= 1
            pf = frames_data[pidx]

            head_pos_tracking = np.array(pf["head_pos"], dtype=np.float64) - tracking_origin
            head_rot = np.array(pf["head_rot"], dtype=np.float64)

            state, left_cam, right_cam = compute_frame_state(pf, head_pos_tracking, head_rot, eye_offset)

            # Draw overlay
            if left_cam is not None:
                px, py, valid = project_camera_to_pixel(left_cam, fx, fy, cx, cy)
                draw_hand(frame, np.stack([px, py], axis=1), valid, LEFT_HAND_COLOR)
            if right_cam is not None:
                px, py, valid = project_camera_to_pixel(right_cam, fx, fy, cx, cy)
                draw_hand(frame, np.stack([px, py], axis=1), valid, RIGHT_HAND_COLOR)
            draw_head_gizmo(frame, head_rot)
            draw_hud(
                frame,
                video_time,
                pf["t"],
                frame_idx,
                total_video_frames,
                pf.get("left_tracked", False),
                pf.get("right_tracked", False),
            )
            overlay_writer.write(frame)

            if next_sample < len(sampled_indices) and frame_idx == sampled_indices[next_sample]:
                states.append(state)
                head_poses.append(np.concatenate([head_pos_tracking, head_rot]).astype(np.float32))
                next_sample += 1

        overlay_writer.release()
        print(f"Overlay video written to {overlay_path}")
    else:
        for i in range(num_output_frames):
            fidx = sampled_indices[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break

            video_time = fidx / video_fps + args.time_offset
            pidx = int(np.clip(np.searchsorted(pose_timestamps, video_time), 0, len(pose_timestamps) - 1))
            if pidx > 0 and abs(pose_timestamps[pidx - 1] - video_time) < abs(
                pose_timestamps[pidx] - video_time
            ):
                pidx -= 1
            pf = frames_data[pidx]

            head_pos_tracking = np.array(pf["head_pos"], dtype=np.float64) - tracking_origin
            head_rot = np.array(pf["head_rot"], dtype=np.float64)

            state, _, _ = compute_frame_state(pf, head_pos_tracking, head_rot, eye_offset)
            states.append(state)
            head_poses.append(np.concatenate([head_pos_tracking, head_rot]).astype(np.float32))

    cap.release()

    num_frames = len(states)
    if num_frames == 0:
        print("Error: no frames processed.", file=sys.stderr)
        sys.exit(1)

    actions = []
    for i in range(num_frames - 1):
        delta_pos = head_poses[i + 1][:3] - head_poses[i][:3]
        delta_rot = quat_multiply(head_poses[i + 1][3:], quat_inverse(head_poses[i][3:]))
        actions.append(np.concatenate([states[i + 1], delta_pos, delta_rot]).astype(np.float32))
    delta_zero = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    actions.append(np.concatenate([states[-1], delta_zero]))

    # --- Create LeRobot dataset ---
    image_key = "observation.images.camera"
    state_key = "observation.state"
    action_key = "action"

    features = {
        image_key: {
            "dtype": "video",
            "shape": (video_h, video_w, 3),
            "names": ["height", "width", "channel"],
        },
        state_key: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": None,
        },
        action_key: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=output_path.name,
        fps=fps_int,
        root=output_path,
        robot_type="human",
        features=features,
        use_videos=True,
    )

    # --- Pass 2: read sampled frames and add to dataset ---
    print(f"Writing LeRobot dataset ({num_frames} frames, fps={fps_int}) ...")
    cap = cv2.VideoCapture(str(video_path))
    for i in range(num_frames):
        fidx = sampled_indices[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dataset.add_frame(
            {
                image_key: frame_rgb,
                state_key: states[i],
                action_key: actions[i],
                "task": args.prompt,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_frames}")
    cap.release()

    dataset.save_episode()
    print(f"LeRobot dataset written to {output_path} ({num_frames} frames, 1 episode)")


if __name__ == "__main__":
    main()
