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

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    """OpenCV-only camera factory (trimmed from upstream LeRobot)."""
    cameras: dict[str, Camera] = {}
    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)
        else:
            raise ValueError(
                f"Camera type '{cfg.type}' is not supported by the vendored SO-101 stack. Supported: opencv."
            )
    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None
