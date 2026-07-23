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

"""List OpenCV-visible cameras and save a test frame from each.

Example:
    python -m opentau.scripts.so101.find_cameras --save-dir /tmp/so101_cams
"""

import argparse
from pathlib import Path

from opentau.scripts.so101.cameras.opencv.camera_opencv import OpenCVCamera
from opentau.scripts.so101.device_paths import VIDEO_BY_PATH_DIR, stable_video_path


def find_and_capture(save_dir: Path) -> None:
    import cv2

    cameras_info = OpenCVCamera.find_cameras()
    if not cameras_info:
        print("No cameras found.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cameras_info)} camera(s) found:")
    for info in cameras_info:
        print(f"  {info}")
        cam_id = info.get("id", info.get("index"))
        stable_id = stable_video_path(cam_id)
        if stable_id is not None:
            # by-path is keyed by physical USB port; prefer it over by-id, which
            # collides when two cameras of the same model share a serial number.
            print(f"    configure this camera as: {stable_id}")
        else:
            print(f"    no {VIDEO_BY_PATH_DIR} entry; '{cam_id}' may change across reboots")
        cap = cv2.VideoCapture(cam_id)
        ok, frame = cap.read()
        if ok:
            out = save_dir / f"camera_{str(cam_id).replace('/', '_')}.png"
            cv2.imwrite(str(out), frame)
            print(f"    frame saved to {out}")
        else:
            print("    could not read a frame")
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/captured_images"),
        help="directory for the captured test frames",
    )
    args = parser.parse_args()
    find_and_capture(args.save_dir)


if __name__ == "__main__":
    main()
