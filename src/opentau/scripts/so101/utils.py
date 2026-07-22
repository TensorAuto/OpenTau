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

"""Small terminal / timing helpers vendored from LeRobot.

Extracted from ``lerobot.utils.utils`` / ``lerobot.utils.robot_utils`` so the
hardware stack does not import their torch/accelerate-heavy parents.
"""

import logging
import platform
import select
import subprocess
import sys
import time


def init_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines: int) -> None:
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")


def say(text: str, blocking: bool = False) -> None:
    """Best-effort text-to-speech; never raises when no TTS backend exists."""
    system = platform.system()
    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")
    else:
        return
    try:
        if blocking:
            # timeout: spd-say --wait hangs indefinitely when no speech
            # dispatcher is running; never let TTS wedge the caller.
            subprocess.run(cmd, check=False, timeout=10)
        else:
            subprocess.Popen(cmd)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
    logging.info(text)
    if play_sounds:
        say(text, blocking)


def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.005) -> None:
    """Wait for ``seconds`` with better precision than time.sleep alone.

    If more than ``spin_threshold`` remains, sleep in chunks leaving
    ``sleep_margin`` before the deadline; spin for the final stretch.
    Defaults favor timing accuracy for the common 30 FPS control loop.
    """
    if seconds <= 0:
        return
    end_time = time.perf_counter() + seconds
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            break
        if remaining > spin_threshold:
            time.sleep(max(remaining - sleep_margin, 0))
        # else: spin
