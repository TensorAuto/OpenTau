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

"""Interactively find the serial port of a motor bus (unplug/replug).

Example:
    python -m opentau.scripts.so101.find_port
"""

import os
import platform
import time
from pathlib import Path

from opentau.scripts.so101.device_paths import SERIAL_BY_ID_DIR, stable_link_map


def find_available_ports() -> list[str]:
    if platform.system() == "Windows":
        from serial.tools import list_ports

        return [port.device for port in list_ports.comports()]
    return [str(path) for path in Path("/dev").glob("tty*")]


def find_port() -> None:
    print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    # A by-id symlink is removed along with its device, so snapshot the mapping
    # now, while the bus we are about to identify is still plugged in.
    links_before = stable_link_map(SERIAL_BY_ID_DIR)
    print("Ports before disconnecting:", ports_before)

    print("Remove the USB cable from your MotorsBus and press Enter when done.")
    input()
    time.sleep(0.5)

    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        stable_port = links_before.get(os.path.realpath(port))
        if stable_port is not None:
            print(f"The port of this MotorsBus is '{stable_port}'")
            print(f"Use that path, not '{port}' — ACM/USB numbering shuffles across reboots.")
        else:
            print(f"The port of this MotorsBus is '{port}'")
            print(
                f"No {SERIAL_BY_ID_DIR} entry points at it, so this path may change "
                "across reboots — re-run this script if the bus stops connecting."
            )
        print("Reconnect the USB cable.")
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def main() -> None:
    find_port()


if __name__ == "__main__":
    main()
