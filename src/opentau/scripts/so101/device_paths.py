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

"""Resolve kernel device nodes to the stable udev symlinks that alias them.

``/dev/ttyACM*`` and ``/dev/video*`` are assigned in enumeration order, so they
shuffle across reboots and replugs. udev also publishes stable aliases:

- ``/dev/serial/by-id/`` — keyed by the adapter's USB serial number. Use it for
  motor buses: it follows an arm to whichever port it is plugged into.
- ``/dev/v4l/by-path/`` — keyed by the physical USB topology. Use it for
  cameras, *not* ``by-id``: identical camera models often share one USB serial
  number, which makes their ``by-id`` links collide.

Both directories are Linux/udev only. Every helper here degrades to "no stable
path known" on macOS and Windows rather than raising.
"""

import os
from pathlib import Path

SERIAL_BY_ID_DIR = Path("/dev/serial/by-id")
VIDEO_BY_PATH_DIR = Path("/dev/v4l/by-path")


def stable_link_map(link_dir: Path) -> dict[str, str]:
    """Map each real device node under ``link_dir`` to its stable symlink.

    Returns an empty dict when the directory does not exist (non-Linux, or a
    container without udev) or cannot be read.

    udev may publish several aliases for one node — a V4L device typically gets
    both a ``...-usb-...`` and a ``...-usbv2-...`` link. Links are visited in
    sorted order and the first one wins, so the choice is deterministic across
    runs and prefers the canonical (shorter, un-suffixed) spelling.
    """
    try:
        links = sorted(link_dir.iterdir())
    except OSError:
        return {}

    mapping: dict[str, str] = {}
    for link in links:
        try:
            target = os.path.realpath(link)
        except OSError:
            continue
        mapping.setdefault(target, str(link))
    return mapping


def _stable_path(device: str | int | None, link_dir: Path) -> str | None:
    """Resolve one device node to its stable symlink, or None if there is none."""
    # find_cameras reports bare integer indices on macOS/Windows; nothing to resolve.
    if not isinstance(device, str):
        return None
    try:
        target = os.path.realpath(device)
    except OSError:
        return None
    return stable_link_map(link_dir).get(target)


def stable_serial_path(device: str | int | None) -> str | None:
    """``/dev/ttyACM0`` -> ``/dev/serial/by-id/usb-...``, or None if unavailable."""
    return _stable_path(device, SERIAL_BY_ID_DIR)


def stable_video_path(device: str | int | None) -> str | None:
    """``/dev/video0`` -> ``/dev/v4l/by-path/pci-...``, or None if unavailable."""
    return _stable_path(device, VIDEO_BY_PATH_DIR)
