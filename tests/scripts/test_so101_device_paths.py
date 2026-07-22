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

"""Tests for udev stable-path resolution used by find_port / find_cameras."""

import os
from pathlib import Path

from opentau.scripts.so101 import device_paths


def _make_dev_tree(root: Path, nodes: list[str], links: dict[str, str]) -> tuple[Path, Path]:
    """Build a fake /dev with ``nodes`` device files and ``links`` name -> node."""
    dev = root / "dev"
    dev.mkdir()
    for node in nodes:
        (dev / node).write_text("")

    link_dir = dev / "by-something"
    link_dir.mkdir()
    for name, node in links.items():
        (link_dir / name).symlink_to(dev / node)
    return dev, link_dir


def test_stable_link_map_resolves_symlinks(tmp_path):
    dev, link_dir = _make_dev_tree(
        tmp_path,
        nodes=["ttyACM0", "ttyACM1"],
        links={"usb-Serial_AAAA-if00": "ttyACM0", "usb-Serial_BBBB-if00": "ttyACM1"},
    )

    mapping = device_paths.stable_link_map(link_dir)

    assert mapping == {
        os.path.realpath(dev / "ttyACM0"): str(link_dir / "usb-Serial_AAAA-if00"),
        os.path.realpath(dev / "ttyACM1"): str(link_dir / "usb-Serial_BBBB-if00"),
    }


def test_stable_link_map_picks_first_alias_deterministically(tmp_path):
    """udev publishes both a `-usb-` and a `-usbv2-` alias for one V4L node."""
    dev, link_dir = _make_dev_tree(
        tmp_path,
        nodes=["video0"],
        links={
            "pci-0000:80:14.0-usb-0:2.2:1.0-video-index0": "video0",
            "pci-0000:80:14.0-usbv2-0:2.2:1.0-video-index0": "video0",
        },
    )

    mapping = device_paths.stable_link_map(link_dir)

    # Sorted order puts the canonical `-usb-` spelling first ('-' < 'v').
    assert mapping == {
        os.path.realpath(dev / "video0"): str(link_dir / "pci-0000:80:14.0-usb-0:2.2:1.0-video-index0")
    }


def test_stable_link_map_missing_dir_is_empty(tmp_path):
    """macOS / Windows / a container without udev must not raise."""
    assert device_paths.stable_link_map(tmp_path / "nonexistent") == {}


def test_stable_serial_path_resolves_and_misses(tmp_path, monkeypatch):
    dev, link_dir = _make_dev_tree(
        tmp_path,
        nodes=["ttyACM0", "ttyACM1"],
        links={"usb-Serial_AAAA-if00": "ttyACM0"},
    )
    monkeypatch.setattr(device_paths, "SERIAL_BY_ID_DIR", link_dir)

    assert device_paths.stable_serial_path(str(dev / "ttyACM0")) == str(link_dir / "usb-Serial_AAAA-if00")
    # A node with no by-id link resolves to nothing rather than a wrong guess.
    assert device_paths.stable_serial_path(str(dev / "ttyACM1")) is None


def test_stable_video_path_ignores_non_path_ids(tmp_path, monkeypatch):
    """find_cameras reports bare integer indices off Linux."""
    _, link_dir = _make_dev_tree(tmp_path, nodes=["video0"], links={"pci-0:1.0-video-index0": "video0"})
    monkeypatch.setattr(device_paths, "VIDEO_BY_PATH_DIR", link_dir)

    assert device_paths.stable_video_path(0) is None
    assert device_paths.stable_video_path(None) is None
