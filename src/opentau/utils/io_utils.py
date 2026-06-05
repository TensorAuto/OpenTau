#!/usr/bin/env python

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
"""Utilities for file I/O operations.

This module provides functions for reading and writing JSON files, saving videos,
and deserializing JSON data into structured objects with type checking. It also
provides :func:`silence_output_unless_error` for muting noisy subprocess output
(captured and replayed only if the wrapped block raises).
"""

import contextlib
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import imageio

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


@contextlib.contextmanager
def silence_output_unless_error(label: str = "") -> Iterator[None]:
    r"""Mute this process's stdout/stderr for the block, replaying it only on error.

    Redirection happens at the **file-descriptor** level (``os.dup2`` on fds 1 and
    2), not by swapping ``sys.stdout`` / ``sys.stderr``. That is what lets it capture
    output a Python-level :func:`contextlib.redirect_stdout` would miss: writes from
    C extensions (mujoco / EGL) and ``logging`` handlers that grabbed a reference to
    the original stream at import time (e.g. robosuite's logger).

    The motivating use is muting the noisy one-per-worker robosuite/robocasa
    import-and-construction banner emitted inside ``gym.vector.AsyncVectorEnv`` spawn
    workers during sim eval — ``n_envs * world_size`` duplicate copies of the
    private-macro / mink / mimicgen / controller-config lines on every eval. On
    success the captured output is discarded; if the wrapped block raises, the
    captured bytes are written to the real stderr first (prefixed with ``label``) so
    the failing worker stays debuggable, then the exception propagates unchanged.

    Args:
        label: Optional identifier (e.g. ``"task=CloseFridge idx=3"``) prepended to
            the replayed output so a failing worker can be attributed.

    Yields:
        None. Wrap the code whose output should be muted in the ``with`` block.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    failed = False
    try:
        with tempfile.TemporaryFile(mode="w+b") as buffer:
            try:
                # Inside the try so a failure of the second dup2 (fd 1 already
                # redirected) still hits the finally and restores both fds.
                os.dup2(buffer.fileno(), 1)
                os.dup2(buffer.fileno(), 2)
                yield
            except BaseException:
                failed = True
                raise
            finally:
                # Restore the real stdout/stderr fds whether or not the block raised,
                # then (on failure only) replay the captured bytes so they are not lost.
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                if failed:
                    buffer.seek(0)
                    captured = buffer.read()
                    if captured:
                        prefix = f"[silenced output — {label}]\n" if label else "[silenced output]\n"
                        payload = prefix.encode(errors="replace") + captured
                        while payload:
                            payload = payload[os.write(saved_stderr_fd, payload) :]
    finally:
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def write_video(video_path: str | Path, stacked_frames: list, fps: float) -> None:
    """Write a list of frames to a video file.

    Args:
        video_path: Path where the video file will be saved.
        stacked_frames: List of image frames to write.
        fps: Frames per second for the output video.
    """
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """Load JSON data and recursively fill an object with matching structure.

    Loads the JSON data from fpath and recursively fills obj with the
    corresponding values (strictly matching structure and types).
    Tuples in obj are expected to be lists in the JSON data, which will be
    converted back into tuples.

    Args:
        fpath: Path to the JSON file to load.
        obj: Template object with the desired structure and types.

    Returns:
        Object with the same structure as obj, filled with values from the JSON file.

    Raises:
        TypeError: If structure or types don't match between JSON and obj.
        ValueError: If dictionary keys or list/tuple lengths don't match.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        Recursively overwrite the structure in `target` with data from `source`,
        performing strict checks on structure and type.
        Returns the updated version of `target` (especially important for tuples).
        """

        # If the target is a dictionary, source must be a dictionary as well.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # Check that they have exactly the same set of keys.
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # Recursively update each key.
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # If the target is a list, source must be a list as well.
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # Check length
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # Recursively update each element.
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # If the target is a tuple, the source must be a list in JSON,
        # which we'll convert back to a tuple.
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # Convert each element, forming a new tuple.
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # Return a brand new tuple (tuples are immutable in Python).
            return tuple(converted_items)

        # Otherwise, we're dealing with a "primitive" (int, float, str, bool, None).
        else:
            # Check the exact type.  If these must match 1:1, do:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # Perform the in-place/recursive deserialization
    updated_obj = _deserialize(obj, data)
    return updated_obj
