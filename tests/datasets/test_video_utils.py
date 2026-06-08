# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the torchcodec probe and explicit-backend fallback paths in video_utils.

These cover the behavior introduced in the torchcodec 0.4 -> 0.10 bump:

- the probe ``_load_torchcodec_videodecoder`` returning ``(None, exc)`` when
  torchcodec can't be imported,
- ``get_safe_default_codec`` reflecting that fallback,
- ``decode_video_frames_torchcodec`` raising an ``ImportError`` that chains
  the original probe exception,
- ``decode_video_frames(backend="torchcodec")`` silently routing to pyav and
  emitting the explicit-fallback warning at most once per process.
"""

import importlib.util
import logging
import shutil
import subprocess

import pytest

from opentau.datasets import video_utils


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Reset every ``@functools.cache``'d probe/sentinel in video_utils.

    The probe and the explicit-fallback warning sentinel are process-lifetime
    caches; without an explicit reset, the state set by one test leaks into
    the next.
    """
    video_utils._load_torchcodec_videodecoder.cache_clear()
    video_utils._warn_explicit_torchcodec_unloadable.cache_clear()
    yield
    video_utils._load_torchcodec_videodecoder.cache_clear()
    video_utils._warn_explicit_torchcodec_unloadable.cache_clear()


@pytest.fixture
def torchcodec_missing(monkeypatch):
    """Make ``importlib.util.find_spec('torchcodec')`` report the package as missing.

    The probe short-circuits before it can ``import torchcodec.decoders``, so
    this works regardless of whether torchcodec is actually installed in the
    test environment.
    """
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None
        if name == "torchcodec"
        else original_find_spec(name, *args, **kwargs),
    )


def _make_solid_color_mp4(path, *, fps=10, duration=2.0, width=64, height=48):
    """Create a tiny single-color MP4 via ffmpeg. Skips the test if ffmpeg is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={width}x{height}:r={fps}:d={duration:.6f}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-g",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_probe_returns_module_not_found_when_torchcodec_missing(torchcodec_missing):
    cls, exc = video_utils._load_torchcodec_videodecoder()
    assert cls is None
    assert isinstance(exc, ModuleNotFoundError)


def test_get_safe_default_codec_falls_back_to_pyav_when_missing(torchcodec_missing):
    assert video_utils.get_safe_default_codec() == "pyav"


def test_decode_video_frames_torchcodec_raises_chained_when_missing(torchcodec_missing):
    with pytest.raises(ImportError, match="ModuleNotFoundError") as exc_info:
        video_utils.decode_video_frames_torchcodec("nonexistent.mp4", [0.0], 0.1)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_explicit_torchcodec_falls_back_to_pyav_with_single_warning(tmp_path, caplog, torchcodec_missing):
    """``decode_video_frames(..., backend='torchcodec')`` must downgrade to pyav
    AND emit the explicit-fallback warning at most once per process — even when
    the function is called many times (matches the dataloader hot path).
    """
    video_path = _make_solid_color_mp4(tmp_path / "test.mp4")
    timestamps = [0.1, 0.5, 1.0]

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            frames = video_utils.decode_video_frames(
                str(video_path), timestamps, tolerance_s=0.1, backend="torchcodec"
            )
            # pyav returned tensor shape: (T, C, H, W)
            assert frames.shape == (len(timestamps), 3, 48, 64)

    fallback_warnings = [
        r for r in caplog.records if "requested backend='torchcodec' but it failed to load" in r.getMessage()
    ]
    assert len(fallback_warnings) == 1, (
        f"expected exactly one explicit-fallback warning across 3 calls, got {len(fallback_warnings)}"
    )


def test_torchcodec_clamps_out_of_range_frame_index(monkeypatch):
    """A query timestamp mapping past the last frame is clamped to ``num_frames-1``.

    Consolidated v3.0 videos are occasionally encoded with marginally fewer frames
    than their metadata implies, so ``round(ts * average_fps)`` can land at/just
    past ``num_frames``. ``torchcodec.get_frames_at`` raises a hard ``IndexError``
    on such an index; the loader must instead clamp to the nearest decodable frame
    (matching the pyav path), letting the tolerance check decide acceptance.
    """
    import torch

    captured = {}
    num_frames, avg_fps = 100, 20.0

    class _FakeBatch:
        def __init__(self, indices):
            self.data = [torch.zeros(3, 4, 4, dtype=torch.uint8) for _ in indices]
            self.pts_seconds = [torch.tensor(i / avg_fps) for i in indices]

    class _FakeDecoder:
        def __init__(self, *a, **k):
            self.metadata = type("M", (), {"num_frames": num_frames, "average_fps": avg_fps})()

        def get_frames_at(self, indices):
            captured["indices"] = list(indices)
            return _FakeBatch(indices)

    monkeypatch.setattr(video_utils, "_load_torchcodec_videodecoder", lambda: (_FakeDecoder, None))

    # ts = num_frames / avg_fps -> round() == num_frames (out of range by one).
    ts = num_frames / avg_fps
    frames = video_utils.decode_video_frames_torchcodec("fake.mp4", [ts], tolerance_s=0.5)

    assert captured["indices"] == [num_frames - 1], f"index not clamped: {captured['indices']}"
    assert frames.shape[0] == 1
