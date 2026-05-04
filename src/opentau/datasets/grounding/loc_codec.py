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

"""Codec between pixel coordinates and PaliGemma `<locNNNN>` strings.

PaliGemma's 1024-bin grounding format quantizes a coordinate axis into
10 bits and emits each bin as a single `<locNNNN>` token (zero-padded to
four digits — `<loc23>` does not match the tokenizer). A bounding box is
encoded as four loc tokens in **(y_min, x_min, y_max, x_max)** order
(y-then-x; do not swap), then a space, then the label, and `; ` separates
multiple boxes:

    "<loc0234><loc0567><loc0890><loc1023> dog ; <loc0010><loc0050><loc0200><loc0500> cat"

A point is two loc tokens in **(y, x)** order:

    "<loc0234><loc0567> spout"

The 1024 grid is **abstract** — it is not the input image resolution.
Coordinates are normalized using the original image dimensions, then
quantized as ``int(round(coord_norm * 1023))`` and clamped to ``[0, 1023]``.
Pass the original image's `(width, height)` from the dataset (e.g.
``Image.open(...).size``), NOT the post-resize tensor shape that the
policy actually consumes.

TODO: eval-side decoding will use ``loc_tokens_to_xyxy`` /
``loc_tokens_to_points`` against decoded response strings to recover
bounding boxes for IoU/mAP. Tracked as a follow-up to the configurable
response-formatter work.
"""

from __future__ import annotations

import re

NUM_BINS = 1024
MAX_BIN = NUM_BINS - 1

_LOC_TOKEN_RE = re.compile(r"<loc(\d{4})>")

# Segment separator the encoder emits between adjacent box/point entries
# (e.g. ``"<...> dog ; <...> cat"``). Decoders split on this so that a
# malformed segment cannot misalign every subsequent one.
SEGMENT_SEPARATOR = ";"


def _quantize(coord: float, extent: float) -> int:
    """Map a single pixel coordinate to a `[0, 1023]` bin index.

    Args:
        coord: Pixel coordinate (e.g. an x or y in original-image space).
        extent: The image dimension along this axis (width for x, height for y).

    Returns:
        An integer bin index in ``[0, 1023]``.
    """
    if extent <= 0:
        return 0
    bin_idx = int(round((coord / extent) * MAX_BIN))
    if bin_idx < 0:
        return 0
    if bin_idx > MAX_BIN:
        return MAX_BIN
    return bin_idx


def _dequantize(bin_idx: int, extent: float) -> float:
    """Inverse of `_quantize`: map a bin index back to a pixel coordinate."""
    return (bin_idx / MAX_BIN) * extent


def _loc(bin_idx: int) -> str:
    return f"<loc{bin_idx:04d}>"


def xyxy_to_loc_tokens(box_xyxy: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    """Encode an `(x_min, y_min, x_max, y_max)` box as four loc tokens.

    The output order is `<loc Y_min><loc X_min><loc Y_max><loc X_max>`
    (y-then-x), matching PaliGemma's convention.

    Args:
        box_xyxy: ``(x_min, y_min, x_max, y_max)`` in pixel coordinates of the
            **original** image.
        img_w: Original image width in pixels.
        img_h: Original image height in pixels.

    Returns:
        A four-token string with no separators.
    """
    x_min, y_min, x_max, y_max = box_xyxy
    return (
        _loc(_quantize(y_min, img_h))
        + _loc(_quantize(x_min, img_w))
        + _loc(_quantize(y_max, img_h))
        + _loc(_quantize(x_max, img_w))
    )


def xywh_to_loc_tokens(box_xywh: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    """Same as `xyxy_to_loc_tokens` but accepts COCO-style ``(x, y, w, h)``."""
    x, y, w, h = box_xywh
    return xyxy_to_loc_tokens((x, y, x + w, y + h), img_w, img_h)


def point_to_loc_tokens(x: float, y: float, img_w: int, img_h: int) -> str:
    """Encode an `(x, y)` point as two loc tokens in y-then-x order."""
    return _loc(_quantize(y, img_h)) + _loc(_quantize(x, img_w))


def loc_tokens_to_xyxy(s: str, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Parse a string of loc tokens into `(x_min, y_min, x_max, y_max)` pixel boxes.

    Tolerant and segment-aware: the input is split on the encoder's segment
    separator (``;``), and each segment must contribute exactly four loc
    tokens to yield a box. A segment with any other count (0, 1, 2, 3, 5,
    ...) is dropped silently — its tokens do NOT spill into the next
    segment, so a single malformed box cannot misalign every subsequent one.
    Garbage strings or partial decodes return ``[]``.

    Args:
        s: A string that may contain `<locNNNN>` tokens, e.g. a decoded
            response. Non-loc text within a segment is ignored.
        img_w: Original image width in pixels.
        img_h: Original image height in pixels.

    Returns:
        A list of `(x_min, y_min, x_max, y_max)` tuples in pixel coordinates.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for segment in s.split(SEGMENT_SEPARATOR):
        bins = [int(m) for m in _LOC_TOKEN_RE.findall(segment)]
        if len(bins) != 4:
            continue
        y_min_b, x_min_b, y_max_b, x_max_b = bins
        boxes.append(
            (
                _dequantize(x_min_b, img_w),
                _dequantize(y_min_b, img_h),
                _dequantize(x_max_b, img_w),
                _dequantize(y_max_b, img_h),
            )
        )
    return boxes


def loc_tokens_to_points(s: str, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Parse a string of loc tokens into `(x, y)` pixel points.

    Tolerant and segment-aware in the same sense as `loc_tokens_to_xyxy`:
    the input is split on ``;``, and each segment must contribute exactly
    two loc tokens (in `(y, x)` order per the PaliGemma convention) to
    yield a point. Segments with any other count are dropped — a malformed
    segment cannot shift later ones.
    """
    points: list[tuple[float, float]] = []
    for segment in s.split(SEGMENT_SEPARATOR):
        bins = [int(m) for m in _LOC_TOKEN_RE.findall(segment)]
        if len(bins) != 2:
            continue
        y_b, x_b = bins
        points.append((_dequantize(x_b, img_w), _dequantize(y_b, img_h)))
    return points
