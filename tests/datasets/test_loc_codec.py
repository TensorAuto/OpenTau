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

"""Unit tests for the loc-token codec — pure functions, no network."""

from __future__ import annotations

from opentau.datasets.grounding.loc_codec import (
    NUM_BINS,
    loc_tokens_to_points,
    loc_tokens_to_xyxy,
    point_to_loc_tokens,
    xywh_to_loc_tokens,
    xyxy_to_loc_tokens,
)


# Slack equal to one bin step in pixel units, plus a hair for the round-trip
# `int(round(...))` quantization. (img_dim / 1023) * 1.0 is one quantum;
# we allow 0.55 of that on either side.
def _tol(extent: int) -> float:
    return extent / (NUM_BINS - 1) * 0.55


def test_xyxy_round_trip_integer_aligned() -> None:
    img_w, img_h = 1024, 1024  # one pixel per bin: round-trip is exact.
    box = (100.0, 200.0, 800.0, 900.0)
    s = xyxy_to_loc_tokens(box, img_w, img_h)
    assert s.count("<loc") == 4

    decoded = loc_tokens_to_xyxy(s, img_w, img_h)
    assert len(decoded) == 1
    x_min, y_min, x_max, y_max = decoded[0]
    assert abs(x_min - 100.0) <= _tol(img_w)
    assert abs(y_min - 200.0) <= _tol(img_h)
    assert abs(x_max - 800.0) <= _tol(img_w)
    assert abs(y_max - 900.0) <= _tol(img_h)


def test_xywh_matches_xyxy() -> None:
    img_w, img_h = 640, 480
    s_xywh = xywh_to_loc_tokens((50.0, 60.0, 100.0, 120.0), img_w, img_h)
    s_xyxy = xyxy_to_loc_tokens((50.0, 60.0, 150.0, 180.0), img_w, img_h)
    assert s_xywh == s_xyxy


def test_point_round_trip() -> None:
    img_w, img_h = 1920, 1080
    s = point_to_loc_tokens(960.0, 540.0, img_w, img_h)
    assert s.count("<loc") == 2

    decoded = loc_tokens_to_points(s, img_w, img_h)
    assert len(decoded) == 1
    x, y = decoded[0]
    assert abs(x - 960.0) <= _tol(img_w)
    assert abs(y - 540.0) <= _tol(img_h)


def test_y_then_x_order() -> None:
    """Bin order matches PaliGemma's y-then-x convention.

    For an asymmetric image (width != height), swapping x and y in the input
    must produce a different bin output — proving we are not silently treating
    the two coordinates as interchangeable.
    """
    img_w, img_h = 1000, 500
    a = point_to_loc_tokens(100.0, 200.0, img_w, img_h)
    b = point_to_loc_tokens(200.0, 100.0, img_w, img_h)
    assert a != b


def test_clamping_negative_and_overflow() -> None:
    img_w, img_h = 100, 100
    s = xyxy_to_loc_tokens((-50.0, -10.0, 5_000.0, 999.0), img_w, img_h)
    # Lowest bin is <loc0000>; highest is <loc1023>. The clamped string
    # should start with two <loc0000> tokens and end with two <loc1023>'s.
    assert s.startswith("<loc0000><loc0000>")
    assert s.endswith("<loc1023><loc1023>")


def test_multi_box_concat() -> None:
    img_w, img_h = 1024, 1024
    box1 = xyxy_to_loc_tokens((10.0, 20.0, 30.0, 40.0), img_w, img_h)
    box2 = xyxy_to_loc_tokens((100.0, 200.0, 300.0, 400.0), img_w, img_h)
    response = f"{box1} dog ; {box2} cat"

    decoded = loc_tokens_to_xyxy(response, img_w, img_h)
    assert len(decoded) == 2
    # First box
    x_min, y_min, x_max, y_max = decoded[0]
    assert abs(x_min - 10.0) <= _tol(img_w)
    assert abs(y_min - 20.0) <= _tol(img_h)
    # Second box
    x_min, y_min, x_max, y_max = decoded[1]
    assert abs(x_min - 100.0) <= _tol(img_w)
    assert abs(y_min - 200.0) <= _tol(img_h)


def test_garbage_input_returns_empty() -> None:
    assert loc_tokens_to_xyxy("garbage with no tokens", 100, 100) == []
    assert loc_tokens_to_points("garbage with no tokens", 100, 100) == []


def test_segment_with_non_four_token_count_is_dropped() -> None:
    """A single segment with anything other than 4 loc tokens yields 0 boxes.

    The decoder is segment-aware (splits on ``;``); a malformed segment is
    dropped silently rather than spilling its tokens into a neighbour. Six
    loc tokens in one segment is malformed and produces 0 boxes — not 1
    box plus two orphans, which would be the buggy behaviour of a global
    pairing scheme.
    """
    img_w, img_h = 1024, 1024
    response = "<loc0010><loc0020><loc0030><loc0040><loc0500><loc0600>"
    boxes = loc_tokens_to_xyxy(response, img_w, img_h)
    assert boxes == []


def test_malformed_segment_does_not_misalign_following_segments() -> None:
    """Regression: a 5-loc-token segment must NOT shift the boundary of the next one.

    If the parser collected loc tokens globally and grouped them in fours,
    the second box would absorb the orphan from the first segment and end
    up encoding completely wrong coordinates — silent box-to-label
    misattribution at eval time. Segment-aware parsing drops the bad
    segment and decodes the next one cleanly.
    """
    img_w, img_h = 1024, 1024
    good_box = xyxy_to_loc_tokens((100.0, 200.0, 300.0, 400.0), img_w, img_h)
    # 5 loc tokens in the first segment — malformed.
    bad_segment = "<loc0010><loc0020><loc0030><loc0040><loc0500> dog"
    response = f"{bad_segment} ; {good_box} cat"

    decoded = loc_tokens_to_xyxy(response, img_w, img_h)
    assert len(decoded) == 1
    x_min, y_min, x_max, y_max = decoded[0]
    assert abs(x_min - 100.0) <= _tol(img_w)
    assert abs(y_min - 200.0) <= _tol(img_h)
    assert abs(x_max - 300.0) <= _tol(img_w)
    assert abs(y_max - 400.0) <= _tol(img_h)


def test_points_segment_aware() -> None:
    """`loc_tokens_to_points` is segment-aware — a 3-token bad segment doesn't shift the next."""
    img_w, img_h = 1024, 1024
    good_point = point_to_loc_tokens(500.0, 500.0, img_w, img_h)
    response = f"<loc0010><loc0020><loc0030> noise ; {good_point} target"

    decoded = loc_tokens_to_points(response, img_w, img_h)
    assert len(decoded) == 1
    x, y = decoded[0]
    assert abs(x - 500.0) <= _tol(img_w)
    assert abs(y - 500.0) <= _tol(img_h)


def test_codec_uses_original_image_dims_not_post_resize() -> None:
    """Regression: bins must be computed against the original image dims.

    A common mistake is to pass the post-resize tensor shape (e.g. 224, 224)
    that the policy actually consumes. The codec must use the original
    `(img_w, img_h)` from the dataset so loc tokens carry the same spatial
    meaning regardless of input pipeline resizing.
    """
    # Same pixel coordinate, different "original" extents.
    bin_at_1920 = xyxy_to_loc_tokens((960.0, 540.0, 960.0, 540.0), 1920, 1080)
    bin_at_224 = xyxy_to_loc_tokens((960.0, 540.0, 960.0, 540.0), 224, 224)
    # If the codec ignored extent, both would tokenize to the same string;
    # they must not.
    assert bin_at_1920 != bin_at_224


def test_format_string_is_zero_padded_to_four_digits() -> None:
    """`<loc23>` does not match the PaliGemma tokenizer; only `<loc0023>` does."""
    img_w, img_h = 102_400, 102_400  # 100 px / bin -> bin 0 for coord 23
    s = xyxy_to_loc_tokens((23.0, 23.0, 23.0, 23.0), img_w, img_h)
    # We don't care about the exact bin index — only that every emitted token
    # is exactly 9 characters long: "<loc" + 4 digits + ">".
    for tok in s.replace(">", ">|").split("|"):
        if not tok:
            continue
        assert tok.startswith("<loc")
        assert tok.endswith(">")
        assert len(tok) == len("<loc0000>")
