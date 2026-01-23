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

import math
from dataclasses import dataclass

import numpy as np
from utils import grep_file

from opentau.configs.parser import wrap


@dataclass
class Arg:
    log_path: str
    expected_length: int
    re_pattern: str = r"mse_loss:([0-9.eE+-]+)"
    gauss_sigma: float = 4.0
    gauss_truncate: float = 4.0
    pad_mode: str = "reflect"
    resume_log_path: str | None = None
    resume_expected_length: int | None = None


def gaussian_smooth(
    series: list[float], sigma: float, *, truncate: float = 4.0, mode: str = "reflect"
) -> list[float]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    x = np.asarray(series, dtype=np.float64)

    radius = int(math.ceil(truncate * sigma))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(k**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # normalize

    pad_width = (radius, radius)
    x_padded = np.pad(x, pad_width, mode=mode)
    smoothed = np.convolve(x_padded, kernel, mode="valid")

    return smoothed.tolist()


def check_smooth_loss(losses: list[float], expected_length: int, arg: Arg, prefix: str) -> list[float]:
    print(f"{prefix} raw losses:", losses)
    assert len(losses) == expected_length, (
        f"Expected {expected_length} losses, found {len(losses)} in {arg.log_path}."
    )
    smoothed = gaussian_smooth(losses, arg.gauss_sigma, truncate=arg.gauss_truncate, mode=arg.pad_mode)
    print(f"{prefix} smoothed losses:", smoothed)
    assert smoothed[0] >= smoothed[-1], "Losses should drop over time when smoothed."
    return smoothed


@wrap()
def main(arg: Arg):
    losses = grep_file(arg.log_path, arg.re_pattern, processor=float)
    smoothed = check_smooth_loss(losses, arg.expected_length, arg, "Training")

    if arg.resume_expected_length is None and arg.resume_log_path is None:
        return

    if arg.resume_expected_length is None or arg.resume_log_path is None:
        raise ValueError(
            "Both resume_log_path and resume_expected_length must be provided if one is given. "
            f"Got resume_log_path: {arg.resume_log_path}, "
            f"Got resume_expected_length: {arg.resume_expected_length}, "
        )

    resume_losses = grep_file(arg.resume_log_path, arg.re_pattern, processor=float)
    resume_smoothed = check_smooth_loss(resume_losses, arg.resume_expected_length, arg, "Resume")

    # resuming start should be closer to the end of the training than the start
    resume_start = resume_smoothed[0]
    training_start = smoothed[0]
    training_end = smoothed[-1]
    print(
        f"{resume_start=}, {training_start=}, {training_end=}, "
        f"{abs(resume_start - training_end)=}, {abs(resume_start - training_start)=}."
    )
    assert abs(resume_start - training_end) <= abs(resume_start - training_start), (
        "Resuming start loss should be closer to the end of the training than the start."
    )


if __name__ == "__main__":
    main()
