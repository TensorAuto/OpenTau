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

import time
from dataclasses import dataclass
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.utils.download_utils import check_libero_dataset, libero_dataset_download

from opentau.configs import parser


@dataclass
class Args:
    suite: str | None = None
    download_dir: str = get_libero_path("datasets")

    def __post_init__(self):
        if self.suite not in [None, "object", "spatial", "goal", "10", "90"]:
            raise ValueError(
                f"Invalid suite: {self.suite}. Available suites are: 'object', 'spatial', 'goal', '10', or '90'."
            )


@parser.wrap()
def main(args: Args):
    # Ask users to specify the download directory of datasets
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    download_dir = str(download_dir.resolve())
    print("Datasets will be downloaded to:", download_dir)

    datasets = "all" if args.suite is None else f"libero_{args.suite}"
    print("Datasets to download:", datasets)

    libero_dataset_download(datasets=datasets, download_dir=download_dir, use_huggingface=True)
    time.sleep(1)
    check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
