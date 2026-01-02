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

import argparse
import logging
import os

from dotenv import load_dotenv
from PIL import Image

from opentau.planner import NavHighLevelPlanner
from opentau.utils.utils import (
    init_logging,
)

load_dotenv()


def main(img_dir_path):
    frames = sorted(os.listdir(img_dir_path))
    logging.info("Loading the frames")
    img_dict1 = {}
    for i, image_path in enumerate(frames):
        img = Image.open(img_dir_path + "/" + image_path).convert("RGB")
        img_dict1[i] = img

    # dummy instructions
    task = "The goal is to reach till fridge"
    nav_planner = NavHighLevelPlanner()
    logging.info("Inferencing the navigational planner")
    actions = nav_planner.inference(image_dict=img_dict1, model_name="gpt4o", task=task, mem=None)

    logging.info(f"The instructions are {actions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the navigation high level planner with a specified image directory."
    )

    # 2. Add the --img_path argument
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the directory containing the image frames."
    )

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    init_logging()
    main(args.img_path)
