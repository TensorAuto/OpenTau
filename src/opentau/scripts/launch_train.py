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
import subprocess
import sys
from pathlib import Path

import opentau.scripts.train as train_script


def main():
    parser = argparse.ArgumentParser(
        description="Launch OpenTau training with Accelerate",
        usage="opentau-train [--accelerate-config CONFIG] [TRAINING_ARGS]",
    )
    parser.add_argument(
        "--accelerate-config", type=str, help="Path to accelerate config file (yaml)", default=None
    )
    # We use parse_known_args so that all other arguments are collected
    # These will be passed to the training script
    args, unknown_args = parser.parse_known_args()

    # Base command
    cmd = ["accelerate", "launch"]

    # Add accelerate config if provided
    if args.accelerate_config:
        cmd.extend(["--config_file", args.accelerate_config])

    # Add the path to the training script
    # We resolve the path to ensure it's absolute
    train_script_path = Path(train_script.__file__).resolve()
    cmd.append(str(train_script_path))

    # Add all other arguments (passed to the training script)
    cmd.extend(unknown_args)

    # Print the command for transparency
    print(f"Executing: {' '.join(cmd)}")

    # Replace the current process with the accelerate launch command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
