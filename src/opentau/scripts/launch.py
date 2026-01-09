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
from types import ModuleType

import opentau.scripts.eval as eval_script
import opentau.scripts.export_to_onnx as export_script
import opentau.scripts.train as train_script


def launch(script_module: ModuleType, description: str, use_accelerate: bool = True):
    """Generic launcher for OpenTau scripts using Accelerate or Python."""
    parser = argparse.ArgumentParser(
        description=description,
        usage=f"{Path(sys.argv[0]).name} {'[--accelerate-config CONFIG] ' if use_accelerate else ''}[ARGS]",
    )
    if use_accelerate:
        parser.add_argument(
            "--accelerate-config", type=str, help="Path to accelerate config file (yaml)", default=None
        )

    # We use parse_known_args so that all other arguments are collected
    # These will be passed to the target script
    args, unknown_args = parser.parse_known_args()

    # Base command
    if use_accelerate:
        cmd = ["accelerate", "launch"]
        # Add accelerate config if provided
        if args.accelerate_config:
            cmd.extend(["--config_file", args.accelerate_config])
    else:
        cmd = [sys.executable]

    # Add the path to the target script
    # We resolve the path to ensure it's absolute
    script_path = Path(script_module.__file__).resolve()
    cmd.append(str(script_path))

    # Add all other arguments (passed to the target script)
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


def train():
    launch(train_script, "Launch OpenTau training with Accelerate")


def eval():
    launch(eval_script, "Launch OpenTau evaluation with Accelerate")


def export():
    launch(export_script, "Launch OpenTau ONNX export", use_accelerate=False)
