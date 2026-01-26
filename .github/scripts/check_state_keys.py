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

from dataclasses import dataclass

from opentau.configs.parser import wrap

MISSING_KEYS = {
    "hf": {
        "normalize_inputs.buffer_state.max",
        "normalize_inputs.buffer_state.min",
        "normalize_targets.buffer_actions.mean",
        "normalize_targets.buffer_actions.std",
        "normalize_actions.buffer_actions.max",
        "normalize_actions.buffer_actions.min",
        "unnormalize_outputs.buffer_actions.mean",
        "unnormalize_outputs.buffer_actions.std",
        "model.paligemma_with_expert.discrete_action_embedding.weight",
        "model.paligemma_with_expert.da_head.weight",
        "model.paligemma_with_expert.da_head.bias",
    },
    "local": None,
}


@dataclass
class Arg:
    log_path: str
    source: str

    def __post_init__(self):
        if self.source not in MISSING_KEYS:
            raise ValueError(f"--source must be one of {MISSING_KEYS.keys()}. Got {self.source}")


def parse_missing_keys(log_path: str) -> list[set[str]]:
    """Parse missing keys from log file.

    The log format is:
    Missing keys when loading state dict: N keys
      - key1
      - key2
      ...
    """
    all_key_sets = []
    current_keys = None

    with open(log_path) as f:
        for line in f:
            if "Missing keys when loading state dict:" in line:
                # Start collecting keys for a new occurrence
                if current_keys is not None:
                    all_key_sets.append(current_keys)
                current_keys = set()
            elif current_keys is not None:
                # Check if line is a key entry (starts with "  - ")
                stripped = line.strip()
                if stripped.startswith("- "):
                    key = stripped[2:].strip()
                    current_keys.add(key)
                elif stripped and not stripped.startswith("-"):
                    # Non-empty line that's not a key entry means section ended
                    all_key_sets.append(current_keys)
                    current_keys = None

    # Don't forget the last set if file ended while collecting
    if current_keys is not None:
        all_key_sets.append(current_keys)

    return all_key_sets


def check_no_unexpected_keys(log_path: str):
    """Check that 'Unexpected keys when loading state dict:' does not appear in the log."""
    print("Checking for unexpected keys")
    with open(log_path) as f:
        for line in f:
            if "Unexpected keys when loading state dict:" in line:
                raise ValueError(f"Found unexpected keys in log: {line.strip()}")
    print("Passed - no unexpected keys found")


def check_missing_keys(key_sets: list[set[str]], source: str):
    """Check that all missing key sets match the expected keys."""
    print("Checking missing keys")
    expected_keys = MISSING_KEYS[source]

    if expected_keys is None:
        if key_sets:
            raise ValueError(f"Found missing keys but expecting none: {key_sets}")
    elif not key_sets:
        raise ValueError(f"No missing keys found, should be {expected_keys}")
    else:
        for i, keys in enumerate(key_sets):
            if keys != expected_keys:
                missing_from_expected = expected_keys - keys
                extra_in_found = keys - expected_keys
                raise ValueError(
                    f"Missing keys mismatch at occurrence {i + 1}:\n"
                    f"  Expected but not found: {missing_from_expected}\n"
                    f"  Found but not expected: {extra_in_found}"
                )
    print("Passed")


@wrap()
def main(arg: Arg) -> None:
    # Check that no unexpected keys appear
    check_no_unexpected_keys(arg.log_path)

    # Parse and check missing keys
    key_sets = parse_missing_keys(arg.log_path)
    check_missing_keys(key_sets, arg.source)


if __name__ == "__main__":
    main()
