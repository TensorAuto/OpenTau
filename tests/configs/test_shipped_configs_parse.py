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

"""Every JSON file under ``configs/`` must be parseable JSON.

This exists because a shipped config was committed with trailing commas and
nothing caught it. The file was valid when it was written and when it was used;
it was corrupted afterwards by running ``ruff format`` over a ``.json`` path.
Ruff parses JSON as Python — JSON is very nearly a Python literal — and its
formatter adds *magic trailing commas* when it explodes a long collection. The
result is still valid Python and no longer valid JSON, and neither
``ruff check`` nor ``ruff format --check`` complains, because as Python it is
correct.

The consequence was worse than a broken file: the config named in the PR's
own "how to try this" instructions could not be loaded, which cast doubt on a
run that had in fact happened. A parse test is the cheapest possible guard, and
it covers every config in the repo rather than just the one that broke.
"""

import json
from pathlib import Path

import pytest

import opentau

_CONFIGS_ROOT = Path(opentau.__file__).parent.parent.parent / "configs"


def _config_files() -> list[Path]:
    """Collects every JSON file shipped under ``configs/``.

    Returns:
        Sorted list of JSON paths. Empty only if the tree moved, which the
        test below turns into a failure rather than a silent pass.
    """
    return sorted(_CONFIGS_ROOT.rglob("*.json"))


def test_configs_root_is_found():
    """Guards the guard: an empty sweep must fail, not vacuously pass."""
    assert _CONFIGS_ROOT.is_dir(), f"configs/ not found at {_CONFIGS_ROOT}"
    assert _config_files(), f"no JSON configs found under {_CONFIGS_ROOT}"


@pytest.mark.parametrize("path", _config_files(), ids=lambda p: p.name)
def test_shipped_config_is_valid_json(path: Path):
    """A shipped config must parse as JSON.

    Args:
        path: The config file under test.
    """
    try:
        json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{path.relative_to(_CONFIGS_ROOT.parent)} is not valid JSON: {exc}. "
            "If this appeared after a formatting pass, check whether a Python formatter "
            "was pointed at a .json path — ruff will happily add magic trailing commas."
        )


def _sequence_config_files() -> list[Path]:
    """Shipped train configs that enable sequence training.

    Returns:
        JSON configs whose ``dataset_mixture.sequence_length`` exceeds 1 —
        the ones the sequence-stride contract applies to.
    """
    out = []
    for path in _config_files():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # the parse test above owns this failure
        if not isinstance(data, dict):
            continue  # e.g. list-shaped configs (accelerate deepspeed fragments)
        mixture = data.get("dataset_mixture") or {}
        if isinstance(mixture, dict) and (mixture.get("sequence_length") or 1) > 1:
            out.append(path)
    return out


@pytest.mark.parametrize("path", _sequence_config_files(), ids=lambda p: p.name)
def test_shipped_sequence_configs_satisfy_the_stride_contract(path: Path):
    """Every shipped sequence config must pass the stride-equals-chunk guard.

    The guard (`TrainPipelineConfig._validate_sequence_stride`) rejects
    sub-chunk strides because overlapping action targets leak into the
    teacher-forced context and collapse closed-loop rollouts. A shipped config
    that fails it dies at startup for whoever copies it — exactly what the
    JSON-parse test above exists to prevent for syntax, extended here to this
    one semantic contract (a full ``validate()`` has side effects and external
    dependencies, so only the stride guard is exercised).
    """
    import draccus

    from opentau.configs.train import TrainPipelineConfig

    cfg = draccus.parse(TrainPipelineConfig, config_path=str(path), args=[])
    cfg._validate_sequence_stride()
