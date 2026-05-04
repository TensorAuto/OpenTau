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
"""Resolve ``$ref`` includes inside JSON config files.

Lets a JSON config split itself across multiple files. Anywhere a JSON value
is expected, an object of the form ``{"$ref": "path/to/other.json"}`` loads
that file's content and inlines it. Sibling keys deep-merge over the loaded
content (sibling values win on conflict), enabling small, targeted overrides
of a referenced fragment.

Example:

.. code-block:: json

    // main.json
    {
        "dataset_mixture": {
            "$ref": "mixtures/big_mixture.json",
            "weights": [0.5, 0.5]
        }
    }

Rules:

- ``$ref`` value must be a string. Relative paths resolve against the file
  that contains the ``$ref`` (not CWD).
- Refs may appear at any depth, in dicts or list elements.
- Refs may chain (a referenced file may itself contain ``$ref`` keys); cycles
  raise :class:`RefError`.
- Sibling keys are only allowed when the referenced content is a JSON object.
- Currently only whole-file references are supported (no JSON-pointer
  fragments). HuggingFace Hub paths are not resolved here — only local files.
- Path resolution follows symlinks (``Path.resolve()``). HuggingFace cache
  snapshots are symlinks into a content-addressed blob directory, so a
  relative ``$ref`` inside an HF-cached config will resolve against that
  flat blob dir — not the snapshot dir — and is unlikely to find the target.
  Use absolute paths or copy the config out of the HF cache before adding
  ``$ref`` includes.
"""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REF_KEY = "$ref"


class RefError(Exception):
    """Raised when ``$ref`` resolution fails."""


def resolve_refs(config_path: str | Path) -> Any:
    """Load a JSON file and recursively resolve all ``$ref`` includes.

    Args:
        config_path: Path to the root JSON file.

    Returns:
        The fully resolved JSON tree (typically a dict, but may be a list or
        scalar depending on what the file and its includes contain).

    Raises:
        RefError: If a referenced file is missing, contains invalid JSON,
            forms a cycle, has a non-string ``$ref`` value, or has sibling
            keys alongside a ``$ref`` whose target is not a JSON object.
    """
    abs_path = Path(config_path).resolve()
    return _resolve_file(abs_path, _stack=())


def resolve_refs_to_tempfile(config_path: str | Path) -> Path:
    """Resolve ``$ref`` includes and write the result to a temp JSON file.

    The caller is responsible for unlinking the returned path.
    """
    resolved = resolve_refs(config_path)
    fd, tmp_path = tempfile.mkstemp(prefix="opentau_refs_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(resolved, f, indent=4)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def _resolve_file(abs_path: Path, _stack: tuple[Path, ...]) -> Any:
    if abs_path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, abs_path))
        raise RefError(f"Cyclic $ref detected: {chain}")
    if not abs_path.is_file():
        raise RefError(f"$ref target not found: {abs_path}")
    try:
        with open(abs_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RefError(f"Invalid JSON in {abs_path}: {e}") from e
    return _resolve_node(data, base_dir=abs_path.parent, _stack=(*_stack, abs_path))


def _resolve_node(node: Any, base_dir: Path, _stack: tuple[Path, ...]) -> Any:
    if isinstance(node, dict):
        if REF_KEY in node:
            ref_value = node[REF_KEY]
            if not isinstance(ref_value, str):
                raise RefError(
                    f"{REF_KEY} value must be a string path, got {type(ref_value).__name__} "
                    f"in {_stack[-1] if _stack else '<root>'}"
                )
            target = (base_dir / ref_value).resolve()
            loaded = _resolve_file(target, _stack)
            siblings = {k: v for k, v in node.items() if k != REF_KEY}
            if not siblings:
                return loaded
            if not isinstance(loaded, dict):
                raise RefError(
                    f"Cannot merge sibling keys {sorted(siblings)} with non-object "
                    f"content loaded from {target}"
                )
            resolved_siblings = {k: _resolve_node(v, base_dir, _stack) for k, v in siblings.items()}
            return _deep_merge(loaded, resolved_siblings)
        return {k: _resolve_node(v, base_dir, _stack) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_node(item, base_dir, _stack) for item in node]
    return node


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Return a deep-merged copy of ``base`` with ``overrides`` applied.

    For each key in ``overrides``: if both sides hold a dict, recurse; else the
    override value replaces the base value (lists are replaced, not concatenated).
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
