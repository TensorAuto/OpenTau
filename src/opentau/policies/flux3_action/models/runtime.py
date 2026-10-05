# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Small runtime helpers shared by the flux3 model components."""

import os

import torch

MOCK_INIT_SEED = 0
TRUNK_WEIGHTS_FILENAME = "dit.safetensors"


def parse_weight_spec(spec: str) -> tuple[str, str | None, str | None]:
    """``repo_id[:filename][@revision]`` -> ``(repo_id, filename, revision)``.

    ``@revision`` pins a commit sha, branch or tag so a recipe keeps loading the same bytes after the file
    is replaced on the repository head. Local paths are not parsed; check :func:`os.path.exists` first.
    """
    body, _, revision = spec.partition("@")
    repo_id, _, filename = body.partition(":")
    if not repo_id:
        raise ValueError(f"weight spec must be a local path or repo_id[:filename][@revision], got {spec!r}")
    return repo_id, filename or None, revision or None


def _looks_like_local_path(repo_id: str) -> bool:
    """Whether a spec's repo_id reads as a filesystem path rather than a Hub ``namespace/name``."""
    return (
        repo_id.startswith((os.sep, "." + os.sep, ".." + os.sep, "~"))
        or repo_id.endswith((".safetensors", ".pt", ".bin"))
        or repo_id.count("/") > 1
    )


def resolve_weights(spec: str, default_filename: str) -> str:
    """A local file or directory is returned as is; anything else is downloaded from the Hub."""
    if os.path.exists(spec):
        return spec
    repo_id, filename, revision = parse_weight_spec(spec)
    if _looks_like_local_path(repo_id):
        raise FileNotFoundError(
            f"no such file or directory: {spec!r}. Configured weight paths are relative to the working "
            f"directory; download the weights first (see docs/setup.md). Pass a Hub spec as "
            f"repo_id[:filename][@revision] to download instead."
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename or default_filename, revision=revision)


def random_init_(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
    """Fill a model with small random values and cast to ``dtype`` (weightless wiring/smoke runs).

    Seeded so repeated builds are identical.
    """
    torch.manual_seed(MOCK_INIT_SEED)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.normal_(0, 0.02)
    model.to(dtype=dtype)
