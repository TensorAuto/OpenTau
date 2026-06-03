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

r"""Pre-download RoboCasa365 kitchen assets into the venv-external assets root.

The same download happens lazily on the first RoboCasa env build (see
``opentau.envs.robocasa._ensure_robocasa_assets``), but warming the cache up front — e.g.
on a login node before launching a multi-rank eval — means non-zero ranks never block on
rank-0's first download.

Examples::

    python -m opentau.scripts.download_robocasa_assets
    python -m opentau.scripts.download_robocasa_assets --obj_registries '["lightwheel","objaverse"]'
    ROBOCASA_ASSETS_ROOT=/data/robocasa python -m opentau.scripts.download_robocasa_assets
    python -m opentau.scripts.download_robocasa_assets --assets_root /data/robocasa
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from opentau.configs import parser
from opentau.envs.robocasa import (
    ROBOCASA_ASSETS_ROOT_ENV,
    _ensure_robocasa_assets,
    _needed_asset_packs,
    _resolve_robocasa_assets_root,
)


@dataclass
class Args:
    # External directory to download into. ``None`` -> ROBOCASA_ASSETS_ROOT env var, else
    # the HF_OPENTAU_HOME default (kept outside the ephemeral uv venv).
    assets_root: str | None = None
    # Object-mesh registries whose packs to fetch (matches RoboCasaEnv.obj_registries).
    obj_registries: list[str] = field(default_factory=lambda: ["lightwheel"])


@parser.wrap()
def main(args: Args):
    root = Path(args.assets_root).expanduser() if args.assets_root else _resolve_robocasa_assets_root()
    # Export before downloading so the loader redirect (run during the robocasa import in
    # `_ensure_robocasa_assets`) and the download destinations agree on the same path.
    os.environ[ROBOCASA_ASSETS_ROOT_ENV] = str(root)

    print("RoboCasa assets root:", root)
    print("Asset packs to ensure:", _needed_asset_packs(args.obj_registries))
    _ensure_robocasa_assets(root, args.obj_registries)
    print("Done. RoboCasa assets ready at:", root)


if __name__ == "__main__":
    main()
