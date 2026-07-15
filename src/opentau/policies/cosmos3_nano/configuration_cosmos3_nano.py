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

"""Configuration for the cosmos3_nano policy.

cosmos3_nano is the cosmos3 recipe (see ``configuration_cosmos3.py``) with the frozen
Qwen3-VL-32B backbone (the Cosmos3-Super reasoning tower) swapped for a **frozen
Qwen3-VL-8B** — the reasoning tower of NVIDIA ``Cosmos3-Nano``, extracted into a
standalone checkpoint by ``opentau.scripts.extract_cosmos3_reasoner`` and published as
``TensorAuto/cosmos3-reason-8b`` (**private** — like the 32B extraction, the training
environment needs an HF token with read access to the TensorAuto org;
``from_pretrained`` picks up the ambient ``HF_TOKEN`` / ``~/.cache/huggingface/token``).

The Nano tower is byte-for-byte the Qwen3-VL-8B geometry: hidden 4096, **36 layers**,
32 attention heads, and — decisive for the action expert's cross-attention — the same
8 key/value heads x head_dim 128 as the 32B tower. The expert's hard KV-concat
constraints therefore carry over unchanged, and the only default that must move is
``expert_num_hidden_layers`` (36, matching the backbone depth for the default per-layer
correspondence). With the inherited expert widths (hidden 1024, 16 query heads,
intermediate 2048) the trainable expert + projections total ~0.51B parameters.

``condition_on_layer`` works exactly as in cosmos3 (valid range [-36, 35]); every other
field, constraint, and preset is inherited from ``Cosmos3Config``.
"""

from dataclasses import dataclass

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.cosmos3.configuration_cosmos3 import Cosmos3Config


@PreTrainedConfig.register_subclass("cosmos3_nano")
@dataclass
class Cosmos3NanoConfig(Cosmos3Config):
    """Configuration class for the cosmos3_nano policy.

    Identical to :class:`Cosmos3Config` except for the two nano defaults below; see the
    cosmos3 docstrings for the meaning of every field and the hard expert/backbone
    geometry constraints (re-validated against the loaded backbone at model-build time).

    Args:
        pretrained_backbone_repo_id: HF repo id (or local path) for the Qwen3-VL-8B
            backbone weights — the Cosmos3-Nano reasoning tower extracted by
            ``opentau.scripts.extract_cosmos3_reasoner`` (run once on the ungated
            ``nvidia/Cosmos3-Nano``). Defaults to ``TensorAuto/cosmos3-reason-8b``,
            the published extraction (**private** — needs an HF token with TensorAuto
            org read access). Re-run the extraction script to reproduce or re-host it.
        expert_num_hidden_layers: Action-expert depth. With ``condition_on_layer=None``
            this MUST equal the backbone text tower depth (36 for Qwen3-VL-8B); with a
            single ``condition_on_layer`` selected, any depth >= 1 is allowed.
            Defaults to 36.
    """

    # --- Backbone (Qwen3-VL-8B reasoning tower extracted from NVIDIA Cosmos3-Nano) ---
    pretrained_backbone_repo_id: str = "TensorAuto/cosmos3-reason-8b"

    # --- Action-expert sizing (must match the 36-layer Nano tower; see module docstring) ---
    expert_num_hidden_layers: int = 36
