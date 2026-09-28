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

"""OpenTau wrapper around the vendored ``FluxActionPolicy``.

This adapter is deliberately thin. Upstream's policy already exposes OpenTau's
``PreTrainedPolicy`` contract almost exactly -- both descend from the same LeRobot
lineage -- so the wrapper's whole job is:

1. **Batch-key translation.** OpenTau keys state/actions as ``observation.state`` /
   ``actions``; upstream keys them ``state`` / ``action``. Cameras need no translation
   at all: upstream resolves them by plain ``batch[key]`` lookup over
   ``config.camera_keys``, so those are configured with OpenTau-style
   ``observation.images.*`` names and the frames are found where they already live.
2. **Normalization ownership.** F3A range-normalizes state and actions *itself* from the
   q01/q99 bounds carried in its own config, so every OpenTau normalization mode is
   ``IDENTITY``. Running both would double-normalize the action targets.
3. **Deferring the rest.** ``forward`` already returns ``(loss, aux_dict)``, and
   ``get_optim_params`` already returns PyTorch param groups (the heads run at 5x the
   trunk LR, which is part of the recipe) -- both pass straight through.

The ``video_vae`` / ``text_encoder`` injection points exist so CPU tests can build the
policy against ``MockTextEncoder`` and a stub VAE instead of pulling ~7B of frozen
encoder weights from the Hub.
"""

from typing import Any

from torch import Tensor

from opentau.constants import ACTION, OBS_STATE
from opentau.policies.normalize import Normalize, Unnormalize, resolve_num_datasets
from opentau.policies.pretrained import PreTrainedPolicy

from .configuration_flux3_action import Flux3ActionConfig
from .policy import FluxActionPolicy


class Flux3ActionPolicy(PreTrainedPolicy):
    """OpenTau wrapper around ``FluxActionPolicy`` (FLUX 3 Action, Black Forest Labs)."""

    config_class = Flux3ActionConfig
    name = "flux3_action"
    # The trunk is packed into one variable-length sequence per micro-batch, so shapes
    # are dynamic by construction; leave compile off until seeded runs are verified
    # bit-identical (CLAUDE.md rule 3).
    supports_torch_compile = False

    def __init__(
        self,
        config: Flux3ActionConfig,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
        *,
        video_vae: Any | None = None,
        text_encoder: Any | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Every mode is IDENTITY (see the module docstring), so these carry no stats and
        # exist only to keep the policy's buffer surface uniform with the rest of the repo.
        num_datasets = resolve_num_datasets(per_dataset_stats, dataset_names, config)
        zero_range_center = config.zero_range_centers_on_zero()
        eps = config.normalization_epsilon()
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
            eps=eps,
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
            eps=eps,
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            per_dataset_stats=per_dataset_stats,
            dataset_names=dataset_names,
            num_datasets=num_datasets,
            zero_range_center=zero_range_center,
            eps=eps,
        )

        self.model = FluxActionPolicy(
            config.to_policy_config(),
            video_vae=video_vae,
            text_encoder=text_encoder,
        )

    # ------------------------------------------------------------------ batch keys
    def _upstream_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Alias OpenTau's state/action keys to upstream's, sharing the same tensors.

        A shallow copy with two extra aliases -- no tensor is cloned or moved, and any
        key upstream reads opportunistically (``task``, ``action_prev``, ``window_seed``,
        every ``*_is_pad`` flag it scans to drop windows that overrun an episode) passes
        through untouched.
        """
        out = dict(batch)
        if OBS_STATE in batch:
            out.setdefault("state", batch[OBS_STATE])
        if ACTION in batch:
            out.setdefault("action", batch[ACTION])
        return out

    # ------------------------------------------------------------------ contract
    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict | None]:
        """Training loss for a micro-batch.

        Returns upstream's scalar flow-matching loss unchanged, with its ``video_mse`` /
        ``action_mse`` / ``n_valid_windows`` diagnostics as the auxiliary dict -- the
        video term is part of the joint objective, not an optional extra (see
        ``configuration_flux3_action``).
        """
        return self.model(self._upstream_batch(batch))

    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Select the next action, refilling upstream's internal chunk queue as needed."""
        return self.model.select_action(self._upstream_batch(batch), **kwargs)

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Predict a full ``chunk_size`` action chunk in dataset units."""
        return self.model.predict_action_chunk(self._upstream_batch(batch), **kwargs)

    def get_optim_params(self) -> list[dict]:
        """Upstream's param groups, which put the embodiment heads at 5x the trunk LR."""
        return self.model.get_optim_params()

    def reset(self) -> None:
        """Clear the action queue. Called on every environment reset.

        No construction-time guard is needed: ``PreTrainedPolicy.reset`` is abstract and
        the base ``__init__`` never invokes it, and the inner policy already resets itself
        at the end of its own ``__init__``.
        """
        self.model.reset()
