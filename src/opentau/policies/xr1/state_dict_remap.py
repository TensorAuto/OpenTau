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

"""Reference-checkpoint -> ``xr1`` state-dict key remap.

``XR1FlowMatching`` subclasses ``Qwen3VLWithDiT``, whose children are named exactly as the
reference names them (``vlm``, ``dit``, ``state_projector``, ``action_projector``,
``action_output_layer``, ``t_embedder``, ``t_projector``, ``sink``), and ``XR1Policy``
holds it as ``self.model``. So the whole remap is **one rule**: prefix everything the
checkpoint carries with ``model.``.

Keeping it to one rule is the point. A remap with per-module special cases is where a
partial load hides -- ``load_state_dict(strict=False)`` reports a mismatched key as
"missing" and returns a model that is silently half random. :func:`remap_reference_state_dict`
therefore also reports what it did, and :func:`assert_full_coverage` turns "some keys did
not land" into an exception rather than a log line.
"""

from __future__ import annotations

from torch import Tensor

#: Top-level module names the reference checkpoint carries.
REFERENCE_TOP_LEVEL_MODULES = frozenset(
    {
        "vlm",
        "dit",
        "state_projector",
        "action_projector",
        "action_output_layer",
        "t_embedder",
        "t_projector",
        "sink",
    }
)

#: The two ends of Qwen3-VL's tied word embedding. ``tie_word_embeddings=True`` means the
#: input table and the output head are the **same storage**, so a checkpoint carries only
#: one of them -- and *which* one depends on where it came from:
#:
#: * the reference checkpoint omits ``lm_head.weight`` (it stores ``embed_tokens.weight``);
#: * a checkpoint written by ``save_pretrained`` omits ``embed_tokens.weight``, because
#:   ``safetensors`` refuses to serialize aliased storage twice and keeps one name.
#:
#: So exactly one of the pair may be missing, and whichever it is, loading the other
#: populates it once ``tie_weights()`` re-establishes the alias. Both missing is a genuine
#: failure -- the embedding table really did not arrive.
TIED_WEIGHT_KEYS = frozenset(
    {"model.vlm.lm_head.weight", "model.vlm.model.language_model.embed_tokens.weight"}
)


def remap_reference_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Prefix reference keys with ``model.``; leave already-namespaced keys alone.

    Idempotent, and a no-op for OpenTau-native checkpoints (whose keys already start with
    ``model.`` or are ``normalize_*`` / ``unnormalize_*`` buffers).
    """
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        top = key.split(".", 1)[0]
        if top in REFERENCE_TOP_LEVEL_MODULES:
            remapped[f"model.{key}"] = value
        else:
            remapped[key] = value
    return remapped


def assert_full_coverage(
    missing_keys: list[str], unexpected_keys: list[str], *, stripped_keys: frozenset[str] = frozenset()
) -> None:
    """Raise unless every key landed, tolerating tied weights and deliberate strips.

    Args:
        missing_keys: ``load_state_dict(strict=False)``'s missing keys.
        unexpected_keys: ...and its unexpected ones.
        stripped_keys: Normalization buffers intentionally removed via
            ``config.skip_normalization_weights``.

    Raises:
        ValueError: if anything other than **one** end of the tied word embedding or a
            stripped buffer is missing, or if any key was unexpected.
    """
    missing_tied = [key for key in missing_keys if key in TIED_WEIGHT_KEYS]
    if len(missing_tied) == len(TIED_WEIGHT_KEYS):
        raise ValueError(
            f"Both ends of the tied word embedding are missing ({sorted(missing_tied)}). Exactly "
            "one may be absent (safetensors stores aliased storage once); both means the "
            "embedding table did not arrive at all."
        )
    unintended_missing = [
        key for key in missing_keys if key not in TIED_WEIGHT_KEYS and key not in stripped_keys
    ]
    if unintended_missing or unexpected_keys:
        raise ValueError(
            "The xr1 state-dict remap did not fully cover the checkpoint. "
            f"Missing ({len(unintended_missing)}): {unintended_missing[:20]}. "
            f"Unexpected ({len(unexpected_keys)}): {unexpected_keys[:20]}. "
            "A partial load leaves the model half randomly-initialized, which looks like a "
            "successful load until the numbers are wrong — so this raises rather than warns."
        )


def validate_against_manifest(
    state_dict: dict[str, Tensor], manifest: dict[str, dict[str, list[int] | str]]
) -> None:
    """Check a *reference-keyed* state dict against a captured name/shape manifest.

    Raises:
        ValueError: on a key-set difference or a shape mismatch.
    """
    got = set(state_dict)
    want = set(manifest)
    if got != want:
        raise ValueError(
            f"Key-set mismatch against the manifest: missing {sorted(want - got)[:10]}, "
            f"extra {sorted(got - want)[:10]}."
        )
    mismatched = [
        (key, list(state_dict[key].shape), manifest[key]["shape"])
        for key in sorted(got)
        if list(state_dict[key].shape) != list(manifest[key]["shape"])
    ]
    if mismatched:
        raise ValueError(f"Shape mismatches against the manifest: {mismatched[:10]}")
