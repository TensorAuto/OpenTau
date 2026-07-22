# Changelog

All notable, behavior-affecting changes to OpenTau are recorded here. This file
tracks the **checkpoint config schema version** (`config_version`) alongside the
package version: a `config_version` bump means the model's inputs/outputs changed
in a way that must be gated so existing checkpoints keep the behavior their
weights were trained under.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed — normalization of zero-range (padded / constant) dims — **breaking, gated**

`config_version` **0 → 1.**

Under MIN_MAX and QUANTILE normalization, a zero-range band (`max == min`: a
zero-padded action/state tail dim, or a genuinely-constant real dim) now maps to
**`0.0`** instead of the legacy **`-1.0`**. This matches the output of the
reference implementation
[openpi](https://github.com/Physical-Intelligence/openpi), which normalizes only
the real dims (its `Normalize` slices `stats.q01[..., : x.shape[-1]]`) and
zero-pads *after* normalization — so its padded columns are a `0.0` pad constant.
OpenTau keeps padding in the dataset and reproduces that output arithmetically: a
`0.5 * (denom - (max - min))` numerator offset that is a float-exact no-op on
healthy dims and cancels the `* 2 - 1` re-centering exactly where the zero-range
guard fires. Healthy dims are bit-identical to before; `Unnormalize` round-trips
exactly; MEAN_STD is unchanged (it already emitted `0.0` on a zero-std dim).

Where OpenTau still, deliberately, differs from openpi: on a genuinely-constant
*real* dim whose value deviates from the constant, openpi divides by `~1e-6` and
emits `~1e6` (the "Outlier normalized state" blow-up); OpenTau's zero-range guard
keeps it bounded and invertible at `2 * deviation`. We are openpi-faithful where
openpi is correct and deliberately divergent only where it is numerically broken.

**Why this is gated.** Every checkpoint trained before this change learned a
constant `-1.0` across its padded dims (e.g. 24 of 32 dims for an openpi DROID
norm-stats file padded to 32). Flipping the behavior unconditionally would
invalidate those weights. The `config_version` field on the policy config gates
it, and resolves automatically — **users do not set it manually**:

* **Fresh training run** (`config_version` absent from your JSON): resolves to the
  current version (1) → new `0.0` behavior.
* **Loading pre-fix weights** whose config carries no tag: resolves to `0`
  (legacy `-1.0`), so existing checkpoints, and serving / ONNX export of them,
  keep their trained behavior. Resolution happens in
  `PreTrainedPolicy.from_pretrained` — the single chokepoint every weight load
  (factory fine-tune, resume, gRPC serve, export) crosses — keyed off the
  weights, not the config file ("disk is truth").
* **Fine-tuning from a pre-fix checkpoint stays on `config_version` 0**, including
  the *new* checkpoints the fine-tune writes — the version tracks the
  normalization convention the weights were trained under, not the code that
  produced them. This is intentional: silently changing normalization under a
  pretrained backbone would corrupt it. To deliberately migrate a fine-tune to
  the new convention, pass `--policy.config_version=1`.
* **Resume across this code change**: a run started before the change and resumed
  after keeps `config_version` 0 mid-training (no discontinuity).

Every saved checkpoint self-describes: `config.json` / `train_config.json` now
carry a concrete `config_version` (and an informational `opentau_version`).

### Added

* `config_version: int | None` and `opentau_version: str | None` on
  `PreTrainedConfig`. `config_version` is a monotonic schema version for
  behavioral conventions; `opentau_version` is an informational package-version
  stamp, never read by any branch.
* `fit_fast_tokenizer.py --config-version` (default: current). The FAST
  discrete-action tokenizer bakes the normalization convention into its BPE
  corpus, so it is versioned alongside the policy: the fit writes an
  `opentau_action_norm.json` sidecar, and every policy that loads a
  discrete-action tokenizer raises at construction if the tokenizer's convention
  disagrees with its `config_version`. Upstream / pre-versioning tokenizers
  (e.g. `physical-intelligence/fast`) carry no sidecar and are a no-op.

### Migration notes

* **Discrete-action (FAST) runs at `config_version` 1 need a tokenizer re-fit.**
  A `config_version` 1 policy pointed at a tokenizer fit under the old convention
  raises a clear error. Re-fit with `fit_fast_tokenizer.py --config-version 1`,
  or pin the policy to the tokenizer's convention with `--policy.config_version=0`.
* **Forward-incompatibility.** Checkpoints written after this change carry the new
  `config_version` key. An **older** OpenTau install reading such a config fails
  with a draccus `DecodingError` (unknown field). Mixed-version resume — an older
  install resuming a run whose config was written by a newer one — is
  unsupported; upgrade all workers to a build that includes this change.
* **Converted / uploaded checkpoints.** Checkpoint convert-and-upload tooling that
  publishes the training config as a full passthrough of `train_config.json`
  (e.g. under the name `hf_config.json`) carries `config_version` through
  automatically — the resolver peeks `config.json`, `train_config.json`, and
  `hf_config.json`. Tooling that instead writes a HuggingFace `config.json` from a
  hand-built dict must re-emit it from the loaded config, or the tag is absent and
  the checkpoint is read as legacy.
