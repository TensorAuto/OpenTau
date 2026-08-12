# Changelog

All notable, behavior-affecting changes to OpenTau are recorded here. This file
tracks the **checkpoint config schema version** (`config_version`) alongside the
package version: a `config_version` bump means the model's inputs/outputs changed
in a way that must be gated so existing checkpoints keep the behavior their
weights were trained under.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added — `accel`, a cost-free flow-matching uncertainty proxy — **opt-in, no config_version bump**

Denoising acceleration (`accel`, [arXiv:2607.27933](https://arxiv.org/abs/2607.27933))
read off the flow-matching samplers' existing Euler loop. A maximally *certain* CFM field
is an affine-isotropic contraction, so its denoising trajectory is a straight line at
constant velocity; the normalized total variation of that velocity over the first `p` of
`T` Euler steps is a proxy for the endpoint posterior's spread. It needs **no extra
network evaluations, no resampling, no training, and no auxiliary probe** — the whole cost
is two vector norms per denoise step, on velocities the sampler already computed.

**No `config_version` bump.** Nothing about what the model sees or produces changes.
`accel` is off by default (`policy.accel_prefix is None`), and with it off every added
line in a sampler is dead, so the traced/compiled graph is unchanged.

- `opentau.policies.accel` — the estimator (`AccelMeter`), the action-dim mask recovered
  from the normalization buffers, `AccelProvenance`, and the shared `configure_accel`
  enable knob (`OPENTAU_ACCEL_PREFIX=auto`, or an explicit prefix `>= 2`).
- `opentau.utils.accel_detector` — offline one-sided CUSUM + split-conformal calibration
  over the per-chunk score stream, fitted on **successful** rollouts only.
- `opentau-accel-diagnose` — measures the bfloat16 rounding floor, the run-to-run spread,
  and the prefix profile of a checkpoint. **Run this first**: `accel`'s numerator is a
  difference of nearly-equal vectors, so bf16 puts a positive-biased floor under the score
  that compresses exactly the low-uncertainty end the method's premise depends on.

Measured on one real 6-DoF SO101 pi05 checkpoint (`T = 10`, QUANTILE action norm, delta
actions, `max_delay = 4`) over 8 real frames of its own training data: rounding floor
**0.0056**, redrawn-noise spread **0.0094**, between-observation spread **0.097** — a
signal-to-floor ratio of **17**, i.e. the score is measuring field geometry rather than
bf16 arithmetic on that checkpoint. Re-measure per checkpoint; this is not a constant.
The prefix sweep on the same run reproduces the paper's terminal singularity: step-over-step
growth in `accel_p` decelerates monotonically (122% → 62% → 69% → 38% → 41% → 33% → 22%)
and then *reverses* to +48% at the last step, which is why `default_prefix` returns `T - 1`
and not `T`.

Note that the serving precision is fixed in code rather than chosen by the caller: the
pi0/pi05 family routes its embedding path through **two** module-level `_preferred_dtype()`
hooks (one in `modeling_pi05`, one in `paligemma_with_expert`) that return bfloat16
unconditionally, so `policy.to(torch.float32)` alone does not produce a float32 forward.
- `opentau-accel-calibrate` — fits a threshold from an eval run's `eval_info.json`.
- `eval.py` records a per-episode `accel` stream (done-truncated, paired with `success`);
  the gRPC `ActionChunkResponse` gained an `optional float accel = 5` (explicit presence,
  so unset is distinguishable from a maximally-confident `0.0`); the RoboCasa server sends
  an `accel` envelope only to clients that set `"include_accel": true`.

Known limits, all inherited from the method: it is a *local* statement about the action
expert's posterior at one observation, blind to confident-but-wrong VLM-level errors and
to precision-sensitive failures, and it degrades on undertrained models. It is a monitor,
not a guarantee of correctness.


### Changed — openpi-faithful normalization (zero-range dims + epsilon) — **breaking, gated**

`config_version` **0 → 1.** Two openpi-parity normalization changes are bundled
into this one version (no checkpoint has been saved at `config_version` 1 yet, so
they are all-or-nothing rather than split across versions).

**(1) Zero-range dims map to `0.0`.** Under MIN_MAX and QUANTILE normalization, a
zero-range band (`max == min`: a zero-padded action/state tail dim, or a
genuinely-constant real dim) now maps to **`0.0`** instead of the legacy
**`-1.0`**. This matches the output of the reference implementation
[openpi](https://github.com/Physical-Intelligence/openpi), which normalizes only
the real dims (its `Normalize` slices `stats.q01[..., : x.shape[-1]]`) and
zero-pads *after* normalization — so its padded columns are a `0.0` pad constant.
OpenTau keeps padding in the dataset and reproduces that output arithmetically: a
`0.5 * (denom - (max - min))` numerator offset that is a float-exact no-op on
healthy dims and cancels the `* 2 - 1` re-centering exactly where the zero-range
guard fires. Healthy dims are bit-identical to before; `Unnormalize` round-trips
exactly; MEAN_STD is unchanged (it already emitted `0.0` on a zero-std dim).

**(2) Normalization epsilon is openpi's `1e-6`.** The epsilon added to every
normalization denominator (and used as the zero-range / zero-variance snap
threshold) changes from `1e-8` to openpi's `1e-6`
(`openpi.transforms.Normalize`). At `config_version` 1 a policy trained here and
one trained in openpi agree to well under float32 resolution on the same inputs.
Threaded per-policy via `config.normalization_epsilon()` → `Normalize(eps=...)`,
gated exactly like (1): legacy checkpoints (`config_version` 0) keep `1e-8`, so
their weights normalize as trained.

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

* **`repo_id@revision` checkpoint specs.** `--policy.path`, `--config_path` and
  `policy.pretrained_path` accept a revision suffix, so a published checkpoint can
  be pinned to one training step:

  ```bash
  --policy.path=TensorAuto/<runid>-<runname>@6000   # the "6000" tag
  --policy.path=TensorAuto/<runid>-<runname>        # main = the latest step
  ```

  Published checkpoints are now one repo per run with each saved step as a git
  tag, rather than one repo per step. Bare repo ids are unaffected (no `@`, no
  split, `main` as before), so previously published `<run>-<step>` repos keep
  working. **A local path is never split**, even one containing `@` — an existing
  path short-circuits, and a path-shaped string (leading `/`, `.`, `~`, a
  backslash, or more than one `/`) is left alone so it still fails as a path
  rather than as a bogus repo id. Parsed by `split_repo_revision` in
  `opentau/utils/hub.py`; passing both an `@` suffix and a conflicting
  `revision=` raises rather than silently picking one.
* **`--policy.path=<repo>` works against a published checkpoint.** Uploaded repos
  carried only `hf_config.json`, so `--policy.path=<repo>` failed on the missing
  `config.json` and `--config_path=<repo>` on the missing `train_config.json`; you
  had to fetch `hf_config.json` by hand. `PreTrainedConfig.from_pretrained` and
  `TrainPipelineConfig.from_pretrained` now fall back to `hf_config.json` (reading
  its `.policy` sub-object for the former), so the file is always found.

  **Scope:** finding the file is not the same as decoding it. `--policy.path` reads
  only the `.policy` sub-object and works on checkpoints going back as far as we
  have tested. `--config_path` decodes the *whole* pipeline config, so it still
  fails on a checkpoint whose train config predates an unrelated schema change —
  verified against a real 2-step consolidated repo, where
  `dataset_mixture.datasets[].grounding` (renamed to `vqa` in #124) raises a
  `DecodingError`. That is deliberate: `grounding` was renamed, not deleted, so
  stripping it would silently load the config with `vqa` at its default and lose a
  setting the run was actually trained with. Failing loudly is the correct
  behavior; use `--policy.path` for old checkpoints, or hand-edit the fetched
  config.
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

### Fixed

* **The `config_version` normalization gate now actually runs for every policy.**
  It is documented above as the convention a checkpoint's weights were trained
  under, resolved at load time by `PreTrainedPolicy.from_pretrained` — described
  as "the single chokepoint every weight load crosses". It was not: the seven
  policies that override `from_pretrained` (pi05, pi05_mem, pi06, both pi07
  low/high levels, both pi07_paligemma low/high levels — i.e. every policy in
  production use) never peeked the version, and `make_policy` always passes
  `config=`, so their config-loading branch never ran either. A **pre-versioning
  checkpoint loaded through any of them therefore normalized with the *current*
  convention instead of the legacy one it was trained under** — silent, since the
  weights load cleanly and only the numbers are wrong. Extracted as
  `resolve_checkpoint_provenance` (`policies/pretrained.py`) and called by all
  eight loaders immediately before `cls(config)` — the Normalize/Unnormalize
  modules read `config_version` at construction, so resolving it later is too
  late. Pinned by AST tests over the whole policy registry, including
  call-ordering. `validate_input_resolution` was missing from the same seven and
  is resolved alongside it.
* **A failed weight *resolution* no longer returns a randomly-initialized policy.**
  All seven per-policy `from_pretrained` overrides (pi05, pi05_mem, pi06, both
  pi07 low/high levels, both pi07_paligemma low/high levels) wrapped weight
  resolution in a bare `except Exception` that logged a warning and returned the
  freshly constructed model. An unknown repo, an unknown revision, a permissions
  error or a failed download therefore produced an *untrained* model that looked
  like a successful load until the loss curve said otherwise. These now raise.
  The one case still tolerated is a **local directory with no
  `model.safetensors`** — the DeepSpeed/ZeRO resume shape, where
  `accelerator.load_state` supplies the weights immediately afterwards — and it is
  now a distinct `CheckpointWeightsNotFoundError` rather than "any exception at
  all", logged as a warning that says the policy is otherwise randomly
  initialized.

  **Scope:** this covers *resolving and reading* the weights file. The later
  key-remap / `load_state_dict` stage in those same seven overrides is still
  wrapped in a broad `except Exception` that logs "Could not remap state dict
  keys" and returns a partially-loaded model — pre-existing, unchanged here, and
  a narrower hazard (`load_state_dict(strict=False)` already absorbs key
  mismatches, so what it hides is a genuine shape conflict). Tightening it is
  deliberately left to its own change, since it can unmask the documented
  partial-load warm-start paths.
* **Download arguments reached the Hub.** Five of those seven overrides read
  their own parameters back as `kwargs.get("revision")`, `kwargs.get("token")`,
  `kwargs.get("cache_dir")` and so on. Those are *keyword-only parameters*, so
  they never appear in `**kwargs` and every one of the six lookups was
  unconditionally `None`: `revision` was silently ignored, and private-repo loads
  worked only via an ambient `HF_TOKEN`. All seven now resolve weights through a
  single shared `resolve_pretrained_weights_file`, which is pinned by AST tests
  over the whole policy registry.
* Hub failures now carry actionable messages — an unknown revision names the
  repo's `/tags` page and points out that omitting `@<step>` loads the latest.
* Weights are resolved *before* the model is constructed, so a typo'd repo or tag
  fails in milliseconds instead of after a multi-billion-parameter init.
* `export_to_onnx.py` no longer creates a `./<org>/<repo>/` directory under the
  current working directory for a Hub checkpoint (which an `@<step>` suffix would
  have turned into a directory named after a git ref). Hub exports go to
  `<output_dir>/onnx/<spec>/`; local checkpoint dirs are unchanged.

### Migration notes

* **Legacy checkpoints served through pi05 / pi05_mem / pi06 / pi07 /
  pi07_paligemma change numerically.** Because those policies never resolved
  `config_version` (see Fixed), a pre-versioning checkpoint was being normalized
  with the current convention. It is now correctly read as legacy `0` — matching
  what the weights were trained under, and matching what the base-class policies
  (pi0, value, cosmos3) already did. Expect small output differences on such a
  checkpoint; they are the *correction*. To deliberately keep the previous
  behavior, set `--policy.config_version=1` explicitly, which is honored untouched.
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

### Added — row cap for on-the-fly delta-action stats — *opt-in, no default change*

`DatasetMixtureConfig.delta_stats_max_rows: int | None` (default `None`) caps how
many anchor frames each `use_delta_joint_actions` dataset contributes to the
delta-action normalization stats computed on the fly at dataset load
(`opentau/datasets/delta_action_stats.py`). That pass is `O(frames x chunk_size)`
— an `H`-fold blow-up over a per-frame stats pass — so on a very large source the
first run (before the disk cache is warm) can cost hours.

Set a positive int to bound it. Rows are subsampled with a **uniform stride over
the whole dataset**, never a prefix, and the stride's phase carries across episode
boundaries so episodes aren't all anchored at frame 0; a dataset's cap is split
across its parquet files in proportion to the rows each actually contributes
(counting only what an `episodes` / `excluded_episodes` filter selects), so a
capped dataset pools in the same file ratio an uncapped one would. Cross-dataset
pooling into a shared normalization head is unaffected either way — that path
weights members by `info["total_frames"]`, not by the sampled `count`, so capping
changes a dataset's estimator *precision*, never its share of the head. The cap
applies **per dataset**, so
one value bounds the worst case across a heterogeneous mixture while leaving
datasets smaller than the cap byte-identical to before.

The cap is folded into the stats cache key, so changing it recomputes instead of
serving numbers from a different sampling budget. It is folded in **only when
set**, so uncapped configs keep the digest — and therefore the already-computed
cache files — they had before this change.

Cost and fidelity, measured on a 100k-frame / 50-step-horizon source: the capped
run drops the accumulation from `O(frames x chunk_size)` to `O(cap x chunk_size)`
(2.10s → 0.23s of compute at a 20x subsample; 7.36s → 0.44s at a 200-step
horizon), leaving only the parquet column read, which the cap does not shrink.
`mean`/`std` and `q01`/`q99` stay within 0.02 / 0.05 standard deviations of the
uncapped values at that subsample; `min`/`max` degrade first (~0.9 sd), since a
subsample doesn't see the extremes — so MIN_MAX normalization is the mode most
sensitive to a tight cap, and MEAN_STD / QUANTILE the least.

Default `None` reads every row, exactly as before; nothing changes unless you set
the field.
