# Changelog

All notable, behavior-affecting changes to OpenTau are recorded here. This file
tracks the **checkpoint config schema version** (`config_version`) alongside the
package version: a `config_version` bump means the model's inputs/outputs changed
in a way that must be gated so existing checkpoints keep the behavior their
weights were trained under.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed — `accel`'s prefix is now *measured* rather than assumed

`default_prefix` returns `T - 1` because that is the prefix the paper's online detector
used. The 0.13.0 notes below cite the prefix sweep as confirming it, via the step-over-step
growth in `accel_p` reversing at the last step. **That inference does not hold**, and this
entry supersedes it: both of `accel_p`'s sums accumulate over the prefix, so its magnitude
— and the shape of its growth — rise with `p` whether or not the extra Euler steps carry
posterior information. The sweep is descriptive; it would produce a similar curve on data
with no signal at all, so its peak cannot select a prefix.

The criterion that *can* is the rank correlation between `accel_p` and an independent
measure of posterior spread, and that reference has to be bought the expensive way — `K`
resampled denoises of one observation, whose disagreement is the posterior width.
`opentau-accel-diagnose` now measures it (`OPENTAU_ACCEL_OBSERVATIONS`, default 24;
`OPENTAU_ACCEL_RESAMPLES`, default 32; `0` skips the study) and reports `rho` per prefix
with a recommendation.

Re-measured on the same 6-DoF SO101 pi05 checkpoint (`T = 10`, 24 real frames, `K = 32`):
`rho` climbs to **+0.833 at `p = 9`** and falls to **+0.812 at `p = 10`**. So `T - 1` is
right on this checkpoint and the terminal step really is the weaker one — the same
conclusion as before, now resting on evidence that supports it. Every prefix scored
+0.70…+0.83, so prefix choice is not very consequential here; the paper's mid-schedule
optimum (`p/T ~ 0.4-0.5`, i.e. `p = 4-5`, `rho` +0.75) does **not** transfer. Re-measure
per checkpoint — a `T = 5` schedule (pi06/pi07) is unexamined, and there `T - 1` and the
paper's mid-schedule choice are far apart.

Supporting changes, none of which alter a served score:

- `AccelMeter` gained an **opt-in** per-step trace and `value_at(p)`, so every prefix is
  recoverable from **one** denoise pass instead of `T - 1` — that is what makes the study
  affordable, and it makes the descriptive sweep `T - 1` times cheaper too. Off in serving.
- `record_traces()` hands the meters built inside it back to a diagnostic, so no policy has
  to expose its internals. A `ContextVar`, not a global: the gRPC server samples from
  several threads, and a global would splice a diagnostic into serving traffic.
- `action_dim_scale()` converts a raw-unit spread into the normalized units `accel` is
  computed in, sharing one buffer extractor with `resolve_action_dim_mask`.
- Diagnostic observations are drawn across **every** dataset in the configured mixture
  rather than only the first — the correlation is taken across observations, so their
  variety is the study's resolution.
- The float32 leg of the dtype-floor measurement can be skipped
  (`OPENTAU_ACCEL_MEASURE_DTYPE_FLOOR=0`). It holds a second copy of the weights and is the
  first thing to fail on a shared GPU; skipping reports `NaN` rather than a fabricated ratio.

## [0.13.0] - 2026-08-17

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
* **`policy.train_state_action_representation_only`** — an opt-in finetuning mode
  for adapting a generic VLA checkpoint to a new dataset by training only the
  parameters that are actually dataset-specific: the discrete-action
  representation (`discrete_action_embedding` and `da_head`) and the state/action
  projections (`state_proj`, `action_in_proj`, `action_out_proj`), which are the
  only outer projections whose shape is a function of the robot rather than of the
  architecture. Everything else freezes — the VLM backbone and its tied `lm_head`,
  the vision/video encoder *and* the multimodal projector, the action expert, the
  time MLPs, and the optional modality embeddings. Declared on the nine policy
  types that have such a representation; the two high-level planners and `value`
  deliberately do not carry the field, since it would train nothing there.
  `config.dropout` is a no-op under the flag, because the frozen trunk is pinned to
  `eval()`. Default `False`.
* **Injectable request hooks on the gRPC inference server.** An embedding
  application can observe accepted inference requests without coupling OpenTau to
  platform-specific activity tracking. The default hook is a no-op, every
  `GetActionChunk` request invokes it (including messages processed through
  `StreamActionChunks`), and hook failures are isolated so they can never fail an
  inference RPC.

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
* **`action_chunk` is re-checked against the policy's execution horizon.**
  `TrainPipelineConfig` copies its pipeline-level `action_chunk` onto
  `policy.chunk_size` by `setattr`, which lands *after* the policy config's
  `__post_init__` has run — so the policy's own `n_action_steps <= chunk_size`
  invariant (and the shortened-horizon-vs-`max_delay`/`safety_buffer` pair) only
  ever saw the `chunk_size` the policy *declared*, never the propagated one. A
  config setting `action_chunk` and leaving `n_action_steps` at its default
  therefore reached, silently, exactly the combination the very same policy
  config rejects when it appears in JSON. Every in-repo config dodges it only by
  setting the two equal by hand. The checks are now one shared
  `PreTrainedConfig.validate_action_horizon` — replacing a copy in each of the
  seven policy configs that carried it — called from both policy `__post_init__`
  and the end of `_propagate_shape_fields_to_policy`, and the error names
  `action_chunk` as the cause.
  **A config with `n_action_steps > action_chunk` now fails at config time**; the
  model always decoded only `chunk_size` actions, so set `n_action_steps` to
  `action_chunk` (what such a run was effectively doing) to keep its behavior.
  As a side effect of sharing the check, pi0 gains the `safety_buffer <=
  chunk_size` bound its `max_delay` siblings already enforced.
* **The train/val split is identical on every rank.** `make_dataset` drew its
  `random_split` from the global RNG, which `set_seed(seed, accelerator=...)`
  deliberately offsets per process (`seed += process_index * 12345`) so that
  per-sample draws are decorrelated across ranks. Every rank therefore partitioned
  the data differently, and because each builds its own `Subset` views, each rank's
  validation frames landed in some other rank's training set. Simulating 8 ranks
  over a 353,222-frame dataset at a 2% val ratio, the intersection of the per-rank
  validation sets is **empty** — 100% of the frames behind the reported
  `Validation/*` metrics were in at least one rank's training set. Validation loss
  was optimistically biased, `running_best` checkpoint selection was driven by that
  contaminated metric, and val numbers were not comparable across runs with
  different world sizes. The split now uses a dedicated, rank-independent
  `torch.Generator` seeded from `cfg.seed` alone (with a fixed
  `DEFAULT_VAL_SPLIT_SEED` for `cfg.seed = None`, which `train.py` otherwise leaves
  on each process's own entropy — the same bug by another route). It is now a pure
  function of `cfg.seed`, `len(dataset)` and the effective `val_split_ratio`,
  independent of `process_index`, world size, mixture composition, and how much
  global RNG ran beforehand. `warn_if_resumed_split_differs` warns when a resume
  changes any of those inputs. The per-rank offset in `set_seed` is load-bearing
  elsewhere and was left alone.
* **The training sampler is seeded from `cfg.seed`.**
  `WeightedDatasetMixture.get_dataloader` constructed `HierarchicalSampler` with
  neither `seed` nor `generator`, so it fell back to a bare `torch.Generator()`.
  That fallback is *not* unseeded — PyTorch default-constructs generators with a
  fixed constant — so the stream was deterministic and identical on every rank,
  and unlike the split above there was no cross-rank divergence and no leak. Two
  things were wrong anyway: `cfg.seed` never reached the sampler, so a seed sweep
  varied only model initialization and augmentation draws while feeding all its
  runs identical data in identical order; and the data order rested on an
  undocumented PyTorch implementation detail. The seed is now passed through as the
  raw config value, so it stays rank-independent — which matters because
  `accelerate.prepare` shards with `dispatch_batches=False`, where every rank
  iterates the whole sampler stream and keeps the batches at its own offset. See
  Migration notes: **this changes the data order of any run with `cfg.seed` set.**
* **`max_action_dim` reaches the policy config.** `TrainPipelineConfig` assigned it
  to `self.policy.max_action_state` — a field no `PreTrainedConfig` subclass
  declares. Policy configs are plain unslotted dataclasses, so the assignment
  silently created an attribute nothing reads and the width never propagated, while
  its two siblings (`max_state_dim`, `chunk_size`) did. Loading a 6-DoF checkpoint
  under a pipeline with the default `max_action_dim = 32` therefore forced the state
  side to 32 while the action side stayed at 6, with no error. The typo dates to the
  initial commit and was never spelled correctly. Both call sites now delegate to
  one guarded `_propagate_shape_fields_to_policy`, driven by a single
  `PIPELINE_TO_POLICY_SHAPE_FIELDS` mapping, which skips a field the policy
  dataclass does not declare rather than inventing it — three registry policies
  declare no `max_action_dim`, and the `chunk_size` line had been inventing one on
  both high-level planners — and logs at WARNING when an override actually changes
  a value. Every config under `configs/` already set both sides to the same value,
  so no shipped config changes behavior.
* **Partial quantile coverage across an aggregated norm head no longer raises.**
  `aggregate_feature_stats` raised when the contributors sharing one normalization
  head disagreed about carrying `q01`/`q99`, so the first mixture pairing a
  quantile-migrated dataset with an unmigrated one died at startup — before the
  checkpoint was even loaded, and regardless of whether the job used QUANTILE
  normalization at all. The failure was symmetric in which dataset was the migrated
  one. Partial coverage now behaves like the "nobody has it" case — the quantile is
  simply absent from the aggregate — plus a warning naming how many contributors
  were missing it and which. **The no-backfill principle is unchanged:** a quantile
  is still never synthesized from a min/max. Two related fixes ride along:
  `q10`/`q50`/`q90` are aggregated too (the loop covered only `q01`/`q99`, so the
  inner quantiles that fleet stats now carry were silently dropped from every
  aggregated head), and `aggregate_stats` keeps `weights` and `contributor_names`
  aligned with its per-feature contributor filter — an unfiltered `weights` was
  broadcast against a shorter stats stack, inflating `count` and skewing every
  weighted stat against that denominator. Stats-resolution errors now name the
  offending row rather than a bare index.
* **The delta-action horizon is read under the raw action column.**
  `_compute_or_load_delta_stats` read it as `dt_mean["actions"]`, but
  `delta_timestamps_params` is keyed by **raw on-disk column names** — a standard
  LeRobot repo names the column `action` — so **every** `use_delta_joint_actions`
  run died at dataset build with `KeyError: 'actions'`. 13 of the 20 mappings in
  `standard_data_format_mapping.py` use `"actions": "action"` and were affected;
  the other 7 worked. It failed loudly and immediately, so the cost was a wasted
  allocation rather than silent corruption. Introduced by the delta-joint-actions
  feature in 0.12.0 and fixed here.
* **The RoboCasa policy server preserves the float32 SigLIP embeddings.** It
  blanket-cast the loaded policy to bfloat16, re-rounding the patch-embedding conv
  and position-embedding table that `to_bfloat16_like_physical_intelligence` pins to
  float32 for openpi parity. Nothing crashed; the precision was simply gone, and a
  later `.to(float32)` cannot recover it — so a pi0/pi05-family checkpoint served
  through this path did not have a bit-identical vision tower to the same checkpoint
  under `eval.py` or the gRPC server, making success rates measured through the two
  paths not strictly comparable. The cast now routes through
  `to_dtype_preserving_siglip_float32`, as the other entry points already did. The
  file predated that helper rather than being skipped by it. An AST guard now fails
  when any script in `scripts/` grows an unlisted `policy.to(dtype=...)`, so the
  next miss-by-omission is loud rather than silent.

### Migration notes

* **Legacy checkpoints served through pi05 / pi05_mem / pi06 / pi07 /
  pi07_paligemma change numerically.** Because those policies never resolved
  `config_version` (see Fixed), a pre-versioning checkpoint was being normalized
  with the current convention. It is now correctly read as legacy `0` — matching
  what the weights were trained under, and matching what the base-class policies
  (pi0, value, cosmos3) already did. Expect small output differences on such a
  checkpoint; they are the *correction*. To deliberately keep the previous
  behavior, set `--policy.config_version=1` explicitly, which is honored untouched.
* **Training data order changes for any run with `cfg.seed` set.** The sampler now
  draws from `cfg.seed` instead of PyTorch's default-constructed generator.
  Sampling is i.i.d. with replacement from the same weighted distribution, so this
  is a different draw from the same distribution rather than a distributional
  change — but two runs of the same config across this release are **not**
  step-for-step comparable, and a run resumed across it does not continue its
  previous stream. `cfg.seed = None` is bit-unchanged. Separately, and unchanged by
  this release: the sampler's position is never checkpointed, so a resumed run
  restarts its stream from the beginning rather than continuing it.
* **Validation metrics change, and the new numbers are the correction.** With the
  split now identical on every rank, `Validation/*` is measured on genuinely
  held-out frames for the first time — previously every reported validation frame
  was in some rank's training set. Expect validation loss to read **worse** than it
  did on the same data before this release, and `running_best` checkpoint selection
  to pick differently. Validation numbers are now also comparable across runs with
  different world sizes, and across a resume that changes world size.
* **A config with `n_action_steps > action_chunk` now fails at config time.** The
  model always decoded only `chunk_size` actions, so set `n_action_steps` to
  `action_chunk` — what such a run was effectively doing — to keep its behavior.
  See Fixed.

## [0.12.0] - 2026-07-24

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

[Unreleased]: https://github.com/TensorAuto/OpenTau/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/TensorAuto/OpenTau/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/TensorAuto/OpenTau/compare/v0.11.0...v0.12.0
