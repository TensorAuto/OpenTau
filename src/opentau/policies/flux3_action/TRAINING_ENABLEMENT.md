# Training `flux3_action`: the camera window

The policy port landed inference and eval. This note records the one dataset-layer change
finetuning needed, why it was needed, and what still stands between here and a real run.

## Why F3A needs something no other policy here needs

Every other policy in this repo consumes observations from the past and predicts actions
into the future. F3A predicts **video and actions jointly** — its trunk denoises action
tokens and video tokens in one packed sequence, and upstream marks both video streams
required:

```python
REQUIRED_CONTENT_STREAMS = ("video", "video_cond")   # config.py
```

So its training loss carries a video term (`video_mse`, weighted by `video_loss_weight`)
whose **targets are the frames after the observation**. Upstream says so directly, and
raises rather than guessing (`policy.py`, in `prepare`):

```python
if t != cfg.window_frames:
    raise ValueError(
        f"training needs {cfg.window_frames} frames per camera (observations plus future frames), got {t}"
    )
```

Inference needs none of this — video tokens are denoised from noise conditioned on the
observed frame — which is why eval worked against the dataloader unchanged and only
training was blocked.

## What was missing, and why the fix was small

`resolve_delta_timestamps` decides which frame offsets the loader fetches per feature key.
Actions read `policy.action_delta_indices`, which may be positive; cameras only ever got
the observation history, whose offsets are all `<= 0`. There was no way to ask for a
future frame.

The **fetch** layer was already sign-agnostic: `LeRobotDataset._get_query_indices_soft`
applies whatever offsets it is handed, clipping `idx + delta` into the episode and raising
`<key>_is_pad` at whichever end overruns. A positive offset clamps at the episode *end*
exactly as a negative one clamps at the start, so that layer needed nothing.

The **standardization** layer did, and an earlier draft of this note wrongly said it did
not. `_standardize_images` decides single- vs multi-frame from the mixture-level knobs
only, so a policy-owned window — which by construction has `n_obs_history` unset and
`sequence_length == 1` — fell through to the scalar path and died on the first training
batch with `a Tensor with 33 elements cannot be converted to Scalar`. The three mechanisms
now resolve through one `temporal_camera_frames` property so a fourth cannot repeat it.

### A known gap, not yet closed

Per-frame camera padding is **not** visible to the policy. `_standardize_images`
deliberately reduces camera `_is_pad` to "is this camera slot absent", discarding which
*frames* were clamped at an episode boundary — and F3A's `_valid_windows` scans only the
flags that survive. Because the camera window runs to `+chunk_size` while `action_is_pad`
stops at `chunk_size - 1`, there is exactly one sample position per episode where the
actions are entirely in-bounds but the final video target is a clamped duplicate of the
last real frame, and nothing marks it.

One position per episode is small, but it is silent, so it is recorded here rather than
assumed harmless. Closing it means surfacing per-frame camera pads (additively, so the
existing slot-level semantics are untouched) or restricting window starts so the camera
window fits — neither belongs in this PR, and neither matters until training actually runs.

## What the change is

`PreTrainedConfig.camera_delta_indices` — a concrete property defaulting to `None`,
deliberately **not** a fourth `@abc.abstractproperty`, so no existing policy config has to
implement it to say "no". At `None` the old history path runs unchanged.

`Flux3ActionConfig` returns the window its `window_frames` implies:

```python
n_obs = self.n_obs_steps if self.inference_profile == "history" else 1
return list(range(-(n_obs - 1), self.chunk_size + 1))
```

Derived independently of `window_frames` and then asserted equal to it, so a drift between
the two fails in the CPU suite rather than at the first training forward.

Constraints, each pinned by a test: cameras only (a forward window on **state** would feed
the policy future joint positions — a label leak no shape check catches); rejected
alongside `n_obs_history` or `sequence_length > 1`, which define the camera window too; and
rejected when unsorted, since frames return in offset order and are stacked as a time axis.

## Fine-tuning: what matches their recipe, and what does not

The policy defaults, the optimizer settings and the learning-rate schedule now follow
Black Forest Labs' published DROID recipe.

Getting the schedule there required two fixes to the shared training path, both of which
were silently blocking any policy that wants per-group hyperparameters:

* `optim/factory.py` filtered its input with `p.requires_grad`, which raises on the
  param-group dicts `get_optim_params()` returns. The filter now reaches inside the
  groups, so the recipe's **5x head learning rate** actually reaches the optimizer — it
  previously could not, meaning the policy could not be trained at all.
* `configs/train.py` overwrote an explicitly chosen `scheduler` whenever
  `use_policy_training_preset` was set. Since that is the only path yielding param groups,
  "per-group learning rates" and "a specific schedule" were mutually exclusive. The preset
  now fills only what the config left unset.

With both in place, `hold_warmup_constant_cooldown` reproduces the recipe's curve: the
pretrained trunk held at zero LR through 1,000 updates then warming to 3,000, the freshly
initialized heads skipping the hold and warming over 1,000, both flat to 25,000 and cooling
linearly to 30,000. It is opt-in — name it in a train config's `scheduler` field.

Two differences remain:

* **Trainer.** Their recipe uses a standalone FSDP2/HSDP trainer with power EMA at
  sigma_rel 0.10/0.05, a 2,048-window global batch and 30,000 updates across 8 GPUs.
  OpenTau trains through accelerate/DeepSpeed. The optimizer, schedule, losses, timestep
  sampling and data window now match; the distribution strategy and EMA do not, so a run
  here will not reproduce their curve exactly.
* **Scale, and one unvalidated step.** A fine-tune at their batch size needs many GPUs,
  not one. Separately, a single training *forward* has not yet been executed end to end:
  the video VAE's NATTEN attention has no efficient CPU path, so a 33-frame window is
  GPU-only, and every attempt during this work ran into a co-tenant holding most of the
  card — the weights place at 16.5 GB, leaving too little for VAE activations. The
  optimizer path is verified; the forward is not.

## Four parameters never receive a gradient, and it matters at scale

Running an actual step (`test_a_training_step_runs_and_moves_both_param_groups`) turned up
something reading the code did not: four of the DiT's output heads take no gradient, on
every step, by construction.

```
model.dit.final_layer.video_cond.linear.weight
model.dit.final_layer.video_cond.adaLN_modulation.1.weight
model.dit.final_layer.action_cond.linear.weight
model.dit.final_layer.action_cond.adaLN_modulation.1.weight
```

The DiT builds one output head per stream, but only the two *content* streams are denoising
targets. `video_cond` and `action_cond` carry clean context the trunk attends to, so
`flow_loss` never reads their projections and autograd never reaches them.

Locally this is harmless -- AdamW skips a parameter whose `.grad` is `None`, so the weights
simply stay at their checkpoint values. Distributed, it is not local at all:

* **DDP** needs `find_unused_parameters=True` or the reducer raises. OpenTau already
  defaults `FIND_UNUSED_PARAMS` to true, so a run works as shipped -- but the audit that
  `scripts/find_unused_params.py` exists for cannot be passed here, and the ~10-15% per-step
  graph walk it buys back is not available to this policy.
* **Parameter-sharding backends (DeepSpeed ZeRO-3, FSDP)** are the real concern. Their
  reduce-scatter / reshard hooks can wait on a gradient that never arrives. `train.py`
  fails fast on the one case it knows about -- a zero entry in `loss_weighting` -- and this
  is the same hazard arriving by a different route, which that guard does not cover. BFL
  train at 8 GPUs under FSDP2, so this is precisely the configuration a faithful
  reproduction reaches for.

**Not fixed here, deliberately.** Freezing the four would remove the hazard and change
nothing numerically today. It is not done because "these heads are never supervised" is
verified for the default profile only: the history profile could not be instantiated to
check (it needs a `command_history` batch key nothing emits yet, see below), and a freeze
that turns out to be wrong there would silently stop training parameters instead of failing.
The set is pinned by name in the test, so an upstream change that supervises them -- or adds
another unused head -- fails rather than drifts.

## What still stands between here and a trained policy

1. ~~Nothing has loaded real weights yet.~~ **Done.** The unmodified
   `black-forest-labs/flux-3-action-droid` package loads under `strict=True` (6.95B
   parameters, every key matched) and predicts a 32-step chunk on a real DROID episode
   with MAE 0.026 against recorded actions of scale 0.70. What remains unvalidated is the
   *training* forward, which needs a GPU (see above).
2. **Throughput is unmeasured.** 33 frames per camera across three DROID cameras is
   roughly a 33x increase in camera decodes per sample. Inherent to a joint video-action
   objective, but it should be measured before anyone plans a run around it.
3. **No validated train config.** `configs/examples/flux3_action_*.json` should be written
   against real checkpoints, not plausible defaults.
4. **Determinism (CLAUDE.md rule 3) is now checkable.** It was not while the policy could
   not train; once it can, a same-seed smoke run should be bit-identical twice.
5. **The history profile is configurable but still not trainable.** Making the camera
   window authoritative removed the config dead end, so
   `Flux3ActionConfig(inference_profile="history")` now validates — but upstream's
   `prepare` additionally requires a `command_history` batch key
   (*"history training requires absolute command_history"*) and nothing in the dataset
   layer emits one. It fails loudly at the first forward rather than silently, so this is
   a gap in scope rather than a correctness risk; the default profile is unaffected.
