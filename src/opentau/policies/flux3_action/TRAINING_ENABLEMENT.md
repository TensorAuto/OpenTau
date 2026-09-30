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

The policy defaults now follow Black Forest Labs' published DROID recipe. The
learning-rate **schedule** does not, and cannot yet — see below.

Four differences remain:

* **Learning-rate schedule.** Their recipe holds the pretrained trunk at zero LR for
  1,000 updates, warms it to 3,000, holds flat to 25,000 and cools linearly to 30,000,
  while the freshly initialized heads skip the hold and warm over 1,000. Reproducing that
  needs two curves over two parameter groups, and OpenTau's optimizer factory cannot
  currently deliver them: `use_policy_training_preset=False` builds the optimizer from
  flat `policy.parameters()` (one group), and `use_policy_training_preset=True` overwrites
  the pipeline's `scheduler` with the policy preset. A scheduler with the right shape was
  written and withdrawn from this PR for that reason; it belongs with the factory work.
* **`get_optim_params()` does not reach the optimizer.** Related and more immediate:
  `optim/factory.py` filters its input with `p.requires_grad`, which raises on the
  param-group dicts this policy returns. So the recipe's 5x head learning rate is not
  applied today either. This predates this PR and needs its own fix.

* **Trainer.** Their recipe uses a standalone FSDP2/HSDP trainer with power EMA at
  sigma_rel 0.10/0.05, a 2,048-window global batch and 30,000 updates across 8 GPUs.
  OpenTau trains through accelerate/DeepSpeed. The losses, timestep sampling and data
  window match; the distribution strategy and EMA do not, and neither does the schedule
  (above).
* **Scale.** A fine-tune at their batch size needs many GPUs, not one. A single training
  step is also not yet validated: the video VAE's NATTEN attention has no efficient CPU
  path, so a 33-frame window is GPU-only, and no single GPU with enough free memory was
  available during this work.

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
