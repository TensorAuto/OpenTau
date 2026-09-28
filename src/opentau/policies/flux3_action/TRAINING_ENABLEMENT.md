# Follow-up: what OpenTau needs before `flux3_action` can be finetuned

This port lands **inference and eval**. Finetuning needs one change in the dataset layer
that is deliberately *not* in this PR, because it touches a path shared by every policy
and deserves its own review. This note records what the change is, why OpenTau needs it,
and why it is smaller than it first appears.

## Why F3A needs something no other policy here needs

Every other policy in this repo consumes observations from the past and predicts actions
into the future. F3A predicts **video and actions jointly** — its trunk denoises action
tokens and video tokens in one packed sequence, and upstream marks both video streams
required:

```python
REQUIRED_CONTENT_STREAMS = ("video", "video_cond")   # config.py
```

So its training loss has a video term (`video_mse`, weighted by `video_loss_weight`)
whose **targets are the frames that come after the observation**. Upstream says so
directly, and raises if they are absent (`policy.py`, in `prepare`):

```python
if t != cfg.window_frames:
    raise ValueError(
        f"training needs {cfg.window_frames} frames per camera (observations plus future frames), got {t}"
    )
```

Inference needs none of this — the video tokens are denoised from noise conditioned on
the observed frame — which is exactly why eval works against the dataloader as it stands
and only training is blocked.

## What blocks it today

`datasets/factory.py::resolve_delta_timestamps` decides, per feature key, which frame
offsets the loader will fetch. Actions get a configurable forward horizon; cameras never
do. Every camera branch emits non-positive offsets:

```python
elif "camera" in standard_key or standard_key == "state":
    n_obs = cfg.dataset_mixture.n_obs_history
    if seq_len > 1:
        delta_timestamps[key] = [-(seq_len - 1 - t) * seq_stride / action_freq for t in range(seq_len)]
    elif n_obs is not None:
        delta_timestamps[key] = [-(n_obs - 1 - i) * interval / action_freq for i in range(n_obs)]
    else:
        delta_timestamps[key] = [0.0]
```

Actions, by contrast, read `cfg.policy.action_delta_indices`, which may be positive.
There is no `camera_delta_indices` equivalent, so a policy cannot ask for future frames
however it is configured.

## Why the change is small: the fetch layer is already sign-agnostic

The important part is that nothing *below* this assumes observation offsets are
non-positive. `LeRobotDataset::_get_query_indices_soft` applies whatever offsets it is
handed:

```python
query_indices = {key: np.clip(idx + delta_idx, ep_start, ep_end - 1) for key, delta_idx in delta_indices.items()}
padding = {
    f"{key}_is_pad": torch.tensor((idx + delta_idx < ep_start) | (idx + delta_idx >= ep_end), dtype=torch.bool)
    for key, delta_idx in delta_indices.items()
}
```

A **positive** offset clips at the episode *end* rather than the start and raises the
matching `_is_pad` flag, symmetrically with a negative one. `_add_padding_keys` is
generic over keys. So episode-boundary handling, padding flags and the windows-that-
overrun-the-episode logic all work already — F3A's own `_valid_windows` scans exactly
those `*_is_pad` flags to drop such windows.

The missing piece is only the **request** side.

## Proposed change

1. Add a `camera_delta_indices` property to `PreTrainedConfig`, defaulting to `None`,
   alongside the existing `observation_delta_indices` / `action_delta_indices`.
2. In `resolve_delta_timestamps`, when a policy supplies it, emit those offsets for
   camera keys instead of the history-only window.
3. Have `Flux3ActionConfig` return the window its `PolicyConfig.window_frames` implies.
4. Test that a positive offset clips at the episode end and sets `_is_pad` — the
   mirror of the existing start-of-episode behaviour.

Two things to weigh in that review, neither of which is a blocker:

- **Throughput.** Each extra frame per camera is an extra decode per sample. F3A's DROID
  layout is three cameras, so the cost is real and worth measuring before it becomes a
  default anywhere.
- **Not F3A-specific.** Any future world-model policy that supervises predicted frames
  needs the same knob, which is the argument for putting it on `PreTrainedConfig` rather
  than special-casing one policy inside the dataset factory.

## Why not just fold it into this PR

It changes a function every policy's data path runs through, to enable a policy that
cannot train until several other things land as well (a dataset with the right camera
layout, the trunk checkpoints, a validated config). Shipping it separately keeps the
blast radius of a shared-path change reviewable on its own terms, and keeps this PR to
what it can actually demonstrate: a registered policy that builds, loads and runs.
