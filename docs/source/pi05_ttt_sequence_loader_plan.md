# pi05_ttt sequence dataloader — Step 0 findings and contract

Status: **Step 0 complete.** No code written yet. This records what already exists,
so the implementation steps can shrink to what is genuinely missing.

---

## 1. What already exists

`delta_timestamps` is live in OpenTau and is what `pi05_mem` runs on. Three parts
of the problem are already solved by it, and solved in production code rather than
in something we would be writing fresh.

**Relative-offset fetch.** `datasets/factory.py::resolve_delta_timestamps`
builds a per-feature list of time offsets, which the LeRobot fetch layer turns
into a stacked tensor:

```python
# actions: the current frame's chunk only
delta_timestamps[key] = [i / action_freq for i in cfg.policy.action_delta_indices]
# cameras and state: a history window ending at the current frame
delta_timestamps[key] = [-(n_obs - 1 - i) * interval / action_freq for i in range(n_obs)]
```

**Episode-boundary clamping, with padding reported.** This is the important one.
`lerobot_dataset.py:957` states it directly:

> Per-frame temporal padding info (from clamped episode boundaries) is tracked by
> `obs_history_is_pad`.

and `action_is_pad` comes from the same fetch layer (`lerobot_dataset.py:1032`).
So a window that would run off the end of an episode is clamped, and the batch
says which entries were clamped.

**Temporal-dim normalization.** `n_obs_history == 1` collapses to `(C, H, W)` in
the fetch path and is re-expanded, so the `(T, C, H, W)` contract holds at `T == 1`
too (`lerobot_dataset.py:945-953`).

## 2. What is missing

Exactly one thing: **per-timestep action targets.**

`action_delta_indices` is `list(range(chunk_size))` — one chunk, for the current
frame. Sequence training needs `T` supervised timesteps, each with its own chunk
and its own loss contribution, i.e. `(T, chunk, dim)` rather than `(chunk, dim)`.

## 3. Consequence: three of the planned steps shrink or disappear

| Planned step | Revised |
|---|---|
| Step 2 — `Dataset` wrapper calling `__getitem__` T times | **Not the implementation.** Build it only as the *reference* to diff against in Step 4.1. T separate fetches pay the video decode T times, and decode dominates step time. |
| Step 3 — derive episode bounds from `episode_index` / `meta/episodes.jsonl`; a window spanning two episodes silently corrupts memory | **Already handled.** The fetch layer clamps and reports. No `starts` precomputation, no boundary arithmetic. |
| Step 3 — `_is_pad` masks need a T axis | **Already produced** (`obs_history_is_pad`, `action_is_pad`), from the same clamping. |

What remains real from Step 3: **batch-major ordering** (already pinned by tests
in #530), **which keys get a T axis**, and the new **`(B, T)` timestep loss mask**.

## 4. Design decision: anchor the window at its last timestep

Observation offsets in the existing mechanism are all `<= 0` — a history window
ending at the current frame. Anchoring the sequence window the same way means
observations need **no new code at all**: timestep `t`'s observation offset is
`-(T - 1 - t) * stride`, which is exactly the existing formula with
`interval = stride`.

Only the action offsets are new:

```python
# timestep t, chunk position h  ->  one flat list of T * H offsets
[(-(T - 1 - t) * stride + h) / action_freq
 for t in range(T) for h in range(H)]
```

Fetched as `(T * H, dim)`, reshaped to `(T, H, dim)`. Batch-major by
construction, since `t` is the outer loop.

It also matches inference semantics: the memory is built from the past and the
policy predicts *now*, so the last timestep is the live one.

## 5. Why not reuse `n_obs_history`

Tempting — the observation offsets would be free — but two problems:

1. **Validation conflict.** `configs/train.py:427` requires
   `policy.n_obs_steps == dataset_mixture.n_obs_history`, and
   `PI05Config.__post_init__` raises when `n_obs_steps != 1`.
2. **Different meaning.** For `pi05_mem`, `n_obs_steps` is the number of frames
   *one* encoder forward sees. For `pi05_ttt`, the T timesteps are T separate
   forwards folded into the batch axis. Overloading the field would make the
   space-time video encoder and the TTT memory look like the same feature.

So: new `sequence_length` / `sequence_stride` fields on the dataset mixture,
reusing the `delta_timestamps` *code path* but not the *field*.

## 6. The contract the loader must satisfy

Taken from the scratch harness that already produced a working `(1, 4, 50, 32)`
batch, and from `PI05TTTPolicy._flatten_sequence_batch`.

| Key | Shape | Per timestep? |
|---|---|---|
| `camera0` (per camera) | `(B, T, 3, 224, 224)` | yes |
| `state` | `(B, T, max_state_dim)` | yes |
| `actions` | `(B, T, chunk_size, max_action_dim)` | yes |
| `action_is_pad` | `(B, T, chunk_size)` | yes |
| `obs_history_is_pad` | `(B, T)` | yes |
| `img_is_pad` | `(B, n_cameras)` | no — camera presence, not time |
| `loss_mask` **(new)** | `(B, T)` bool | consumed in `(B, T)` form |
| `prompt` / `task` | length-`B` list | no — shared across the trajectory |
| `dataset_index`, `real_action_dim` | `(B,)` | no — per trajectory |

**Row order is batch-major.** `_flatten_sequence_batch` folds `(B, T) -> (B * T)`
with `rearrange(v, "b t ... -> (b t) ...")`, and the TTT hook inverts it with
`rearrange(out, "(b t) s w -> b (t s) w", t=T)`. Time-major would run, train, and
be silently wrong — it would interleave trajectories inside one memory. Pinned by
`test_per_timestep_tensors_are_flattened_batch_major` and
`test_trajectories_do_not_interleave_at_batch_size_two`.

## 7. Revised step list

- [x] **Step 0** — check for `delta_timestamps`. Found; three steps shrink.
- [ ] **Step 1** — extend `resolve_delta_timestamps` with the T*H action offsets, behind the new config fields.
- [ ] **Step 2** — reshape `(T * H, dim) -> (T, H, dim)` in item assembly; add the `(B, T)` loss mask.
- [ ] **Step 3** — config plumbing: `sequence_length`, `sequence_stride`, defaults `1` and `chunk_size`; validate `sequence_length % tbptt_segment_length == 0`.
- [ ] **Step 4** — validate, in order:
  1. `T=1` reproduces the single-frame path **bit-exactly** (regression guard: proves the change is inert when off).
  2. `T=4` matches the harness contract in §6, shape and key set.
  3. Output at `t=3` differs from `t=0` on a window with repeated frames — otherwise TTT is not running.
  4. **Determinism** (CLAUDE.md rule 3, mandatory for `datasets/`): two seeded runs, per-step loss series bit-identical.
  5. 10 steps, `grad_norm` nonzero.
- [ ] **Step 5** — deferred perf: cache the frozen VLM prefix; per-segment backward (the `tbptt_backward_fn` hook was removed in review as unusable, so this is now a prerequisite for `T > ~32`, not a half-built feature).

## 8. Open question, worth settling before the perf work

The reviewer's closing point on #530: **measure `pi05_mem` as the baseline
first.** It is already in the repo, needs no dataloader change, and if a few
history frames with MEM-style token dropping close most of the gap, that changes
the case for this path. Cheaper than the sequence work and it makes any later
result interpretable.

Also unaddressed: collapsing `B` independent timesteps into `B/T` correlated
sequences changes gradient variance, so LR and warmup need re-tuning. Both
shipped configs currently inherit π₀.₅'s optimizer settings unexamined.
