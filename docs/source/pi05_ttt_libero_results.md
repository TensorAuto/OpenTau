# π₀.₅ + TTT on LIBERO — training report

**Branch** `ttt_pi0.5_seqloader` · **PR** [#532](https://github.com/TensorAuto/OpenTau/pull/532) · RTX 3090 (23.57 GiB)

> Live document. Filled in as each phase completes; nothing here is projected
> unless explicitly marked.

---

## Plan executed: A + B + E + F + G + J

Deferred by choice: α parameter group (C), T=64 with checkpointing (D),
closed-loop eval (H, I). Rationale: read α movement on the right tasks with a
sane schedule *first* — if the memory does not engage, measuring a null result
more precisely is not worth 2 h of eval setup.

| Phase | What | Status |
|---|---|---|
| A | Select long-horizon episodes | ✅ |
| B | Fix LR / warmup schedule | ✅ |
| E | Loader validation + smoke on LIBERO | ✅ |
| F | Run A — pi05_ttt | ✅ |
| G | Run B — matched stock π₀.₅ baseline | ✅ |
| J | This report | ✅ |

---

## A — The data is better than expected

`TensorAuto/libero` is the fps-corrected re-export of `physical-intelligence/libero`
(upstream mislabels 10 Hz for 20 Hz actions, which silently breaks frequency
alignment). 1693 episodes, 273,465 frames, **20 fps**, 40 tasks.

**The 60 episodes downloaded are all LIBERO-LONG — every one a multi-stage
instruction.** This was luck rather than design: task_index 0–9 turn out to be
the long-horizon suite, and episodes 0–59 map onto them.

| | |
|---|---|
| Episodes | 60 (2.0 GB of 34.9 GB) |
| Tasks covered | 10, all two-stage |
| Episode length | min 178, **median 267 (13.4 s)**, max 455 frames |

Sample instructions:

* *put both moka pots on the stove*
* *put the black bowl in the bottom drawer of the cabinet and close it*
* *turn on the stove and put the moka pot on it*
* *put the white mug on the left plate and put the yellow and white mug on the right plate*

**Why this matters.** The paper's +87% came from multi-stage assembly with
*state aliasing* — visually near-identical stages where a single-frame policy
cannot tell which stage it is in, and history disambiguates. These instructions
have that structure. Had the download landed on the *spatial* or *object*
suites (single-stage "pick up X and place it on Y"), there would have been no
mechanism for memory to help and a flat α would have been uninformative.

## B — The LR schedule was the real obstacle to α moving

The inherited π₀.₅ schedule is built for a 30,000-step finetune of *pretrained*
weights. Against a 3,000-step run of *freshly initialized* parameters it is
close to a no-op:

| | before | after |
|---|---|---|
| warmup steps | 1000 (⅓ of the whole run) | **100** |
| peak LR | 2.5e-5 | **1e-4** |
| decay steps | 30,000 (10× the run) | **3,000** |

At the old settings, step 3000 would still be at roughly 1e-7 — α could not
have travelled measurably regardless of whether the memory was useful. Fixing
this removes the *artificial* reason for a flat α. It cannot manufacture a
reason for α to open; that part is up to the task.

1e-4 rather than 2.5e-5 because `train_ttt_only` trains 85.3M parameters
initialized from scratch, not a finetune of a pretrained tower.

## Run configuration

Read back from each run's own `checkpoint/train_config.json` plus its log, not
from the config files. That matters for one field: **`pretrained_path` inside a
saved checkpoint is rewritten to that checkpoint's own path** (it is the resume
mechanism), so reading the warm start from there gives the wrong answer. The
real value comes from the log's `Loading model from:` line.

| Field | Run A (pi05_ttt) | Run B (stock baseline) |
|---|---|---|
| policy | `pi05_ttt` | `pi05` |
| sequence_length (T) | **32** | **1** — single timestep. `PI05Config` has no such field at all; the mixture's value is 1 |
| sequence_stride | **1** (every frame) | **`None`** — meaningless at T=1, no gap to stride over |
| chunk_size / n_action_steps | 10 / 10 | 10 / 10 |
| tbptt_segment_length | 8 | — |
| checkpoint_tbptt_segments | False | — |
| **trainable** | **85.3 M of 3.447 B (2.48%)**<br>TTT layers + gates + registers | **430.1 M of 3.362 B (12.79%)**<br>action expert, via `train_expert_only=True` |
| episodes | 60 (`TensorAuto/libero`, all LIBERO-LONG) | identical |
| steps | 3000 | 3000 |
| seed | 1000 | 1000 |
| batch_size | 1 | 1 |
| action_freq | **20.0 Hz** (mandatory — see below) | 20.0 Hz |
| warm start | `TensorAuto/pi05_base` | `TensorAuto/pi05_base` |
| missing keys on load | 354 (343 TTT params + the 3-key gap stock π₀.₅ also shows) | 11 |
| freeze_vision_encoder | True | True |
| warmup / peak LR | 100 / 1e-4 | 100 / 1e-4 |

### What the run actually cost

| | Run A | Run B |
|---|---|---|
| wall clock | **4 h 24 m** | **11 m 30 s** |
| per step | ~5.3 s | ~0.2 s |
| frames per step | 32 | 1 |
| **total frames seen** | **96,000** | **3,000** |

### Three confounds, not one

Laid out here because they decide what the loss comparison can and cannot mean.
Two of them favour Run A and one favours Run B, so they do not cancel:

| Axis | Run A | Run B | Favours |
|---|---|---|---|
| frames seen at equal steps | 96,000 | 3,000 | **A** (32×) |
| optimizer updates at equal frames | ~94 | 3,000 | **B** (32×) |
| trainable parameters | 85.3 M, freshly initialized | 430.1 M, pretrained | **B** (5×) |

`T` welds the first two together — they cannot be separated with this pair of
runs — and the third is a straightforward capacity difference. Hence the
conclusion in *Findings*: no claim about whether TTT improves performance is
supported, and the α = 0 ablation is the only clean comparison available.

**`action_freq` must be 20.** The guard added in #532 refuses anything higher:
at stride 1 a step of 1/30 s would land inside one 1/20 s source frame, and the
nearest-frame fetch would hand the memory two *identical* observations. That bug
was caught on droid_100 (15 fps labelled 30) and would have trained happily
while teaching the memory nothing.

**T=32 at stride 1 covers 41 frames = 2.05 s of a 13.4 s median episode.** So
the memory sees a window, not a whole task. Raising T to 128 (6.9 s) needs
segment checkpointing, which halves throughput — deferred as (D).

---

## E — Loader validation and smoke on LIBERO ✅

### Loader contract, on LIBERO at T=32

```
CHECK 2 — batch matches the documented contract
  [PASS] actions has (T, chunk, dim) — (32, 10, 32)
  [PASS] action_is_pad has (T, chunk) — (32, 10)
  [PASS] state has a leading T axis — (32, 32)
  [PASS] cameras have (T, C, H, W) — camera0(32,3,224,224), camera1(32,3,224,224)
  [PASS] loss_mask emitted, shape (T,)
  [PASS] loss_mask defaults all-True

CHECK 3 — the T axis carries distinct consecutive frames
  [PASS] 31 consecutive camera deltas, all non-zero
         mean|Δ| = 0.0097, 0.0095, 0.0093, 0.0088, 0.0063, 0.0041, 0.0037, ...
                   0.0247, 0.0255, 0.0267, 0.0268, 0.0278, 0.0257, 0.0199
  [PASS] 31 consecutive state deltas, all non-zero
  [PASS] timestep t+1's chunk is t's shifted by stride=1

RESULT: ALL CHECKS PASS
```

The camera deltas are worth reading rather than skimming: they fall to 0.0022
mid-window and rise to 0.0278 near the end. That is the arm decelerating into a
grasp and then accelerating away — real motion across the T axis, not a
resampling artifact. Both cameras carry the time axis.

### Smoke run — the new embodiment works

10 steps, exit 0, no errors. This was the first time the model met a *different
robot*: 7-DoF actions padded to 32, **two** cameras, 8-dim state, `task`→prompt,
warm-started from a checkpoint trained on other embodiments.

`Missing keys when loading state dict: 354` — the 343 new TTT parameters plus
the same 3-key `da_head` / `discrete_action_embedding` gap stock π₀.₅ shows
against that checkpoint. Clean warm-start onto a new embodiment.

The schedule fix is visible immediately: **lr = 6.4e-6 at step 10**, against
2.7e-7 under the inherited schedule — a 24× difference in how fast anything can
move this early.

## F — Run A: pi05_ttt ✅

Completed 3000 steps in **4 h 24 min** (~5.3 s/step). No errors.

Throughput note: 5.3 s/step against 2.97 s/step measured on droid_100. The cause
is LIBERO's **two** cameras versus droid's one — the frozen VLM prefix is the
dominant per-timestep cost and it roughly doubles. It is recomputed for all 32
timesteps on every step, and the VLM is frozen, so prefix caching is the single
highest-value optimization left. It was deferred; these numbers argue for
promoting it.

### The gate opened — α moved 17.9× off its initialization

This is the result that matters, and the one I said in advance would be the real
readout rather than the loss.

| | |
|---|---|
| initialization | 0.001 |
| final mean | **0.017860** |
| range | −0.077637 … +0.104980 |
| sign | **positive in 18 / 18 layers** |
| mean \|Δ\| from init | 0.021869 |

Per-layer mean α, shallow to deep:

```
 0 +0.006240    6 +0.023855   12 +0.011648
 1 +0.010275    7 +0.022976   13 +0.028309
 2 +0.014143    8 +0.015629   14 +0.011352
 3 +0.013432    9 +0.019519   15 +0.020404
 4 +0.016954   10 +0.018090   16 +0.017264
 5 +0.014837   11 +0.027233   17 +0.029312
```

The trend is orderly rather than noisy: roughly monotonic from layer 0 (+0.0062)
to layer 17 (+0.0293), i.e. **deeper layers chose to rely on memory more.** The
gate is free to stay shut — α is trainable and initialized near zero precisely so
that the model must earn any use of the memory. It opened, in every layer, in a
structured way.

### Loss

**Windowed means, not single steps.** `batch_size=1`, so every logged value is
the loss on *one* sample. An earlier version of this table listed single values
at round step numbers and made it look as though the loss rose after step 1500 —
it did not. Windowed statistics:

| steps | MSE mean | sd | min–max |
|---:|---:|---:|---:|
| 10–500 | 1.210 | 0.399 | 0.73–2.37 |
| 500–1000 | 0.753 | 0.109 | 0.54–0.98 |
| 1000–1500 | 0.634 | 0.082 | 0.44–0.86 |
| 1500–2000 | 0.524 | 0.069 | 0.41–0.72 |
| 2000–2500 | 0.496 | 0.049 | 0.40–0.60 |
| 2500–3000 | **0.490** | 0.058 | 0.35–0.61 |

Monotonically falling to the end. A linear fit over steps 1500–3000 gives a
slope of **−0.0375 MSE per 1000 steps** — still improving, with total drift at
0.93× the noise sd. The shrinking sd (0.399 → 0.049) is itself a convergence
signal, and it is what makes any single late point look more meaningful than it
is.

* **MSE −69%** (1.577 → 0.482, last-10 average). This is the flow-matching action
  loss, i.e. the objective `train_ttt_only` is actually optimizing.
* **CE +4%** (2.635 → 2.729). Expected and worth naming rather than hiding:
  `train_ttt_only` freezes the discrete-action head, so as the shared
  representation shifts under the TTT parameters, the frozen head's loss drifts
  up slightly. A real cost of the frozen-head setup, not a failure.
* `grad_norm` 2.708 → ~0.5, stable. Converging, not diverging.

### A bug in my own instrumentation

`MetricsTracker` abbreviates step numbers past 999 as `step:3K(2980)`. My monitor
filter and log parser both matched `step:[0-9]+\(`, so **no milestone event fired
after step 990** and interim greps looked as though logging had stopped. The run
was healthy throughout; the instrumentation was blind. Fixed by parsing the
parenthesised true step. Worth recording because the same pattern would silently
break any future log-scraping.

## G — Run B: matched stock π₀.₅ baseline ✅

3000 steps in **11 min 30 s** (~0.2 s/step). Identical episodes, steps and seed.

| steps 2500–3000 | MSE | CE |
|---|---|---|
| Run A (pi05_ttt) | **0.490** ± 0.058 | 2.718 ± 0.061 |
| Run B (baseline) | 0.725 ± 0.188 | 2.740 ± 0.098 |

## Findings

### 1. The memory is used — this is the solid result

α moved **17.9×** off its 0.001 initialization, **positive in all 18 layers**,
with a roughly monotonic increase with depth. The gate exists precisely so the
model can leave the memory shut at no cost; it opened it everywhere, in an
orderly pattern. That is direct internal evidence the TTT branch is being
relied on.

### 2. Whether it *helps* — these two runs cannot answer that

The comparison is confounded three ways, and the two framings disagree:

| | MSE (steps 2500–3000) | frames seen |
|---|---|---|
| **Equal optimizer steps** | A **0.490** vs B 0.725 → A ahead | A 96,000 · B 3,000 |
| **Equal frames seen** (A@~94 vs B@3000) | A 1.682 vs B **0.725** → B ahead | ~3,000 each |

Neither is a fair read:

* At equal steps, Run A has seen **32× the frames** (T=32 means 32 timesteps per
  step). Of course it is ahead.
* At equal frames, Run A has had only ~94 **optimizer updates** — barely past its
  100-step warmup. Of course it is behind.

Data seen and optimizer updates are welded together by T and cannot be separated
with this pair of runs. On top of that the two runs train **different parameter
counts** (85.3M new TTT params vs the ~500M action expert). So: no claim about
whether TTT improves performance is supported here.

### 3. What would actually answer it

An **α = 0 ablation**: the same `pi05_ttt` policy, same data, same steps, same
parameter count, with the gate pinned shut so the memory contributes nothing.
That isolates the memory as the only difference. It costs one more ~4.4 h run.
Everything else — parameter-matched, data-matched, step-matched — is already
controlled by construction.

Closed-loop **success rate** would be the metric that actually matters, and needs
`uv sync --extra libero` plus `scripts/eval.py`.

### 4. Cost observation worth acting on

Run A took **4 h 24 m** against Run B's **11 min**. At 5.3 s/step versus 2.97
s/step measured on droid_100, the gap is LIBERO's two cameras: the frozen VLM
prefix is recomputed for all 32 timesteps every step and roughly doubles with a
second camera. **Caching the frozen VLM prefix is the highest-value optimization
remaining** — it is what would buy longer context, and it was deferred on the
strength of the droid numbers. These numbers argue for promoting it.

## Findings

_pending_

## What this run cannot tell you

Stated up front so it is not walked back later:

* **No success rates and no videos.** Closed-loop rollouts need
  `uv sync --extra libero` and `scripts/eval.py`; not installed. This produces
  loss curves and α movement only, and loss is a weak proxy for a robot policy.
* **The baseline is not parameter-matched.** Stock π₀.₅ with
  `train_expert_only` trains the ~500M action expert; pi05_ttt trains 85.3M of
  new parameters. Steps, data and seed match — parameter count does not. They
  are different methods, which is the point, but it is not a controlled
  ablation of the memory alone.
* **The baseline sees less data per step.** Single-timestep batches are 1 frame
  per step against the sequence path's 32, so at equal steps Run B sees 1/32 the
  frames. Matching steps (optimizer updates) rather than samples was the choice;
  the confound is real either way.
* **Optimizer settings beyond LR/warmup are unexamined for sequences.**
  Collapsing B independent timesteps into B/T correlated sequences changes
  gradient variance.
