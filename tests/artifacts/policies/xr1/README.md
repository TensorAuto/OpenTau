# `xr1` reference-parity fixtures

Goldens captured from **Xiaomi-Robotics-1-RoboCasa365**
(`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`, Apache-2.0) and the inference code at
`XiaomiRobotics/Xiaomi-Robotics-1` @ `4da1db0`, used by `tests/policies/test_xr1_cpu.py`
and `tests/policies/test_xr1_gpu.py`.

Because the port's backbone is byte-equivalent to stock `transformers` and its DiT is a
line-for-line translation, the default gate is **bit-identity** (`torch.equal`), not a
tolerance. Every place a tolerance appears, the test docstring names the operation
responsible.

## Files

| File | Gate | What it holds |
|---|---|---|
| `prompt_tokens.json` | P1 | `input_ids` / `attention_mask` / `video_grid_thw` / the rendered string / the bf16 state, for 12 scenarios |
| `state_from_env.json` | P2 | 16-D `agent_pos` → 14-D XR-1 state pairs at float64, plus quaternion edge cases |
| `crop_resize.json` | P3 | The 0.95 centre-crop box and a sha256 over a fixed synthetic pattern |
| `dit_geometry_static.json` | P4 | The tau schedule, `dt`, the 256-d timestep sinusoid, and the query layout |
| `history_indices.json` | P5 | The clamping history index table for buffer lengths 1-10 |
| `checkpoint_manifest.json` | P6 / P7 | 1121 tensor names + shapes + dtypes, and a global weight sha256 |
| `prefix_kv_fingerprints.json` | P9 | Per-layer K/V content hashes and summary statistics |
| `dit_step_fingerprints.json` | P10 / P11 | Per-Euler-step `x_in` / `v` fingerprints and the executed 16×12 chunk |
| `replay_trace.json` | P12 | 16 open-loop chunks over a recorded observation stream |
| `scene_seeds.json` | P13 | The reference's per-episode scene-seed scheme |

Heavy tensors (`pixel_values_videos`, the 72 prefix K/V tensors, per-step DiT hiddens, the
replay trace) are **not** in the repo — they are written to the capture script's
`--out-heavy` directory and the GPU tests read them from `$XR1_FIXTURE_DIR`.

## Re-capturing

```bash
python tests/artifacts/policies/xr1/capture_reference_fixtures.py \
  --checkpoint ~/xiaomi_r1/ckpt_robocasa365 \
  --out-json   tests/artifacts/policies/xr1 \
  --out-heavy  ~/xiaomi_r1/fixtures
```

The observation stream is **synthetic and deterministic** (`synthetic_observation` in the
capture script), so the fixtures are reproducible without a RoboCasa install, and the GPU
tests regenerate the same frames by importing the capture module rather than storing them.

The bundle records `torch.__version__`, `torch.version.cuda` and the device name, and the
GPU tests **skip loudly** when they differ rather than silently loosening to a tolerance —
bit-identity is kernel-dependent.

## Running the parity gates

```bash
XR1_REFERENCE_CHECKPOINT=~/xiaomi_r1/ckpt_robocasa365 \
XR1_FIXTURE_DIR=~/xiaomi_r1/fixtures \
pytest -m gpu -n 0 tests/policies/test_xr1_gpu.py
```

Both variables are required; the tests skip with an explanatory message when either is
unset.

## The scene-seed contract (P13 / gate G4)

The reference's evaluator seeds each episode as

```
episode_seed = 7 + task_index * 50 + episode_index
```

where `task_index` is the task's position in
`robocasa.utils.dataset_registry.TASK_SET_REGISTRY["target50"]`. OpenTau's eval uses
`start_seed + episode_index`, so the two only roll **the same scenes** when the reference's
seeds are passed explicitly:

```bash
# CloseFridge at target50 index k, 10 episodes
python - <<'PY'
k = 3  # look up the task's index in TASK_SET_REGISTRY["target50"]
print(",".join(str(7 + k * 50 + i) for i in range(10)))
PY

MUJOCO_GL=egl accelerate launch --num_processes=1 src/opentau/scripts/eval.py \
  --config_path=configs/examples/xr1_robocasa365_eval_config.json \
  --env.task=CloseFridge --eval.seed_list=157,158,159,160,161,162,163,164,165,166 \
  --output_dir=outputs/xr1_parity/CloseFridge_n10
```

Without `--eval.seed_list`, per-task rates are not comparable at n=50 no matter how exact
the model parity is: the two implementations would be evaluating different scene sets.

## The object-registry contract (found the hard way)

OpenTau's RoboCasa env defaults `obj_registries` to `("lightwheel",)` because the objaverse
pack is ~30 GB and most setups skip it. That default **changes the scene**: with the same
reset seed, restricting the registries moves the robot's starting `base_position` from
`[1.4366, -3.1005, 0.7]` (RoboCasa's default `("objaverse", "lightwheel")`, and the
reference's) to `[1.4284, -3.2620, 0.7]` on CloseFridge seed 57. The construction seed the
reference passes to `gym.make` turns out to be irrelevant — it was the registries all along.

The consequence is not subtle. With the restricted default, `xr1` scored **0/2** on
CloseFridge; the reference evaluator scored **3/3** on the same box, the same seeds and the
same 900-step horizon. So a sim number produced under the restricted default is not
comparable to a published RoboCasa365 rate, whatever policy produced it.

`configs/examples/xr1_robocasa365_eval_config.json` therefore sets
`env.obj_registries = ["objaverse", "lightwheel"]`, and
`test_eval_config_requests_both_object_registries` pins it. Download the pack with
`python -m opentau.scripts.download_robocasa_assets` (it fetches `objs_objaverse` when the
config asks for it) before trusting any number from this ladder.

## The sim ladder (gate G4)

Stop at the first failure. Accept ranges are Wilson 95 % intervals around the published
per-task rates.

| Order | Task | Published | Accept | **Measured** | Why this task |
|---|---|---|---|---|---|
| 1 | CloseFridge, n=10 | 94 % | >= 5/10 | **10/10** | The canary. `P(X <= 4 \| p = .94) ~ 1e-5`, so it is near-proof of a bug at 1/5 the cost |
| 2 | CloseFridge, n=50 | 94 % | 42-50 | **49/50 (98 %)** | |
| 3 | TurnOnMicrowave, n=50 | 56 % | 21-34 | **24/50 (48 %)** | Most discriminating: shortest horizon (450) *and* mid-range, so it can fail in both directions |
| 4 | OpenDrawer, n=50 | 94 % | 42-50 | **48/50 (96 %)** | Second ceiling task, different scene distribution |
| 5 | CloseBlenderLid, n=50 | 36 % | 12-25 | **20/50 (40 %)** | Catches a bias that helps easy tasks and hurts hard ones |

All five rungs are inside their intervals (measured 2026-09-10, RTX 3090, one rank, batch 2,
`eval.seed_list` set to the reference's per-episode seeds). Deviations from the published
rates are +4, +4, -8, +2 and +4 percentage points — no directionally consistent shortfall,
and the one task below its published value is the mid-range rung that has the room to move
in either direction.

PrepareCoffee is deliberately skipped: at p = 0.16 and n = 50 its interval is ±10 pp, which
cannot distinguish 16 % from 6 %. One task outside its interval across four is ~1-in-5 by
chance, so a single outlier is not a failure — but a **directionally consistent shortfall
across all four** is, regardless, and any task below half its published value is a failure
outright.

The canary earned its keep twice. It first read **0/10**, which localized
`obj_registries` (above); then **4/10**, which localized a *second* harness bug — OpenTau's
`envs/robocasa.py::convert_action` sliced a base-first flat-12 while RoboCasa's own
converter and the RoboCasa365 datasets use EE-first, so every action reaching the simulator
was permuted (fixed; see `tests/envs/test_robocasa_action_layout.py`). With both fixed it
reads **10/10**. Neither bug is xr1-specific, and neither produces an error — the arm moves,
the episode runs, only the success rate falls.

Failing G4 with G3 (model parity) green means the divergence is in the *harness*, not the
model: camera order, instruction text, scene seeds, horizons, or the execution horizon.
That is a much narrower search, which is the point of gating in this order.
