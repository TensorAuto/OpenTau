# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard rules for Claude in this repo

These override defaults — read them before running anything.

1. **Never start a real training run.** Real configs train for 100k+ steps on multi-A100 nodes — they cost money and burn hours, and a botched config wastes both. When you need to exercise the training path, only ever launch:
   - `configs/dev/dev_config.json` or `configs/dev/ci_config.json` (small `steps`, tiny batch, public dataset), or
   - `configs/examples/pi05_training_config.json` (already a smoke config: `steps: 40`, `batch_size: 2`), or
   - `src/opentau/scripts/fake_tensor_training.py` (uses `FakeTensorContext` from `utils/fake_tensor.py` — no real allocation, just shape inference).

   If you find yourself overriding `--steps` to anything > a few hundred, stop and ask.

2. **Default to the CPU test subset.** Local dev typically has no NVIDIA GPU; running the full suite will fail confusingly partway through GPU-only tests.
   - Local / CI-equivalent: `pytest -m "not gpu" -n auto` — this is what `cpu_test.yml` runs and what should always pass.
   - GPU-only subset: `pytest -m "gpu" -n 0` — only run on a CUDA box (the slurm `oracle` cluster or `mlbox`); on a Mac/CPU laptop it will fail at fixture setup. Markers are defined in `pyproject.toml::[tool.pytest.ini_options]`.
   - When CI shows a failure under `-m "gpu"` and you can't reproduce locally, say so explicitly — don't pretend you ran it.

3. **Verify determinism on any change to the training loop or model.** ML bugs hide in stochasticity: a bad change can still produce loss curves that *look* plausible. After touching anything in `scripts/train.py`, `policies/*/modeling_*.py`, `optim/`, or `datasets/sampler.py`, run a smoke config twice with the same `seed` and confirm the per-step loss series is bit-identical (not just "close"). Seeding utilities live in `src/opentau/utils/random_utils.py` (`set_seed`, `serialize_python_rng_state`, etc.). If two seeded runs diverge, that's a bug — investigate before claiming the change works.

4. **Prefer `einops` for tensor reshape / permutation / reduction.** `einops` (`>=0.8.0`, already a hard dependency in `pyproject.toml`) makes tensor shape transformations self-documenting; opaque `view` / `reshape` / `permute` / `transpose` / `squeeze` / `unsqueeze` chains are a frequent source of subtle bugs in VLA model code. When writing or modifying tensor manipulation code:
   - Use `einops.rearrange` instead of `.view(...)` / `.reshape(...)` / `.permute(...)` / `.transpose(...)` / `.squeeze(...)` / `.unsqueeze(...)` whenever the operation can be expressed as a named-axis pattern (e.g. `rearrange(x, "b t (h w) c -> b t h w c", h=H)` instead of `x.view(B, T, H, W, C)`).
   - Use `einops.reduce` instead of `.mean(dim=...)` / `.sum(dim=...)` / `.max(dim=...)` when the reduction axes are clearer as named patterns.
   - Use `einops.repeat` instead of `.expand(...)` / `.repeat(...)` / broadcasting tricks with `unsqueeze`.
   - Use `einops.einsum` for non-trivial contractions instead of `torch.einsum` with single-letter indices, since named axes survive copy-paste better.
   - Exceptions where plain torch ops are fine: simple `.flatten()` of a contiguous suffix, single-axis `.sum()` over the last dim, contiguity calls (`.contiguous()`), and shape-preserving ops (`.to(...)`, `.float()`). Don't rewrite these just for the sake of einops.
   - When matching a reference (HuggingFace `transformers`, upstream LeRobot, the original π model code), preserve the existing op style verbatim inside that block — readability gains are not worth diff churn against an upstream reference.

5. **Distributed forward/backward must keep collective counts aligned across ranks.** FSDP / ZeRO-3 hang at NCCL with mismatched all-gather sizes when ranks disagree on what `forward` does. The failure mode is silent at CPU smoke-test time and only manifests after collectives diverge on a real run, so the patterns below are mandatory — copy them when extending the relevant code, don't reinvent them:
   - **Per-rank branch decisions that fire collectives must be OR-reduced first.** When a `forward` takes a Python-level branch based on what the local micro-batch contains (e.g. `if has_response: embed_language_tokens(...)` in `embed_prefix`), use `_global_or_branch_decisions` in `src/opentau/policies/pi07/low_level/modeling_pi07_low_level.py` — one SUM all-reduce that both OR-reduces the per-rank decisions and asserts cross-rank presence agreement. Adding a new optional branch in distributed `forward` without going through it (or an equivalent pre-branch all-reduce) is the same bug.
   - **Composite forward units must be a single `nn.Module`.** Bundle multi-component decoder steps (e.g. a backbone layer paired with an action-expert layer) into one `nn.Module` so FSDP's all-gather hook prefetches every sub-component together — like `InterleavedDecoderLayer` in `src/opentau/policies/pi07/gemma3_with_expert.py`. Calling sub-components directly on a separately-wrapped layer (`layer.input_layernorm(...)`, `layer.self_attn.q_proj(...)`) bypasses the hook and triggers mismatched all-gather sizes across ranks.

## Project overview

OpenTau is Tensor's open-source PyTorch training toolchain for vision-language-action (VLA) models — a fork of LeRobot with extra capabilities (heterogeneous-dataset co-training, discrete actions for π₀.₅, knowledge insulation, dropout in PaliGemma, π*₀.₆-style RL, validation splits, profilers). Any LeRobot-compliant policy and dataset works directly. Pinned to **Python 3.10**.

## Environment & install

Dependency management is **`uv` (>= 0.8.4) only** — `pyproject.toml`/`uv.lock` are authoritative; do not edit `requirements.txt` (none exists) or call `pip install` against the env. The `>= 0.8.4` floor is enforced by `required-version` in `[tool.uv]` and is needed because `[tool.uv].extra-build-dependencies` (which pins `cmake<4` inside `egl-probe`'s PEP 517 build isolation) is only honored by uv 0.8.4+.

```bash
uv sync --extra dev --extra libero          # standard dev setup (matches CI)
uv sync --all-extras                         # everything (note: libero + urdf are mutually exclusive — see [tool.uv].conflicts)
source .venv/bin/activate
```

Re-run `uv sync` whenever `pyproject.toml`/`uv.lock` change. Add deps with `uv add <pkg>`; lock with `uv lock`.

Installable extras: `dev` (pre-commit, sphinx, pytest), `libero` (sim env — pulls a forked LIBERO from `shuheng-liu/LIBERO`, pins numpy<2), `urdf` (rerun ≥0.28, requires numpy 2.x — incompatible with `libero`), `trt` (TensorRT, Linux/Win x86_64 only).

## Common commands

### Testing

```bash
pytest -m "not gpu" -n auto                  # CPU suite (what cpu_test.yml runs)
pytest -m "gpu" -n 0                         # GPU suite (what gpu_test.yml runs)
pytest -sx tests/path/to/test_x.py::test_y   # single test, fail-fast, no capture
```

Markers defined in `pyproject.toml`: `slow` (>1s runtime), `gpu` (needs GPU). Tests requiring LIBERO need `LIBERO_CONFIG_PATH` set (CI points it at `.github/assets/libero`). CI also runs nightly regression tests on g6.12xlarge — see `.github/workflows/regression_test.yml`.

### Linting / pre-commit

```bash
pre-commit install                           # one-time
pre-commit run --all-files                   # run all hooks
```

The pre-commit suite runs ruff (lint + format, line-length 110, py310), pyupgrade, gitleaks, zizmor (GitHub Actions security), bandit (config in `[tool.bandit]`), and typos. **Run pre-commit *before* tests** — formatters can rewrite files mid-iteration.

### Training / eval / export (console scripts)

`pyproject.toml` defines four console entry points that wrap `accelerate launch` (defined in `src/opentau/scripts/launch.py`):

```bash
# Single-node multi-GPU training (uses 8 GPUs by default — edit num_processes)
opentau-train \
    --accelerate-config configs/examples/accelerate_ddp_config.yaml \
    --config_path=configs/examples/pi05_training_config.json

opentau-eval --accelerate-config <yaml> --config_path=<json>      # eval
opentau-export --config_path=<json>                                # ONNX export (no accelerate)
opentau-dataset-viz --config_path=<json>                           # dataset visualizer (no accelerate)
```

Configs are **draccus-based JSON** (see `configs/examples/`). Any field can be overridden on the CLI: `--policy.type=pi05 --batch_size=8 --steps=1000`. Use `pi_config.json` style for π₀, `pi05_*.json` for π₀.₅. Accelerate configs: `accelerate_ddp_config.yaml` (DDP) or `accelerate_deepspeed_config.yaml` (ZeRO).

Direct invocation (bypasses the launcher; useful in slurm/sbatch wrappers): `accelerate launch --config_file <yaml> src/opentau/scripts/train.py --config_path=<json>`.

### Diagnostics

Three drop-in profiling scripts read the same `TrainPipelineConfig`:

- `src/opentau/scripts/profile_step.py` — per-step wall-clock breakdown (forward / backward / optimizer / sync)
- `src/opentau/scripts/profile_dataloader.py` — dataloader-only ceiling, no model
- `src/opentau/scripts/find_unused_params.py` — lists params that DDP would drop with `find_unused_parameters=False`

`FIND_UNUSED_PARAMS=false` (env var, read in `train.py`) reclaims ~10-15% of step time once a config has been audited.

## Architecture

### Configuration system (draccus)

Everything starts from a `TrainPipelineConfig` (`src/opentau/configs/train.py`) decorated as a draccus dataclass. The `@parser.wrap()` decorator on `train()` / `eval()` is OpenTau's extension over draccus that supports loading a config from a local path or HuggingFace Hub via `--config_path=...`, then merging in CLI overrides. The pattern is **never instantiate config objects directly in scripts** — always go through `parser.wrap` so that hub paths, plugin discovery (`PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"`), and CLI overrides all flow correctly.

Key invariant on `TrainPipelineConfig`: `batch_size == dataloader_batch_size * gradient_accumulation_steps`. When sweeping per-rank batch, override all three together — the validator will reject inconsistent combinations. When using DeepSpeed, `gradient_accumulation_steps` must also match the value in the DeepSpeed JSON.

`PreTrainedConfig` (`configs/policies.py`) is the abstract base for policy configs. Subclasses register themselves via the `policy.type` choice key (draccus `CHOICE_TYPE_KEY`). To add a new policy: subclass `PreTrainedConfig`, subclass `PreTrainedPolicy`, and register both in `policies/factory.py::get_policy_class` / `get_policy_class_config`.

### Module layout (`src/opentau/`)

- `configs/` — dataclass configs (train, eval, policies, envs, optim, deployment, libero, ros2lerobot)
- `datasets/` — LeRobot-compatible datasets, `WeightedDatasetMixture` (heterogeneous co-training), VQA datasets, v1→v2 / v2→v2.1 converters under `v2/`, `v21/`
- `envs/` — gym/gymnasium envs (currently LIBERO); `factory.make_envs()`
- `optim/` — optimizer + LR-scheduler dataclass-configured factories
- `planner/` — high-level planner using `prompts.yaml`
- `policies/` — `pi0`, `pi05`, `pi05_mem`, `pi06`, `pi07/{high_level_planner,low_level}` (current π0.7 impl: Gemma 3 backbone + SpaceTime SigLIP video encoder; note `low_level/` — not `low_level_planner/`, since the low-level policy is a controller, not a planner), `pi07_paligemma/{high_level_planner,low_level_planner}` (legacy PaliGemma variant of π0.7 — kept for older checkpoints; a fix targeting π0.7 usually needs to land in `pi07/`, not here), `value`. Each subdir has a `configuration_*.py` and `modeling_*.py`. Vision backbone wrappers: `paligemma_with_expert.py` (pi0/pi05/pi05_mem/pi07_paligemma) and `gemma3_with_expert.py` (pi06/pi07).
- `scripts/` — entry points (train/eval/launch), profilers, dataset converters (LIBERO, ROS, human video, RECAP), gRPC inference server, ONNX/TensorRT export
- `utils/` — `accelerate_utils` (sets process-global accelerator for rank detection), `train_utils` (checkpoint save/load/prune), `logging_utils` (`AverageMeter`, `MetricsTracker`), `transformers_patch`, `monkey_patch`

### Training loop (`scripts/train.py`)

`train()` is the entry; uses `accelerate.Accelerator` for distributed coordination. Loss is a weighted sum of `MSE` (action regression) and `CE` (discrete-action / language) terms — weights from `cfg.loss_weighting`. Metrics use `MetricsTracker` whose `__setattr__` is **not idempotent** — setting `metrics.loss = x` calls `AverageMeter.update(x)`, so repeat assignments accumulate a running mean rather than overwriting. Every reduction goes through `accelerator.gather_for_metrics` for rank-0 logging.

`save_checkpoint` / `load_training_state` (in `utils/train_utils.py`) handle resume — when resuming, the checkpoint config takes precedence over CLI args. `prune_old_checkpoints` keeps the most recent N.

## PR / contribution workflow

- **Always run pre-commit hooks before running unit tests.** The hooks (ruff format, end-of-file-fixer, trailing-whitespace, pyupgrade) will rewrite files or prompt for fixes; running tests first wastes a run when the formatter then changes the source. Sequence: `pre-commit run --all-files` → `pytest`.
- **Commit messages: keep them simple. The first line must be under 80 characters.** No long preambles in the subject.
- **PRs must follow `.github/PULL_REQUEST_TEMPLATE.md` verbatim.** The `check-pr-checklist.yml` CI grep-matches the exact phrases for the docstring and policy-change checkboxes — do not rephrase them, do not delete sections. Fill in `What this does`, `How it was tested`, and `How to checkout & try`.
- Policy-related PRs require running GPU pytests + nightly regression tests; check both boxes.
- Required tests/lint that gate merge: `cpu_test.yml` (PRs/pushes) and `pre-commit.yml`. `gpu_test.yml` and `regression_test.yml` run nightly on AWS g6 runners (cron 10:00 UTC).
- Per-file ruff overrides in `pyproject.toml`: gRPC server (PascalCase methods, `N802`), `recordhuman_to_lerobot.py` (math-convention uppercase names, `N803`/`N806`).
- **Claude integration:** three workflows under `.github/workflows/` add bots — `claude-pr-review.yml` does auto-review on PR open/sync (single edit-in-place summary tagged `[claude-review]`), `claude-implement-fixes.yml` handles `@claude fix` and addresses feedback in one coalesced commit (replies tagged `[claude-fix]`), and `extract-claude-lessons.yml` does post-merge lessons extraction into `chore(claude): learn from #N` PRs.

## Reference paths

- Console-scripts: defined in `pyproject.toml [project.scripts]`
- Training entry: `src/opentau/scripts/launch.py::train` → `accelerate launch src/opentau/scripts/train.py`
- Example configs: `configs/examples/*.json`
- Local notebooks: `notebooks/pi05_training.ipynb`, `notebooks/pi05_evaluation_only.ipynb`
- Docs source (Sphinx, RTD-built): `docs/source/`
- Pretrained checkpoints: `huggingface.co/TensorAuto/*` (see README table)
