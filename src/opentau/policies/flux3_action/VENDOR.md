# Vendored code: `black-forest-labs/flux-action`

Everything in this directory except `configuration_flux3_action.py`,
`modeling_flux3_action.py`, `__init__.py` and this file is vendored from
upstream and is **byte-identical to it** apart from the three-line delta below.

| | |
|---|---|
| Upstream | https://github.com/black-forest-labs/flux-action |
| Commit | `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f` (2026-09-23) |
| License | Apache-2.0 (same as OpenTau; upstream headers retained verbatim) |
| Attribution | Upstream's `NOTICE` is carried alongside this file, as Apache-2.0 §4(d) requires of derivative works |
| Upstream root | `src/flux_action/` -> this directory |

## Why the subtree layout is preserved

Upstream's modules import each other relatively (`from ..models.positional import ...`).
Keeping `models/`, `processing/`, `inference/` and `checkpoints/` where upstream puts
them means those imports work unchanged, so re-syncing is a mechanical diff:

```bash
diff -r <upstream>/src/flux_action src/opentau/policies/flux3_action \
    --exclude=__pycache__ --exclude=__init__.py
```

That command should report **only** the three-line delta below (plus `Only in ...` lines
for the OpenTau-authored files). `__init__.py` is excluded because it exists on both
sides and ours is OpenTau-authored -- upstream's is a bare license header.

## The entire delta (3 lines, 2 files)

Upstream imports itself absolutely in three places, which cannot resolve under
`opentau.policies.flux3_action`. Each was rewritten to the equivalent relative import:

| File | Upstream | Here |
|---|---|---|
| `processing/packing.py:48` | `from flux_action.models.positional import ...` | `from ..models.positional import ...` |
| `models/video_vae.py:862` | `from flux_action.processing.packing import padded_chunk_length` | `from ..processing.packing import padded_chunk_length` |
| `models/video_vae.py:922` | `from flux_action.models.runtime import resolve_weights` | `from .runtime import resolve_weights` |

## Deliberately not vendored

- `models/transformer_inf_fp8r.py` — FP8r quantized inference (1006 LOC). Its only
  import site is lazy (`policy.py`, under `config.quantization == "fp8r"`), so omitting
  it costs nothing as long as that value is rejected — which
  `Flux3ActionConfig.__post_init__` does explicitly rather than leaving an ImportError.
- `training/`, `data/`, `serving/`, `cli.py` — OpenTau supplies its own training loop,
  datasets, inference server and entry points.

## Runtime dependency: NATTEN

`models/video_vae.py` imports `natten` at module scope with no fallback, and the video
VAE is mandatory (asserted in `policy.py`) on the **encode** path, so it is required at
inference, not just for decoding. It is a **required dependency** of OpenTau rather than
an extra — it resolves without conflict, so isolating it would only add a step — and is
marker-gated to linux x86_64 exactly as `torchcodec` and `onnxruntime-gpu` are. NATTEN
ships no PyPI wheels — only an sdist that compiles CUDA kernels — so the prebuilt wheel
matching this stack is pinned by URL:

```
natten==0.21.6+torch2100cu128   # cp310, linux_x86_64
```

`[tool.uv.sources]` pins it by its **GitHub release URL** -- that is the artifact
`uv.lock` records and the one to edit when re-pinning:

```
https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.6/natten-0.21.6%2Btorch2100cu128-cp310-cp310-linux_x86_64.whl
```

`https://whl.natten.org/` is only the *index* to browse when picking a new pin: it lists
one wheel per (python, torch, CUDA, arch) combination. Note 0.21.7 dropped its torch-2.10
builds, which is why 0.21.6 is pinned.

**To re-pin after a torch bump:** find the wheel for the new torch/CUDA pair in that
index, update the URL in `[tool.uv.sources]` and the version in the main `dependencies`
list, then `uv lock`. The wheel is specific to torch 2.10.0 / cu128 / cp310 /
linux-x86_64, so it must stay marker-gated. Because it is a *required* dependency, a
stale pin breaks `uv sync` project-wide rather than only for this policy.
