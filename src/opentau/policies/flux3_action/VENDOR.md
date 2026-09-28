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
inference, not just for decoding. NATTEN ships no PyPI wheels — only an sdist that
compiles CUDA kernels. Use the prebuilt wheel matching this stack exactly:

```
natten==0.21.6+torch2100cu128   # cp310, linux_x86_64, from https://whl.natten.org/
```

The wheel is pinned to torch 2.10.0 / cu128 and to linux-x86_64, so it must stay
marker-gated (as the `trt` extra already is) and a torch bump requires re-pinning it.
