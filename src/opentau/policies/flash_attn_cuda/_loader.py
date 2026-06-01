# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy JIT compilation of the custom block-causal flash-attention CUDA kernel.

The extension is compiled on first use via :func:`torch.utils.cpp_extension.load`
and cached process-wide. Compilation is intentionally lazy (not at import time)
so importing this package on a CPU-only box, or in the CPU CI subset, never
triggers ``nvcc``. :func:`is_available` reports whether the kernel compiled (used
by tests to skip GPU cases); the ``flash_cuda`` attention pathway does **not**
fall back to sdpa/eager — if the kernel is unavailable, calling it raises.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_CSRC = Path(__file__).parent / "_csrc" / "flash_blockmask.cu"

_lock = threading.Lock()
_ext = None  # compiled module once loaded
_load_attempted = False
_load_error: str | None = None


def _do_load():
    """Compile + load the extension. Returns the module or raises."""
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Allow opting out (e.g. to force the sdpa fallback) without code changes.
    if os.environ.get("OPENTAU_DISABLE_FLASH_CUDA", "0") == "1":
        raise RuntimeError("flash_cuda disabled via OPENTAU_DISABLE_FLASH_CUDA=1")

    verbose = os.environ.get("OPENTAU_FLASH_CUDA_VERBOSE", "0") == "1"
    # nosec B614: this is torch.utils.cpp_extension.load (JIT-compiles our own .cu
    # source), not torch.load — no pickle/deserialization of untrusted data.
    return load(  # nosec B614
        name="opentau_flash_blockmask",
        sources=[str(_CSRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )


def get_extension():
    """Return the compiled extension module, or ``None`` if unavailable.

    Thread-safe and memoized: the build is attempted at most once per process.
    """
    global _ext, _load_attempted, _load_error
    if _ext is not None:
        return _ext
    if _load_attempted:
        return None
    with _lock:
        if _ext is not None:
            return _ext
        if _load_attempted:
            return None
        _load_attempted = True
        try:
            _ext = _do_load()
            logger.info("Compiled custom flash-attention CUDA kernel (opentau_flash_blockmask).")
        except Exception as e:  # noqa: BLE001 - record any build/runtime failure
            _load_error = str(e)
            logger.warning(
                "Custom flash-attention CUDA kernel failed to compile (%s); "
                "selecting attention_implementation='flash_cuda' will now raise.",
                _load_error,
            )
            _ext = None
        return _ext


def is_available() -> bool:
    """True if the custom flash-attention CUDA kernel compiled and is usable."""
    return get_extension() is not None


def load_error() -> str | None:
    """The last compilation error string, if any (for diagnostics/tests)."""
    return _load_error
