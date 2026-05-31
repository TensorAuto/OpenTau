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

"""Tests for the fake_tensor_training.py diagnostic script.

Run the script in a subprocess: ``FakeTensorContext`` installs process-global
monkey patches (``torch.isinf``, ``FakeTensor.numpy``, ``Beta.__init__``, and the
module-conversion flag) that are never undone, so exercising it in-process would
leak that state into the rest of the pytest session.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "src" / "opentau" / "scripts" / "fake_tensor_training.py"


@pytest.mark.slow
def test_fake_tensor_training_runs_on_cpu():
    """The no-allocation smoke should run end-to-end on CPU and exit cleanly.

    Guards against torch API drift in ``nn.Module._apply`` / FakeTensor handling
    (the swap-tensors regression this script hit on torch 2.10). ``--device=cpu``
    is explicit because ``auto_torch_device()`` returns ``mps`` on Apple Silicon.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--device=cpu"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    # Params must stay symbolic (no real allocation of the giant hidden dim) ...
    assert "FakeTensor" in result.stdout
    # ... and the symbolic training loop must reach the end.
    assert "Symbolic mean loss" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
