import functools

from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from src.opentau.utils.monkey_patch import (
    torch_fake_tensor_beta_validate_args_patch,
    torch_fake_tensor_is_inf_patch,
    torch_fake_tensor_module_to_patch,
    torch_fake_tensor_to_numpy_patch,
)

# Share the ShapeEnv instance across all FakeTensorContext instances
# Without this, each FakeTensor.item() call would start numbering from 0, which is wrong.
_shared_shape_env = ShapeEnv()


class FakeTensorContext:
    def __init__(self, allow_non_fake_inputs: bool = True):
        self.mode = FakeTensorMode(
            shape_env=_shared_shape_env,
            allow_non_fake_inputs=allow_non_fake_inputs,
        )
        torch_fake_tensor_module_to_patch()
        torch_fake_tensor_to_numpy_patch()
        torch_fake_tensor_beta_validate_args_patch()
        torch_fake_tensor_is_inf_patch()

    def __enter__(self):
        return self.mode.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.mode.__exit__(exc_type, exc_val, exc_tb)


def run_with_fake_tensor(fn):
    r"""Decorator to run a function with FakeTensor enabled."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with FakeTensorContext():
            return fn(*args, **kwargs)

    return wrapper
