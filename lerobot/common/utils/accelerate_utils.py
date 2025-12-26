import warnings

from accelerate import Accelerator

_acc: Accelerator | None = None


def set_proc_accelerator(accelerator: Accelerator, allow_reset=False) -> None:
    global _acc

    assert isinstance(accelerator, Accelerator), (
        f"Expected an `Accelerator` got {type(accelerator)} with value {accelerator}."
    )
    if _acc is not None:
        if allow_reset:
            warnings.warn(
                "Resetting the accelerator. This could have unintended side effects.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise RuntimeError("Accelerator has already been set.")
    _acc = accelerator


def get_proc_accelerator() -> Accelerator:
    return _acc


def acc_print(*args, **kwargs):
    acc = get_proc_accelerator()
    if acc is None:
        print(*args, **kwargs)
    else:
        print(f"Acc[{acc.process_index} of {acc.num_processes}]", *args, **kwargs)
