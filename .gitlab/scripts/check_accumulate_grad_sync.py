from dataclasses import dataclass

from utils import grep_file

from lerobot.configs.parser import wrap


@dataclass
class Arg:
    log_path: str
    expected_length: int
    re_pattern: str = r"accelerator\.sync_gradients=(True|False)"
    gradient_accumulation_steps: int = 2


@wrap()
def main(arg: Arg) -> None:
    sync_grads = grep_file(arg.log_path, arg.re_pattern, processor=bool)
    assert len(sync_grads) == arg.expected_length, (
        f"Expected {arg.expected_length} sync_gradients, found {len(sync_grads)} in {arg.log_path}."
    )
    assert all(sg == ((i + 1) % arg.gradient_accumulation_steps == 0) for i, sg in enumerate(sync_grads)), (
        f"Sync gradients should be set according to "
        f"gradient_accumulation_steps={arg.gradient_accumulation_steps}, "
        f"got {sync_grads}."
    )
