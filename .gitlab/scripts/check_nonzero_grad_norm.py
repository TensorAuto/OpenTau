from dataclasses import dataclass

from utils import grep_file

from lerobot.configs.parser import wrap


@dataclass
class Arg:
    log_path: str
    expected_length: int
    re_pattern: str = r"grad_norm:([0-9.eE+-]+)"


@wrap()
def main(arg: Arg) -> None:
    grad_norm = grep_file(arg.log_path, arg.re_pattern, processor=float)
    assert len(grad_norm) == arg.expected_length, (
        f"Expected {arg.expected_length} grad_norms, found {len(grad_norm)} in {arg.log_path}."
    )
    assert all(g > 0 for g in grad_norm), f"All grad_norms should be greater than zero, got {grad_norm}."
