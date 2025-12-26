from dataclasses import dataclass

from utils import grep_file

from lerobot.configs.parser import wrap

MISSING_KEYS = {
    "hf": {
        "normalize_inputs.buffer_state.mean",
        "normalize_inputs.buffer_state.std",
        "normalize_targets.buffer_actions.mean",
        "normalize_targets.buffer_actions.std",
        "unnormalize_outputs.buffer_actions.mean",
        "unnormalize_outputs.buffer_actions.std",
        "model.paligemma_with_expert.onboard_vision_encoder.conv1.weight",
        "model.paligemma_with_expert.onboard_vision_encoder.conv1.bias",
        "model.paligemma_with_expert.onboard_vision_encoder.conv2.weight",
        "model.paligemma_with_expert.onboard_vision_encoder.conv2.bias",
        "model.paligemma_with_expert.onboard_vision_encoder.conv3.weight",
        "model.paligemma_with_expert.onboard_vision_encoder.conv3.bias",
        "model.paligemma_with_expert.onboard_vision_encoder.conv4.weight",
        "model.paligemma_with_expert.onboard_vision_encoder.conv4.bias",
        "model.paligemma_with_expert.onboard_vision_encoder.final_linear.weight",
        "model.paligemma_with_expert.onboard_vision_encoder.final_linear.bias",
    },
    "local": None,
}

UNEXPECTED_KEYS = {
    "hf": None,
    "local": None,
}

if set(MISSING_KEYS.keys()) != set(UNEXPECTED_KEYS.keys()):
    print("Warning: MISSING_KEYS and UNEXPECTED_KEYS keys do not match. ")


@dataclass
class Arg:
    log_path: str
    source: str
    missing_pattern: str = r"Missing key\(s\) when loading model: (\[[^\]]*\])"
    unexpected_pattern: str = r"Unexpected key\(s\) when loading model: (\[[^\]]*\])"

    def __post_init__(self):
        if self.source not in MISSING_KEYS:
            raise ValueError(f"--source must present in {MISSING_KEYS.keys()}. Got {self.source}")


def check_keys(lists: list[list[str]], source: str, key_type: str):
    print(f"Checking {key_type} keys")

    if key_type == "missing":
        expected_keys = MISSING_KEYS[source]
    elif key_type == "unexpected":
        expected_keys = UNEXPECTED_KEYS[source]
    else:
        raise ValueError("key_type must be either 'missing' or 'unexpected'.")

    if expected_keys is None:
        assert not lists, f"Found {key_type} keys but expecting none"
    elif not lists:
        raise ValueError(f"No {key_type} keys found, should be {expected_keys}")
    else:
        for keys in lists:
            if set(keys) != expected_keys:
                raise ValueError(f"Found {key_type} keys {keys} but expecting {expected_keys}")
    print("Passed")


@wrap()
def main(arg: Arg) -> None:
    missing_lists = grep_file(arg.log_path, arg.missing_pattern, processor=eval)
    check_keys(missing_lists, arg.source, "missing")

    unexpected_lists = grep_file(arg.log_path, arg.unexpected_pattern, processor=eval)
    check_keys(unexpected_lists, arg.source, "unexpected")


if __name__ == "__main__":
    main()
