import sys
from argparse import ArgumentError
from unittest.mock import patch

import draccus
import pytest

from lerobot.configs.parser import filter_path_args, get_cli_overrides, parse_arg

PATH_KEY = "path"


@pytest.mark.parametrize(
    "field_name, input_args, expected_output",
    [
        # Test case 1: Basic extraction
        (
            "policy",
            ["--env.name=pusht", "--policy.lr=0.001", "--policy.optim.beta=0.9"],
            ["--lr=0.001", "--optim.beta=0.9"],
        ),
        # Test case 2: No matching arguments
        (
            "env",
            ["--policy.lr=0.001", "--policy.optim.beta=0.9"],
            [],
        ),
        # Test case 3: Mixed arguments
        (
            "training",
            ["--seed=123", "--training.batch_size=32", "--eval.episodes=10"],
            ["--batch_size=32"],
        ),
        # Test case 4: Argument with no value (flag)
        (
            "debug",
            ["--debug.verbose", "--run_name=test"],
            ["--verbose"],
        ),
        # Test case 5: Empty input args
        (
            "any",
            [],
            [],
        ),
    ],
)
def test_get_cli_overrides_basic_functionality(field_name, input_args, expected_output):
    """Tests the core logic of extracting and de-nesting arguments."""
    result = get_cli_overrides(field_name, args=input_args)
    assert result == expected_output


def test_exclusion_of_special_keys():
    """
    Tests that the special keys for draccus choice type and path are correctly excluded.
    """
    field_name = "policy"
    input_args = [
        f"--{field_name}.lr=0.001",
        f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=some_choice",  # Should be excluded
        f"--{field_name}.{PATH_KEY}=/some/path",  # Should be excluded
        f"--{field_name}.num_layers=4",
    ]
    expected_output = ["--lr=0.001", "--num_layers=4"]

    result = get_cli_overrides(field_name, args=input_args)
    assert result == expected_output


def test_uses_sys_argv_when_args_is_none():
    """
    Tests that the function correctly falls back to using sys.argv.
    """
    # Use mocker to patch sys.argv
    mock_argv = ["script_name.py", "--env.name=pusht", "--policy.lr=0.001"]
    with patch.object(sys, "argv", mock_argv):
        # Call the function without providing the 'args' parameter
        result = get_cli_overrides("policy")

        assert result == ["--lr=0.001"]


@pytest.mark.parametrize(
    "arg_name, input_args, expected_output",
    [
        # Test case 1: Standard case, argument found
        ("config", ["--config=path/to/file.yaml", "--mode=train"], "path/to/file.yaml"),
        # Test case 2: Argument is the only one
        ("user", ["--user=testuser"], "testuser"),
        # Test case 3: Argument not found
        ("password", ["--user=testuser", "--port=8080"], None),
        # Test case 4: Argument has an empty value
        ("api_key", ["--api_key=", "--user=admin"], ""),
        # Test case 5: Input list is empty
        ("any_arg", [], None),
        # Test case 6: Argument name is a substring of another, should not match
        ("env", ["--environment=prod"], None),
        # Test case 7: Argument is a flag (no '=') and should not be matched
        ("verbose", ["--verbose", "--config=c.yaml"], None),
    ],
)
def test_parse_arg_various_scenarios(arg_name, input_args, expected_output):
    """
    Tests the parse_arg function with a variety of inputs and expected outcomes.
    """
    result = parse_arg(arg_name, args=input_args)
    assert result == expected_output


def test_parse_arg_uses_sys_argv_when_args_is_none():
    """
    Tests that the function correctly falls back to using sys.argv when the
    'args' parameter is not provided.
    """
    # 1. Setup: Mock sys.argv to a known value
    mock_argv = ["name_of_script.py", "--user=system", "--data=cifar10"]
    with patch.object(sys, "argv", mock_argv):
        # 2. Act: Call the function without the 'args' parameter
        result = parse_arg("data")

        # 3. Assert: Check that it parsed the value from the mocked sys.argv
        assert result == "cifar10"


def test_parse_arg_returns_none_if_not_in_sys_argv():
    """
    Tests that the function returns None when the arg is not in the mocked sys.argv.
    """
    mock_argv = ["name_of_script.py", "--user=system"]
    with patch.object(sys, "argv", mock_argv):
        result = parse_arg("non_existent_arg")
        assert result is None


def test_does_not_filter_if_no_path_arg():
    """
    Tests that if a field does NOT have a .path arg, its args are NOT removed.
    """
    # Simulate get_path_arg returning None for the 'policy' field
    with (
        patch("lerobot.configs.parser.get_path_arg", return_value=None),
        patch("lerobot.configs.parser.get_type_arg", return_value="some_type"),
    ):
        args = ["--policy.lr=0.01", "--policy.type=some_type", "--env.name=pusht"]

        result = filter_path_args(fields_to_filter=["policy"], args=args)

        # "policy" had no path arg, so it should not be filtered.
        assert result == args


def test_raises_error_on_path_and_type_conflict():
    """
    Tests that an ArgumentError is raised if a field has both .path and .type args.
    """
    # Simulate both helpers returning a value for the 'policy' field

    with (
        patch("lerobot.configs.parser.get_path_arg", return_value="some/path"),
        patch("lerobot.configs.parser.get_type_arg", return_value="some_type"),
    ):
        args = [f"--policy.{PATH_KEY}=some/path", f"--policy.{draccus.CHOICE_TYPE_KEY}=some_type"]

        with pytest.raises(ArgumentError):
            filter_path_args(fields_to_filter=["policy"], args=args)


def test_handles_single_string_for_fields_to_filter():
    """
    Tests that the function works correctly when fields_to_filter is a single string.
    """

    with (
        patch("lerobot.configs.parser.get_path_arg", return_value="a/path"),
        patch("lerobot.configs.parser.get_type_arg", return_value=None),
    ):
        args = ["--config.path=a/path", "--config.name=test", "--seed=123"]

        # Pass a single string instead of a list
        result = filter_path_args(fields_to_filter="config", args=args)

        assert result == ["--seed=123"]


def test_handles_multiple_fields_to_filter():
    """
    Tests correct filtering when multiple fields have path arguments.
    """

    # We need a side_effect to simulate different return values for different calls
    def path_side_effect(field, args):
        if field == "policy":
            return "path/to/policy"
        if field == "env":
            return "path/to/env"
        return None

    with (
        patch("lerobot.configs.parser.get_path_arg", return_value=path_side_effect),
        patch("lerobot.configs.parser.get_type_arg", return_value=None),
    ):
        args = ["--policy.lr=0.01", "--env.name=pusht", "--seed=42", "--policy.path=..."]

        result = filter_path_args(fields_to_filter=["policy", "env", "training"], args=args)

        # Both policy and env args should be filtered out
        assert result == ["--seed=42"]
