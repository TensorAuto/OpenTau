# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Command-line argument parsing and configuration loading utilities.

This module provides utilities for parsing command-line arguments, loading
configuration files from local paths or the HuggingFace Hub, and handling
plugin discovery and loading. It extends draccus functionality with support
for path-based configuration loading and plugin system integration.
"""

import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from functools import wraps
from pathlib import Path
from typing import Sequence

import draccus

from opentau.utils.utils import has_method

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"
draccus.set_config_type("json")


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parse arguments from CLI at a given nested attribute level.

    This function extracts command-line arguments that are nested under a specific
    field name and returns them with the field name prefix removed.

    Args:
        field_name: The field name to extract nested arguments for.
        args: Sequence of command-line arguments to parse. If None, uses sys.argv[1:].
            Defaults to None.

    Returns:
        List of denested arguments with the field name prefix removed, or None if
        no matching arguments are found.

    Example:
        Supposing the main script was called with:
        ```
        python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path
        ```

        If called during execution of myscript.py, `get_cli_overrides("arg2")` will
        return:
        ```
        ["--subarg1=abc", "--subarg2=some/path"]
        ```
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    """Parse a single command-line argument value.

    Args:
        arg_name: Name of the argument to parse (without the '--' prefix).
        args: Sequence of command-line arguments to parse. If None, uses sys.argv[1:].
            Defaults to None.

    Returns:
        The value of the argument if found, or None if not found.

    Example:
        For command-line arguments `['--batch_size=32', '--lr=0.001']`:
        - `parse_arg('batch_size')` returns `'32'`
        - `parse_arg('lr')` returns `'0.001'`
        - `parse_arg('missing')` returns `None`
    """
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ['--env.discover_packages_path=my_package',
        ...         '--other_arg=value']
        >>> parse_plugin_args('discover_packages_path', args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")       # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg):
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    """Get the path argument for a given field name.

    This function extracts the path argument for a field, which is typically
    specified as `--field_name.path=some/path`.

    Args:
        field_name: The field name to get the path argument for.
        args: Sequence of command-line arguments to parse. If None, uses sys.argv[1:].
            Defaults to None.

    Returns:
        The path value if found, or None if not found.

    Example:
        For `--policy.path=/path/to/config`, `get_path_arg('policy')` returns
        `'/path/to/config'`.
    """
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    """Get the type argument for a given field name.

    This function extracts the type argument for a field, which is typically
    specified as `--field_name.type=SomeType`.

    Args:
        field_name: The field name to get the type argument for.
        args: Sequence of command-line arguments to parse. If None, uses sys.argv[1:].
            Defaults to None.

    Returns:
        The type value if found, or None if not found.

    Example:
        For `--policy.type=Pi0Config`, `get_type_arg('policy')` returns `'Pi0Config'`.
    """
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    """Filter out arguments matching a specific field name.

    Args:
        field_to_filter: The field name to filter out (without the '--' prefix).
        args: Sequence of command-line arguments to filter. If None, uses sys.argv[1:].
            Defaults to None.

    Returns:
        List of arguments with the specified field filtered out.

    Example:
        For `['--batch_size=32', '--lr=0.001', '--batch_size=64']`:
        `filter_arg('batch_size')` returns `['--lr=0.001']`.
    """
    if args is None:
        args = sys.argv[1:]
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """Filter command-line arguments related to fields with specific path arguments.

    This function removes all arguments related to specified fields when a path
    argument is present for those fields. It also validates that path and type
    arguments are not both specified for the same field.

    Args:
        fields_to_filter: A single field name or a list of field names whose
            arguments need to be filtered.
        args: The sequence of command-line arguments to be filtered. If None,
            uses sys.argv[1:]. Defaults to None.

    Returns:
        A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a
            type argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def filter_distributed_args(args: Sequence[str] | None = None) -> list[str]:
    """Filter out distributed training arguments.

    This function removes arguments that are automatically injected by distributed
    training frameworks (e.g., DeepSpeed, torchrun) but not recognized by the
    custom argument parser.

    Args:
        args: The sequence of command-line arguments to be filtered. If None,
            uses sys.argv[1:]. Defaults to None.

    Returns:
        A filtered list of arguments with distributed training arguments removed.

    Note:
        Filtered arguments include: local_rank, node_rank, master_addr, master_port,
        world_size, and rank.
    """
    if args is None:
        args = sys.argv[1:]

    # List of distributed training arguments to filter out
    distributed_args = [
        "--local_rank",
        "--local-rank",
        "--node_rank",
        "--node-rank",
        "--master_addr",
        "--master-addr",
        "--master_port",
        "--master-port",
        "--world_size",
        "--world-size",
        "--rank",
    ]

    filtered_args = []
    for arg in args:
        should_filter = False
        for distributed_arg in distributed_args:
            if arg.startswith(f"{distributed_arg}=") or arg == distributed_arg:
                should_filter = True
                break
        if not should_filter:
            filtered_args.append(arg)

    return filtered_args


def wrap(config_path: Path | None = None):
    """Wrap a function to handle configuration parsing with enhanced features.

    This decorator is similar to `draccus.wrap` but provides three additional features:

    1. Removes '.path' arguments from CLI to process them later
    2. If a 'config_path' is passed and the main config class has a 'from_pretrained'
       method, initializes it from there to allow fetching configs from the hub directly
    3. Loads plugins specified in CLI arguments. These plugins typically register
       their own subclasses of config classes, so that draccus can find the right
       class to instantiate from the CLI '.type' arguments

    Args:
        config_path: Optional path to a configuration file. If provided and the
            config class supports `from_pretrained`, will load from this path.
            Defaults to None.

    Returns:
        A decorator function that wraps the target function with enhanced configuration
        parsing capabilities.

    Note:
        This is a HACK wrapper around draccus.wrap to add custom functionality.
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                # Filter out distributed training arguments first
                cli_args = filter_distributed_args(cli_args)
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
