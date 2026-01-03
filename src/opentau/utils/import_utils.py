#!/usr/bin/env python

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
"""Utilities for checking package availability and versions.

This module provides functions to check if packages are installed and optionally
retrieve their versions without importing them, which is useful for conditional
imports and dependency checking.
"""

import importlib
import logging


def is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Check if a package is available and optionally return its version.

    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py

    This function checks if the package spec exists and grabs its version to
    avoid importing a local directory. Note: this doesn't work for all packages.

    Args:
        pkg_name: Name of the package to check.
        return_version: If True, return a tuple of (available, version).
            If False, return only the availability boolean. Defaults to False.

    Returns:
        If return_version is False, returns a boolean indicating availability.
        If return_version is True, returns a tuple of (available, version).
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_torch_available, _torch_version = is_package_available("torch", return_version=True)
_gym_xarm_available = is_package_available("gym_xarm")
_gym_aloha_available = is_package_available("gym_aloha")
_gym_pusht_available = is_package_available("gym_pusht")
