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

import importlib
from unittest.mock import MagicMock, patch

from opentau.utils.import_utils import is_package_available


def test_package_available_standard():
    """
    Tests if is package available is able to load the local dependencies
    """
    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", return_value="1.2.3"),
    ):
        assert is_package_available("some_package") is True
        assert is_package_available("some_package", return_version=True) == (True, "1.2.3")


def test_package_not_available():
    """
    Tests if is package available is able to detect uninstall local dependencies
    """
    with patch("importlib.util.find_spec", return_value=None):
        assert is_package_available("non_existent_package") is False
        assert is_package_available("non_existent_package", return_version=True) == (False, "N/A")


def test_torch_dev_version():
    """
    Tests if is package available to handle torch dependencies
    """
    mock_torch = MagicMock()
    mock_torch.__version__ = "1.13.1+dev"
    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
        patch("importlib.import_module", return_value=mock_torch),
    ):
        assert is_package_available("torch") is True
        assert is_package_available("torch", return_version=True) == (True, "1.13.1+dev")


def test_torch_non_dev_version_fallback():
    """
    Tests if is package available to handle torch dependencies if invalid version
    """
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.0.0"
    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
        patch("importlib.import_module", return_value=mock_torch),
    ):
        assert is_package_available("torch") is False
        assert is_package_available("torch", return_version=True) == (False, "N/A")


def test_other_package_metadata_fails():
    """
    Tests if is package available
    """
    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
    ):
        assert is_package_available("another_package") is False
        assert is_package_available("another_package", return_version=True) == (False, "N/A")


def test_torch_import_fails():
    """
    Tests if is package available to handle torch dependencies with import errors
    """
    with (
        patch("importlib.util.find_spec", return_value=True),
        patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
        patch("importlib.import_module", side_effect=ImportError),
    ):
        assert is_package_available("torch") is False
        assert is_package_available("torch", return_version=True) == (False, "N/A")
