import importlib
from unittest.mock import MagicMock, patch

from lerobot.common.utils.import_utils import is_package_available


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
