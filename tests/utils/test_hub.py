import os

import pytest

from src.opentau.utils.hub import HubMixin
from tests.fixtures.utils import DummyHubMixin


@pytest.mark.parametrize(
    "create_hubmixin_instance",
    [
        (HubMixin(), False, None, None, False),
        (DummyHubMixin(), True, True, None, False),
        (DummyHubMixin(), True, True, "test", True),
    ],
    indirect=True,
)
def test_save_pretrained(get_huggingface_api, create_hubmixin_instance, tmp_path):
    """
    Tests save_pretrained method using dummy HubMixin class. It tries to store object in local directory and hugging face hub
    """

    hubmixin_instance = create_hubmixin_instance["instance"]
    ground_truth = create_hubmixin_instance["ground_truth"]
    _, _, hf_token = get_huggingface_api

    save_directory = tmp_path if create_hubmixin_instance["save_directory"] else None
    push_to_hub = create_hubmixin_instance["push_to_hub"]

    if ground_truth:
        try:
            hubmixin_instance.save_pretrained(save_directory, token=hf_token, push_to_hub=push_to_hub)
            assert os.path.exists(save_directory / "test_file")
        except Exception as e:
            pytest.fail(f"Save pretrained failed due to: {str(e)}")
    else:
        with pytest.raises(NotImplementedError):
            hubmixin_instance.save_pretrained("")


@pytest.mark.parametrize(
    "create_hubmixin_instance",
    [(HubMixin, False, None, None, None), (DummyHubMixin, True, None, None, None)],
    indirect=True,
)
def test_from_pretrained(create_hubmixin_instance):
    """
    Tests from_pretrained method using dummy HubMixin class. It tries to load object from hugging face hub
    """
    hubmixin_instance = create_hubmixin_instance["instance"]
    ground_truth = create_hubmixin_instance["ground_truth"]

    if ground_truth:
        try:
            hubmixin_instance.from_pretrained("config.json")
        except Exception as e:
            pytest.fail(f"Save pretrained failed due to: {str(e)}")
    else:
        with pytest.raises(NotImplementedError):
            hubmixin_instance.from_pretrained("")
