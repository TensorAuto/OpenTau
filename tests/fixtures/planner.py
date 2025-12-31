import pytest
import torch

from opentau.planner import Memory


@pytest.fixture(scope="session")
def mem(request):
    len = request.param
    if len is None:
        return {"object": Memory(), "len": 1000}
    else:
        return {"object": Memory(len=len), "len": len}


@pytest.fixture(scope="session")
def dummy_data_gpt_inference(request):
    no_of_images, task, mem = request.param

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dict = {
        f"camera{i}": torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16, device=device)
        for i in range(no_of_images)
    }

    return {"image_dict": image_dict, "task": task, "mem": mem}
