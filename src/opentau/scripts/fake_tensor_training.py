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

from dataclasses import dataclass

import torch

from opentau.configs import parser
from opentau.utils.fake_tensor import FakeTensorContext
from opentau.utils.utils import auto_torch_device


@dataclass
class Args:
    device: str | None = None
    num_workers: int = 4
    batch_size: int = 2
    dim_in: int = 3
    dim_out: int = 5
    large_hidden_dim: int = 10**9


@parser.wrap()
def main(args: Args):
    device = args.device or auto_torch_device()
    data = [(torch.rand(args.dim_in), torch.rand(args.dim_out)) for _ in range(10)]
    dataloader = torch.utils.data.DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # ---------------------------------------------------------------
    # Everything inside this context will use FakeTensorMode
    with FakeTensorContext():
        # Create a model in FakeTensorContext shouldn't cost real memory for model parameters.
        model = torch.nn.Sequential(
            torch.nn.Linear(args.dim_in, args.large_hidden_dim),
            torch.nn.Linear(args.large_hidden_dim, args.large_hidden_dim),
            torch.nn.Linear(args.large_hidden_dim, args.dim_out),
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # End of FakeTensorContext
    # ---------------------------------------------------------------

    print("Model parameters: ")
    for name, param in model.named_parameters():
        print(f"{name}: {param})")
    print()

    losses = []
    for i, (x, y) in enumerate(dataloader):
        # ---------------------------------------------------------------
        # Everything inside this context will use FakeTensorMode
        # Ideally, we want to iterate the dataloader in FakeTensorMode as well.
        # However, it does not work with multiple workers due to some serialization issue.
        with FakeTensorContext():
            x = x.to(device=device)
            y = y.to(device=device)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()

            optimizer.step()
            losses.append(loss.item())
            print(
                f"Step {i}: symbolic loss = {loss.item()}, dummy numpy array = {loss.detach().cpu().numpy()}"
            )
        # End of FakeTensorContext
        # ---------------------------------------------------------------

    print("\nSymbolic mean loss:", sum(losses) / len(losses))


if __name__ == "__main__":
    main()
