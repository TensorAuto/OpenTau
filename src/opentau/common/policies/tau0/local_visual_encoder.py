import warnings

import torch
import torch.nn as nn
from einops import rearrange


class SmallCNN(nn.Module):
    """Small CNN image encoder for the action expert.

    This image encoder is used to quickly encode local camera images for the action expert.
    The number of layers is kept minimal to reduce the latency of the action expert.

    This model takes in 224x224 images in the range [-1, 1] and outputs 100 tokens of `output_size` dimensions.
    """

    def __init__(self, output_size: int = 1024):
        super().__init__()
        # Strided convolutions for downsampling: 224 -> 112 -> 56 -> 28
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 224 -> 112
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 112 -> 56
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 56 -> 28
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 28 -> 14

        # Avg pooling to get exactly 100 spatial locations (10x10)
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=1)

        # Linear layer to expand from 512 to `output_size` dimensions per token
        self.final_linear = nn.Linear(512, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x: The input tensor of shape (batch_size, 3, 224, 224) with values from [-1, 1].
        :return: A tensor of shape (batch_size, 100, `output_size`).
        """
        x = torch.relu(self.conv1(x))  # (batch_size, 64, 112, 112)
        x = torch.relu(self.conv2(x))  # (batch_size, 128, 56, 56)
        x = torch.relu(self.conv3(x))  # (batch_size, 256, 28, 28)
        x = torch.relu(self.conv4(x))  # (batch_size, 512, 14, 14)

        # Apply max pooling to get 10x10 = 100 spatial locations
        x = self.avg_pool(x)  # Shape: (batch_size, 512, 10, 10)

        # Rearrange to get 100 tokens
        x = rearrange(x, "b c h w -> b (h w) c")  # Shape: (batch_size, 100, 512)

        # Apply final linear layer to each token to expand to `output_size` dimensions
        x = self.final_linear(x)  # Shape: (batch_size, 100, `output_size`)

        return x


class R3M(nn.Module):
    """Pretrained R3M model detailed in https://arxiv.org/pdf/2203.12601.

    The model takes a tensor of shape (batch_size, channels, 224, 224)
    R3M expects image value to be unnormalized (0-255). Source: https://github.com/facebookresearch/r3m/blob/b2334e726887fa0206962d7984c69c5fb09cceab/r3m/example.py#L33
    """

    def __init__(self, output_size: int = 1024):
        """Initialize the R3M model.

        :param output_size: The dimension of the output tensor.
        """
        super().__init__()

        # Load the R3M model.
        # There are several deprecated warnings that are suppressed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import r3m

            # By default, r3m is loaded onto the GPU.
            # We want to load it onto the CPU so that accelerate can handle the device placement later.
            r3m.device = "cpu"
            self.r3m = r3m.load_r3m("resnet18")

        # strip off torch DataParallel and module wrapper
        self.r3m = self.r3m.module.convnet
        self.r3m.to("cpu")

        # R3M resnet 50 has 2048 output features
        # R3M resnet 18 has 512 output features
        # linear layer to reduce the output size to the desired size
        self.linear = nn.Linear(512, output_size)

    def _siglip_to_r3m_format(self, x: torch.Tensor) -> torch.Tensor:
        """TAU0Policy converts all images to SigLIP format [-1.0, 1.0] to R3M format [0, 255].

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :return: A tensor of shape (batch_size, channels, height, width).
        """
        return (x + 1.0) * 127.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor of shape (batch_size, 3, 224, 224) with values from [-1, 1].
        :return: A tensor of shape (batch_size, output_size).
        """
        x = self._siglip_to_r3m_format(x)
        x = self.r3m(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    cnn = SmallCNN()
    # generate a random input tensor of shape (2, 3, 224, 224)
    x = torch.rand((2, 3, 224, 224)) * 2 - 1.0
    # forward pass
    output = cnn(x)
    print(output.shape)  # should be (2, 100, 1024)

    r3m = R3M()
    # forward pass
    output = r3m(x)
    print(output.shape)  # should be (2, 1024)
