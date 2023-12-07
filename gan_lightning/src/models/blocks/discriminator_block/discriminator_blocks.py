import torch
from torch import nn


class simple_1d_discriminator_block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(simple_1d_discriminator_block, self).__init__()  # Call the superclass's __init__ method
        self.block = nn.Sequential(nn.Linear(input_channels, output_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.block(x)


class deepconv_discriminator_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 2,
        kernel_size: int = 4,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_block = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride))

    def forward(self, x: torch.Tensor, is_final: bool = False):
        return self.final_block(x) if is_final else self.block(x)
