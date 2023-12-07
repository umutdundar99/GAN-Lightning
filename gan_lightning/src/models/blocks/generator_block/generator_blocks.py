import torch
from torch import nn


class simple_1d_generator_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.generator(x)


class deepconv_generator_block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2, kernel_size: int = 3):
        super(deepconv_generator_block, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, is_final: bool = False):
        return self.final_block(x) if is_final else self.block(x)
