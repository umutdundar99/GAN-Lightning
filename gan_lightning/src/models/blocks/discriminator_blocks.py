import torch
from torch import nn
from typing import Optional


class simple_1d_discriminator_block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(
            simple_1d_discriminator_block, self
        ).__init__()  # Call the superclass's __init__ method
        self.block = nn.Sequential(
            nn.Linear(input_channels, output_channels), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class deepconv_discriminator_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 2,
        kernel_size: int = 4,
        padding: Optional[int] = 0,
        final_block: bool = False,
    ):
        super().__init__()

        if final_block:
            self.block = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(
                    input_channels, output_channels, kernel_size, stride, padding
                ),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    input_channels, output_channels, kernel_size, stride, padding, bias=False
                ),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x: torch.Tensor):

        return self.block(x)
