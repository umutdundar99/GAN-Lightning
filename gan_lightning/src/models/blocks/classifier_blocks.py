import torch
from torch import nn
from typing import Optional


class controllable_classifier_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size=4,
        stride=2,
        final_layer=False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, is_final: bool = False):
        if is_final:
            x = self.final_block(x)
        else:
            x = self.block(x)
        return x
