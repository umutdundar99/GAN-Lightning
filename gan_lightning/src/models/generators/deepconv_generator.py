import torch

import torch.nn as nn
import lightning.pytorch as pl

from gan_lightning.src.models.blocks.generator_blocks import (
    deepconv_generator_block,
)
from gan_lightning.src.models import model_registration
from typing import Optional

@model_registration("Deep_Convolutional_Generator")
class DeepConv_Generator(pl.LightningModule):
    def __init__(
        self, input_dim: int = 10, img_channel: int = 1, hidden_dim: int = 32, **kwargs
    ):
        super().__init__(),
        self.input_size = kwargs.get("input_size", 64)

        kernel = [4, 4, 4, 4, 4]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]

        self.input_dim = input_dim
        self.generator = nn.Sequential(
            deepconv_generator_block(input_dim, hidden_dim*8, kernel_size=kernel[0], stride=stride[0], padding=padding[0]),
            deepconv_generator_block(hidden_dim*8, hidden_dim*4, kernel_size=kernel[1], stride=stride[1], padding=padding[1]),
            deepconv_generator_block(hidden_dim*4, hidden_dim*2, kernel_size=kernel[2], stride=stride[2], padding=padding[2]),
            deepconv_generator_block(hidden_dim*2, hidden_dim, kernel_size=kernel[3], stride=stride[3], padding=padding[3]),
        )
            
        self.final_block = deepconv_generator_block(
            hidden_dim,
            img_channel,
            stride=stride[4],
            kernel_size=kernel[4],
            padding=padding[4],
            final_block=True,
        )

    def forward(self, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn(len(noise), self.input_dim)
        x = noise.view(noise.shape[0], self.input_dim, 1, 1)
        x = self.generator(x)
        x = self.final_block(x)
        return x

    def weight_init(self, mode):
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                if mode == "normal":
                    nn.init.normal_(m.weight, 0, 0.02)
                elif mode == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif mode == "kaiming":
                    nn.init.kaiming_uniform_(m.weight)
                else:
                    raise ValueError("Invalid weight initialization mode")

            elif isinstance(m, nn.BatchNorm2d):
                if mode == "normal":
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)