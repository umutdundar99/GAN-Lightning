import math
import torch

import torch.nn as nn
import lightning.pytorch as pl

from gan_lightning.src.models.blocks.generator_blocks import (
    deepconv_generator_block,
)
from gan_lightning.src.models import model_registration


@model_registration("Deep_Convolutional_Generator")
class DeepConv_Generator(pl.LightningModule):
    def __init__(
        self, input_dim: int = 10, img_channel: int = 1, hidden_dim: int = 32, **kwargs
    ):
        super().__init__(),
        self.input_size = kwargs.get("input_size", 28)

        if self.input_size == 64:
            kernel = [2, 2, 2, 2, 2, 2]
            stride = [2, 2, 2, 2, 2, 2]
            padding = [0, 0, 0, 0, 0, 0]
        elif self.input_size == 28:
            kernel = [3, 3, 4, 3, 3, 6]
            stride = [1, 1, 1, 1, 1, 2]
            padding = [0, 0, 0, 0, 0, 0]
        else:
            raise ValueError("Invalid input size")

        self.input_dim = input_dim
        self.generator = nn.Sequential(
            deepconv_generator_block(
                input_dim, hidden_dim * 2, stride=stride[0], kernel_size=kernel[0], padding=padding[0]
            ),
            deepconv_generator_block(
                hidden_dim * 2, hidden_dim * 4, stride=stride[1], kernel_size=kernel[1],padding=padding[1]
            ),
            deepconv_generator_block(
                hidden_dim * 4, hidden_dim * 8, stride=stride[2], kernel_size=kernel[2], padding=padding[2]
            ),
            deepconv_generator_block(
                hidden_dim * 8, hidden_dim * 4, stride=stride[3], kernel_size=kernel[3], padding=padding[3]
            ),
            deepconv_generator_block(
                hidden_dim * 4, hidden_dim*2, stride=stride[4], kernel_size=kernel[4], padding=padding[4]
            ),
        )

        self.final_block = deepconv_generator_block(
            hidden_dim*2,
            img_channel,
            stride=stride[5],
            kernel_size=kernel[5],
            padding=padding[5],
            final_block=True,
        )

    # O=(I−1)×stride−2×padding+kernel_size
    def forward(self, noise:None):

        if noise is None:
            noise = torch.randn(len(noise), self.input_dim)

        x = noise.view(len(noise), self.input_dim, 1, 1)
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
  

    def _calculate_kernel_stride(self, target_size, num_layers):

        initial_size = 1
        _kernel_size = []
        _stride = []

        for i in range(num_layers):
            output_size = math.ceil((target_size - (initial_size - 1)) / 2)
            stride = 2
            kernel_size = target_size - (output_size - 1) * stride

            _kernel_size.append(kernel_size)
            target_size = output_size

        return _kernel_size, _stride
