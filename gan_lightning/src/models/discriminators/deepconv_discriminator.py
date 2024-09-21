import lightning.pytorch as pl
import torch
from torch import nn
from gan_lightning.src.models.blocks.discriminator_blocks import (  # noqa
    deepconv_discriminator_block,
)

from gan_lightning.src.models import model_registration


@model_registration("Deep_Convolutional_Discriminator")
class DeepConv_Discriminator(pl.LightningModule):
    def __init__(
        self, img_channel: int = 1, hidden_dim: int = 32, input_size: int = 28, **kwargs
    ):
        super().__init__()
        
        if input_size == 28:
            kernel = [3, 4, 4, 4, 4]
            stride = [2, 2, 2, 2, 2]
            padding = [2, 2, 2, 2, 0]

        elif input_size == 64:
            kernel = [3, 3, 3, 3, 3]
            stride = [2, 2, 2, 2, 2]
            padding = [1, 1, 1, 1, 0]
        else:
            raise ValueError("Invalid input size")

        self.discriminator = nn.Sequential(
            deepconv_discriminator_block(
                img_channel,
                hidden_dim,
                stride=stride[0],
                kernel_size=kernel[0],
                padding=padding[0],
            ),
            deepconv_discriminator_block(
                hidden_dim,
                hidden_dim * 2,
                stride=stride[1],
                kernel_size=kernel[1],
                padding=padding[1],
            ),
            deepconv_discriminator_block(
                hidden_dim * 2,
                hidden_dim * 4,
                stride=stride[2],
                kernel_size=kernel[2],
                padding=padding[2],
            ),
            deepconv_discriminator_block(
                hidden_dim * 4,
                hidden_dim * 2,
                stride=stride[3],
                kernel_size=kernel[3],
                padding=padding[3],
            ),
        )

        self.final_block = deepconv_discriminator_block(
            hidden_dim * 2,
            1,
            stride=stride[4],
            kernel_size=kernel[4],
            padding=padding[4],
            final_block=True,
        )

    def forward(self, x: torch.Tensor):
        x = self.discriminator(x)
        x = self.final_block(x)
        x = x.view(len(x), -1)
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