import lightning.pytorch as pl
import torch
from torch import nn
from gan_lightning.src.models.blocks.discriminator_block.discriminator_blocks import (  # noqa
    deepconv_discriminator_block,
)

from gan_lightning.src.models import model_registration


@model_registration("Deep_Convolutional_Discriminator")
class DeepConv_Discriminator(pl.LightningModule):
    def __init__(self, img_channel: int = 1, hidden_dim: int = 16, **kwargs):
        super().__init__()
        self.discriminator = nn.Sequential(
            deepconv_discriminator_block(img_channel, hidden_dim),
            deepconv_discriminator_block(hidden_dim, hidden_dim * 2),
        )

        self.final_block = deepconv_discriminator_block(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor):
        x = self.discriminator(x)
        x = self.final_block(x, True)
        x = x.view(len(x), -1)
        return x