import torch
from torch import nn
import lightning.pytorch as pl
from gan_lightning.src.models.blocks.generator_block.generator_blocks import (
    deepconv_generator_block,
)
from gan_lightning.utils.noise import create_noise
from gan_lightning.src.models import model_registration


@model_registration("Deep_Convolutional_Generator")
class DeepConv_Generator(pl.LightningModule):
    def __init__(self, z_dim=10, img_channel=1, hidden_dim=64, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.generator = torch.nn.Sequential(
            deepconv_generator_block(z_dim, hidden_dim * 4),
            deepconv_generator_block(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            ),
            deepconv_generator_block(hidden_dim * 2, hidden_dim),
        )

        self.final_block = deepconv_generator_block(
            hidden_dim, img_channel, kernel_size=4
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x = self.generator(x)
        x = self.final_block(x, True)
        return x
