import torch
from torch import nn
import lightning.pytorch as pl
from gan_lightning.src.models.blocks.generator_block.generator_blocks import (
    simple_1d_generator_block,
)

from gan_lightning.src.models import model_registration


@model_registration("Simple_Generator")
class Simple_Generator(pl.LightningModule):
    def __init__(self, z_dim=100, img_channel=784, hidden_dim=128, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.generator_loss = kwargs.get("loss", None)
        self.generator = torch.nn.Sequential(
            simple_1d_generator_block(z_dim, hidden_dim),
            simple_1d_generator_block(hidden_dim, hidden_dim * 2),
            simple_1d_generator_block(hidden_dim * 2, hidden_dim * 4),
            simple_1d_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, img_channel),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.generator(noise)
