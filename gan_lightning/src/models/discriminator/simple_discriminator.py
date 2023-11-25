from torch import nn
from gan_lightning.src.models.blocks.discriminator_block.discriminator_blocks import (  # noqa
    simple_1d_discriminator_block,
)
from gan_lightning.utils.optimizers.get_optimizer import get_optimizer
import lightning.pytorch as pl

from gan_lightning.src.models import model_registration


@model_registration("Simple_Discriminator")
class Simple_Discriminator(pl.LightningModule):
    def __init__(self, img_channel: int = 1, hidden_dim=16, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.discriminator = nn.Sequential(
            simple_1d_discriminator_block(img_channel, hidden_dim * 4),
            simple_1d_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            simple_1d_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.discriminator(x)