from torch import nn
from gan_lightning.src.models.blocks.discriminator_blocks import (  # noqa
    simple_1d_discriminator_block,
)
import lightning.pytorch as pl

from gan_lightning.src.models import model_registration


@model_registration("Simple_Discriminator")
class Simple_Discriminator(pl.LightningModule):
    def __init__(self, img_channel: int = 784, hidden_dim=128, **kwargs):
        super().__init__()
        self.discriminator = nn.Sequential(
            simple_1d_discriminator_block(img_channel, hidden_dim * 4),
            simple_1d_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            simple_1d_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.discriminator(x)

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
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise ValueError("Invalid weight initialization mode")

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
