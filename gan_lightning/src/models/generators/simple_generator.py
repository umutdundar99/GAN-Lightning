from torch import nn
import lightning.pytorch as pl
from gan_lightning.src.models.blocks.generator_blocks import (
    simple_1d_generator_block,
)

from gan_lightning.src.models import model_registration

@model_registration("Simple_Generator")
class Simple_Generator(pl.LightningModule):
    def __init__(self, input_dim=10, img_channel=784, hidden_dim=128, **kwargs):
        super().__init__()
        self.generator = nn.Sequential(
            simple_1d_generator_block(input_dim, hidden_dim),
            simple_1d_generator_block(hidden_dim, hidden_dim * 2),
            simple_1d_generator_block(hidden_dim * 2, hidden_dim * 4),
            simple_1d_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, img_channel),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.generator(noise)

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
