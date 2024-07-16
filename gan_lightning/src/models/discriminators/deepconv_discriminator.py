import lightning.pytorch as pl
import torch
from torch import nn
from gan_lightning.src.models.blocks.discriminator_blocks import (  # noqa
    deepconv_discriminator_block,
)

from gan_lightning.src.models import model_registration


@model_registration("Deep_Convolutional_Discriminator")
class DeepConv_Discriminator(pl.LightningModule):
    def __init__(self, img_channel: int = 1, hidden_dim: int = 16, **kwargs):
        super().__init__()
        self.discriminator = nn.Sequential(
            deepconv_discriminator_block(img_channel, hidden_dim, padding=2, kernel_size=3),
            deepconv_discriminator_block(hidden_dim, hidden_dim * 2, padding=2),
            deepconv_discriminator_block(hidden_dim * 2, hidden_dim * 4, padding=2),
            deepconv_discriminator_block(hidden_dim * 4, hidden_dim * 2, padding=2),
        )

        self.final_block = deepconv_discriminator_block(hidden_dim * 2, 1, final_block=True)

    def forward(self, x: torch.Tensor):
        x = self.discriminator(x)
        x = self.final_block(x)
        x = x.view(len(x), -1)
        return x

    
    def _init_weight(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
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