import torch.nn as nn
import lightning.pytorch as pl
from gan_lightning.src.models.blocks.generator_blocks import (
    deepconv_generator_block,
)
from gan_lightning.src.models import model_registration


@model_registration("Controllable_Generator")
class Controllable_Generator(pl.LightningModule):
    def __init__(
        self, input_dim: int = 10, img_channel: int = 3, hidden_dim: int = 64, **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        
        self.final_block = deepconv_generator_block(hidden_dim, img_channel, kernel_size=4, final_block=True)
        
        self.generator = nn.Sequential(
                deepconv_generator_block(input_dim, hidden_dim * 8, stride=1),
                deepconv_generator_block(hidden_dim * 8, hidden_dim*4),
                deepconv_generator_block(hidden_dim * 4, hidden_dim*2),
                deepconv_generator_block(hidden_dim * 2, hidden_dim)
            )
        
    # O=(I−1)×stride−2×padding+kernel_size
    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        x = self.generator(x)
        x = self.final_block(x, True)
        return x
    
    def weight_init(self, mode):
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
