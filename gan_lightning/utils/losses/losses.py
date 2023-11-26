import torch

from gan_lightning.utils.noise import create_noise
from torch.nn import BCEWithLogitsLoss as BCE
from torch import nn


class BasicGenLoss(torch.nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        z_dim: int,
        device: str,
        loss: nn.Module = BCE(),
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.z_dim = z_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self, batch_size):
        fake_noise = self.create_noise(batch_size, self.z_dim, device=self.device[0])
        fake = self.generator(fake_noise)
        disc_fake_pred = self.discriminator(fake)
        gen_loss = self.loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def __name__(self):
        return "BasicGenLoss"


class BasicDiscLoss(torch.nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        z_dim: int,
        device: str,
        loss: nn.Module = BCE(),
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.z_dim = z_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self, x, batch_size):
        noise = self.create_noise(batch_size, self.z_dim, device=self.device[0])
        gen_out = self.generator(noise)
        disc_fake_pred = self.discriminator(gen_out.detach())
        disc_fake_loss = self.loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.discriminator(x)
        disc_real_loss = self.loss(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def __name__(self):
        return "BasicDiscLoss"
