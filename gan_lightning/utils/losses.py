import torch

from gan_lightning.utils.noise import create_noise
from torch.nn import BCEWithLogitsLoss
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.criterion = BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)

    def __name__(self):
        return "FocalLoss"


class BasicGenLoss(torch.nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        input_dim: int,
        device: str,
        # loss: nn.Module = FocalLoss(alpha=0.5, gamma=2),
        loss: nn.Module = BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.input_dim = input_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self, batch_size):
        fake_noise = self.create_noise(
            batch_size, self.input_dim, device=self.device[0]
        )
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
        input_dim: int,
        device: str,
        loss: nn.Module = BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.input_dim = input_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self, x, batch_size):
        noise = self.create_noise(batch_size, self.input_dim, device=self.device[0])
        gen_out = self.generator(noise)
        assert (
            gen_out.shape == x.shape
        ), "Generator output shape does not match input shape. Please check model input size in config."
        disc_fake_pred = self.discriminator(gen_out.detach())
        disc_fake_loss = self.loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.discriminator(x)
        disc_real_loss = self.loss(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def __name__(self):
        return "BasicDiscLoss"


class WGenLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_fake_prediction):
        return -torch.mean(d_fake_prediction)

    def __name__(self):
        return "WGenLoss"


class WDiscLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        d_real_prediction,
        d_fake_prediction,
        penalty,
        penalty_lambda: float = 10.0,
    ):
        return (
            torch.mean(d_fake_prediction)
            - torch.mean(d_real_prediction)
            + penalty_lambda * penalty
        )

    def __name__(self):
        return "WDiscLoss"


class BCE(BCEWithLogitsLoss):
    def __init__(self):
        super(BCE, self).__init__()
        self.criterion = BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(input, target)

    def __name__(self):
        return "BCEWithLogitsLoss"
