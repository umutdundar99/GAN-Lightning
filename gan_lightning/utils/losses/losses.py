import torch

from gan_lightning.utils.noise import create_noise


class BasicGenLoss(torch.nn.Module):
    def __init__(self, generator, discriminator, loss, num_images, z_dim, device):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.num_images = num_images
        self.z_dim = z_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self):
        fake_noise = self.create_noise(self.num_images, self.z_dim, device=self.device)
        fake = self.generator(fake_noise)
        disc_fake_pred = self.discriminator(fake)
        gen_loss = self.loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def __name__(self):
        return "BasicGenLoss"


class BasicDiscLoss(torch.nn.Module):
    def __init__(self, generator, discriminator, loss, real_img, num_images, z_dim, device):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.num_images = num_images
        self.z_dim = z_dim
        self.device = device
        self.create_noise = create_noise

    def forward(self, x):
        noise = self.create_noise(self.num_images, self.z_dim, device=self.device)
        gen_out = self.generator(noise)
        disc_fake_pred = self.discriminator(gen_out.detach())
        disc_fake_loss = self.loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.discriminator(x)
        disc_real_loss = self.loss(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

    def __name__(self):
        return "BasicDiscLoss"
