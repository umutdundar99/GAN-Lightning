import torch
from typing import Dict, Any
from pytorch_lightning import LightningModule
from gan_lightning.src.models.discriminator.simple_discriminator import Simple_Discriminator
from gan_lightning.src.models.generator.simple_generator import Simple_Generator
from gan_lightning.utils.optimizers.get_optimizer import get_optimizer
from gan_lightning.src.models import model_registration


@model_registration("SimpleGAN")
class SimpleGAN(LightningModule):
    def __init__(self, config: None, losses, optimizer_dict: Dict, **kwargs):
        super().__init__()
        self.G = Simple_Generator()
        self.D = Simple_Discriminator()
        self.optimizer_dict = optimizer_dict

        self.set_attributes(config)
        self.discriminator_loss = losses.get("discriminator_loss", None)
        self.d_loss = self.discriminator_loss(self.G, self.D, self.z_dim, self.device_num)
        self.generator_loss = losses.get("generator_loss", None)
        self.g_loss = self.generator_loss(self.G, self.D, self.z_dim, self.device_num)
        self.automatic_optimization = False

    def training_step(self, batch):
        gen_opt, disc_opt = self.optimizers()
        X, _ = batch
        batch_size = X.shape[0]
        X = X.view(batch_size, -1)

        disc_opt.zero_grad()
        disc_loss = self.d_loss(X, batch_size)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = self.g_loss(batch_size)
        gen_loss.backward()
        gen_opt.step()

        return disc_loss + gen_loss

    def configure_optimizers(self):
        G_optimizer = get_optimizer(self.G.parameters(), self.optimizer_dict)
        D_Optimizer = get_optimizer(self.D.parameters(), self.optimizer_dict)
        return [G_optimizer[0], D_Optimizer[0]], [G_optimizer[1], D_Optimizer[1]]

    def set_attributes(self, config: Dict[str, Any]):
        for key, value in config.items():
            setattr(self, key, value)
