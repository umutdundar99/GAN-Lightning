import torch
import cv2
import os
from typing import Dict, Any, List, Optional
from pytorch_lightning import LightningModule
from gan_lightning.src.models.discriminators.deepconv_discriminator import (
    DeepConv_Discriminator,
)

from gan_lightning.src.models.generators.deepconv_generator import DeepConv_Generator
from gan_lightning.utils.optimizers.get_optimizer import get_optimizer
from gan_lightning.utils.noise import create_noise
from gan_lightning.src.models import model_registration
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


@model_registration("DeepConvGAN")
class DeepConvGAN(LightningModule):
    def __init__(
        self,
        losses: Dict[str, Any],
        optimizer_dict: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.G = DeepConv_Generator(input_dim=64)
        self.G._init_weight(training_config["weight_init"])
        self.D = DeepConv_Discriminator()
        self.D._init_weight(training_config["weight_init"])
        self.optimizer_dict = optimizer_dict
        self.set_attributes(training_config)
        
        self.discriminator_loss = losses.get("discriminator_loss", None)
        self.d_loss = self.discriminator_loss(
            self.G, self.D, self.input_dim, self.device_num
        )
        self.generator_loss = losses.get("generator_loss", None)
        self.g_loss = self.generator_loss(self.G, self.D, self.input_dim, self.device_num)
        
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        return self.G(x)

    def training_step(self, batch: List[torch.Tensor]):
        gen_opt, disc_opt = self.optimizers()
        X, _ = batch
        X = X.unsqueeze(1)
        batch_size = X.shape[0]

        disc_opt.zero_grad()
        disc_loss = self.d_loss(X, batch_size)
        disc_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = self.g_loss(batch_size)
        gen_loss.backward()
        gen_opt.step()

        self.log("discriminator_loss", disc_loss, prog_bar=True)
        self.log("generator_loss", gen_loss, prog_bar=True)
        return disc_loss + gen_loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        noise = create_noise(self.batch_size, self.input_dim)
        generated_images = self(noise)

        for enum, img in enumerate(generated_images):
            image = img.cpu().detach().numpy()
            image = image.transpose(1, 2, 0)
            image = (image + 1) / 2
            epoch = str(self.current_epoch)
            path = os.path.join("generated_images", f"epoch_{epoch}")
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(
                os.path.join(path, f"generated_images_{enum}_.png"),
                image * 255,
            )

    def configure_optimizers(self):
        G_optimizer = get_optimizer(
            self.G.parameters(), self.optimizer_dict, betas=(0.5, 0.999)
        )
        D_Optimizer = get_optimizer(
            self.D.parameters(), self.optimizer_dict, betas=(0.5, 0.999)
        )
        return [G_optimizer[0], D_Optimizer[0]], [G_optimizer[1], D_Optimizer[1]]

    def set_attributes(self, config: Dict[str, Any]):
        for key, value in config.items():
            setattr(self, key, value)


