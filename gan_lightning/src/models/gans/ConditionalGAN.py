import torch
import cv2
import os
from typing import Dict, Any, List, Optional
from pytorch_lightning import LightningModule
from gan_lightning.src.models.discriminators.deepconv_discriminator import (
    DeepConv_Discriminator,
)

from gan_lightning.src.models.generators.deepconv_generator import DeepConv_Generator
from gan_lightning.utils import get_optimizer
from gan_lightning.utils.noise import create_noise
from gan_lightning.src.models import model_registration
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import functional as F
from gan_lightning.utils.constants import Constants


@model_registration("ConditionalGAN")
class ConditionalGAN(LightningModule):
    def __init__(
        self,
        losses: Dict[str, Any],
        optimizer_dict: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        img_channel = getattr(Constants.img_channel, dataset_config["name"])
        self.num_classes = getattr(Constants.num_classes, dataset_config["name"])
        self.input_dim = training_config["input_dim"]
        self.input_dim, self.img_channel = self.set_input_dim(
            self.input_dim, img_channel, self.num_classes
        )
        self.G = DeepConv_Generator(input_dim=self.input_dim)
        self.G.weight_init(training_config["weight_init_name"])
        self.D = DeepConv_Discriminator(
            img_channel=self.img_channel, hidden_dim=training_config["input_dim"]
        )
        self.D.weight_init(training_config["weight_init_name"])
        self.optimizer_dict = optimizer_dict
        self.set_attributes(training_config)
        self.discriminator_loss = losses.get("discriminator_loss", None)
        self.d_loss = self.discriminator_loss()
        self.generator_loss = losses.get("generator_loss", None)
        self.g_loss = self.generator_loss()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        return self.G(x)

    def training_step(self, batch: List[torch.Tensor]):
        gen_opt, disc_opt = self.optimizers()
        X, y = batch
        X = X.unsqueeze(1)
        batch_size = X.shape[0]
        one_hot_labels = F.one_hot(y, num_classes=self.num_classes)
        y = one_hot_labels[:, :, None, None]
        y = y.repeat(1, 1, X.shape[2], X.shape[2])

        disc_opt.zero_grad()
        fake_noise = create_noise(batch_size, self.input_dim)

        # The idea: concatenate the noise with the one-hot labels to create a new input for the generator
        # Generator will then generate images based on the noise and the one-hot labels
        # So that, one_hot-labels can be used to control the generator as a Z vector
        combined_noise = torch.cat((fake_noise, one_hot_labels), dim=1)
        gen_fake_out = self.G(combined_noise)

        fake_image_and_y = torch.cat((gen_fake_out, y), dim=1)
        disc_fake_out = self.D(fake_image_and_y.detach())

        real_image_and_y = torch.cat((X, y), dim=1)
        disc_real_out = self.D(real_image_and_y)

        disc_fake_loss = self.d_loss(disc_fake_out, torch.zeros_like(disc_fake_out))
        disc_real_loss = self.d_loss(disc_real_out, torch.ones_like(disc_real_out))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        fake_image_and_y = torch.cat((gen_fake_out, y), dim=1)
        disc_fake_out_2 = self.D(fake_image_and_y)
        gen_loss = self.g_loss(disc_fake_out_2, torch.ones_like(disc_fake_out_2))
        gen_loss.backward()
        gen_opt.step()

        self.log("discriminator_loss", disc_loss, prog_bar=True)
        self.log("generator_loss", gen_loss, prog_bar=True)
        return disc_loss + gen_loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        noise = create_noise(40, self.input_dim)
        y = 8
        one_hot_labels = F.one_hot(torch.tensor([y] * 40), num_classes=10).to(
            self.device
        )
        noise = torch.cat((noise, one_hot_labels), dim=1)
        generated_images = self(noise)
        if self.current_epoch % 6 == 0:
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

    def set_input_dim(self, input_dim: int, input_shape: List[int], num_classes: int):
        generator_input_dim = input_dim + num_classes
        discriminator_im_chan = input_shape + num_classes

        return generator_input_dim, discriminator_im_chan
