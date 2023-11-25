from pytorch_lightning import LightningModule
from gan_lightning.src.models.discriminator.simple_discriminator import Simple_Discriminator
from gan_lightning.src.models.generator.simple_generator import Simple_Generator
import torch

from gan_lightning.utils.optimizers.get_optimizer import get_optimizer

from gan_lightning.src.models import model_registration


@model_registration("SimpleGAN")
class SimpleGAN(LightningModule):
    def __init__(self, config: None, loss: torch.nn.Module = None, optimizer: str = "Adam", **kwargs):
        super().__init__()
        self.G = Simple_Generator()
        self.D = Simple_Discriminator()
        self.config = config
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            real, _ = batch
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1)
            fake_noise = torch.randn(cur_batch_size, self.z_dim, device=self.device)
            fake = self.gen(fake_noise)

        def configure_optimizers(self):
            G_optimizer = get_optimizer(self.G, self.kwargs["optimizer"])
            D_Optimizer = get_optimizer(self.D, self.kwargs["optimizer"])
            return optimizer
