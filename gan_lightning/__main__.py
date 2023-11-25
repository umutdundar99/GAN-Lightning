import hydra
from omegaconf import DictConfig
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.strategies.ddp import DDPStrategy
from gan_lightning.utils.losses.get_loss import get_loss
from gan_lightning.src.models.get_model import get_model
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from torch.utils.data import DataLoader
from gan_lightning.utils.optimizers.get_optimizer import get_optimizer


@hydra.main(config_path="src/config", config_name="config", version_base=None)
def GAN_Lightning(config: DictConfig):
    if config.training_params.accelerator == "gpu":
        devices = CUDAAccelerator.parse_devices(config.training_params.device)
        if len(devices) > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = None
    else:
        strategy = None

    # TODO: Add MLFLOW Logger
    logger = None

    loss = get_loss(config.training_params.loss)
    model = get_model(config.training_params, loss)

    # TODO: Get it from a dataloader.py file
    dataloader = DataLoader(
        MNIST(".", download=True, transform=transforms.ToTensor()), batch_size=config.trainer.batch_size, shuffle=True
    )

    return model


if __name__ == "__main__":
    GAN_Lightning()
