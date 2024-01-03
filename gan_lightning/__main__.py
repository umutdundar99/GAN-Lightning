import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.strategies.ddp import DDPStrategy
from gan_lightning.utils.losses.get_loss import get_loss
from gan_lightning.src.models.get_model import get_model
from gan_lightning.input.utils.get_dataloader import get_dataloader
from lightning.pytorch.loggers import MLFlowLogger


@hydra.main(config_path="src/config", config_name="config", version_base=None)
def GAN_Lightning(config: DictConfig):
    if config.training_params.accelerator == "gpu":
        devices = CUDAAccelerator.parse_devices(config.training_params.device_num)
        if len(devices) > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = None
    else:
        strategy = None

    # TODO: Add MLFLOW Logger
    logger = MLFlowLogger(experiment_name=config.logger.experiment_name, 
                          run_name = config.logger.run_name, 
                          log_model= config.logger.log_model,
                          tracking_uri="http://localhost:5000")
    
    loss = get_loss(config.training_params.loss)
    model = get_model(config.training_params, loss)
    dataloader = get_dataloader(config.dataset)
    # TODO: Get it from a dataloader.py file


    trainer = pl.Trainer(
        accelerator=config.training_params.accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        max_epochs=config.training_params.n_epochs,
    )

    trainer.fit(model, dataloader)

    return model


if __name__ == "__main__":
    GAN_Lightning()
