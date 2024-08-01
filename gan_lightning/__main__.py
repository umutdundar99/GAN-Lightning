import hydra
import os
import time
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
        devices = CUDAAccelerator.parse_devices(
            config.training_params.device_num
        )  # noqa
        if len(devices) > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = None
    else:
        strategy = None

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        run_name=config.logger.run_name,
        log_model=config.logger.log_model,
        tracking_uri="http://localhost:5000",
    )

    callbacks = []
    hour_day_month = time.strftime("%H-%d-%m")
    if not os.path.exists(os.path.join(os.getcwd(), "models", "checkpoints", hour_day_month)):
        os.makedirs(os.path.join(os.getcwd(), "models", "checkpoints", hour_day_month)) 
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(os.path.join(os.getcwd(), "models")),
            filename=os.path.join("checkpoints", hour_day_month, config.logger.run_name+"-{epoch:02d}-{val_loss:.2f}"),)
    
    RichProgressBar = pl.callbacks.RichProgressBar()
    
    
    
    callbacks.append(RichProgressBar)
    callbacks.append(checkpoint_callback)

    loss = get_loss(config.training_params.loss)
    model = get_model(config.training_params, config.dataset, loss)
    dataloader = get_dataloader(config.dataset)
    # TODO: Get it from a dataloader.py file

    trainer = pl.Trainer(
        accelerator=config.training_params.accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        max_epochs=config.training_params.n_epochs,
        callbacks=callbacks
    )

    trainer.fit(model, dataloader)

    return model


if __name__ == "__main__":
    GAN_Lightning()
