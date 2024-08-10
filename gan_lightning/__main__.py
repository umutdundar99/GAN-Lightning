import os
import time
import pytorch_lightning as pl
from omegaconf import DictConfig
from gan_lightning.utils import get_loss
from gan_lightning.src.models.get_model import get_model
from gan_lightning.input.utils.get_dataloader import get_dataloader
from lightning.pytorch.loggers import MLFlowLogger
import logging

info_logger = logging.getLogger(__name__)

def GAN_Lightning(config: DictConfig):
    if config.training_params.accelerator == "gpu":
        devices = [0]

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        run_name=config.logger.run_name,
        log_model=config.logger.log_model,
        tracking_uri="http://localhost:5000",
        tags=config.logger.tags,
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
    info_logger.info(f"Using loss function {loss}")

    model = get_model(config.training_params, config.dataset, loss)
    dataloader = get_dataloader(config.dataset)
    info_logger.info(f"Using dataset {config.dataset.name}")
    info_logger.info(f"Devices: {devices}")    

    trainer = pl.Trainer(
        accelerator=config.training_params.accelerator,
        devices=devices,
        logger=logger,
        max_epochs=config.training_params.n_epochs,
        callbacks=callbacks
    )

    trainer.fit(model, dataloader)

    return model

if __name__ == "__main__":
    
    import argparse
    from hydra import compose, initialize

    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_name", type=str, choices=["conditional_gan", "controllable_gan", "deepconv_gan", "simple_gan", "wgan"], default="simple_gan")
    args = parser.parse_args()

    initialize(config_path="src/config/gan_configs", version_base=None)
    config_name = f"{args.gan_name}_config"
    cfg = compose(config_name=config_name)
    GAN_Lightning(cfg)
