import os
import time
import pytorch_lightning as pl
from omegaconf import DictConfig
from gan_lightning.utils import get_loss
from gan_lightning.src.models.functions import get_model
from gan_lightning.input.utils.get_dataloader import get_dataloader
from lightning.pytorch.loggers import MLFlowLogger
from gan_lightning.input.utils.augmentations import GAN_Augmentation
from gan_lightning.utils.constants import Constants
import logging

info_logger = logging.getLogger(__name__)


def GAN_Lightning(config: DictConfig):
    if config.training_params.accelerator == "gpu":
        devices = [0]

    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        run_name=config.logger.run_name,
        tracking_uri="mlruns",
    )

    transforms = GAN_Augmentation(
        os.path.join(
            Constants.AUGMENT_CONFIG_DIR,
            f"{config.training_params.model.architecture}.yaml",
        )
    )

    callbacks = []
    hour_day_month = time.strftime("%H-%d-%m")
    if not os.path.exists(
        os.path.join(os.getcwd(), "models", "checkpoints", hour_day_month)
    ):
        os.makedirs(os.path.join(os.getcwd(), "models", "checkpoints", hour_day_month))

    monitor = (
        f"{config.training_params.monitor}_loss"
        if config.training_params.monitor != "None"
        else None
    )
    info_logger.info(f"Using monitor {monitor}")

    # ModelCheckpoint parametrelerini ayarlarken
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "models", "checkpoints", hour_day_month),
        filename=f"{config.training_params.model.architecture}-{hour_day_month}-"
        + "{epoch}-{val_loss:.2f}-best",
        mode="min",
    )

    if monitor is not None:
        checkpoint_callback.monitor = monitor
        checkpoint_callback.save_top_k = 2

    RichProgressBar = pl.callbacks.RichProgressBar()
    callbacks.append(RichProgressBar)
    callbacks.append(checkpoint_callback)

    loss = get_loss(config.training_params.loss)
    info_logger.info(f"Using loss function {loss}")

    model = get_model(config.training_params, config.dataset, loss)
    dataloader = get_dataloader(config.dataset, transforms=transforms)
    info_logger.info(f"Using dataset {config.dataset.name}")
    info_logger.info(f"Devices: {devices}")

    trainer = pl.Trainer(
        accelerator=config.training_params.accelerator,
        devices=devices,
        logger=logger,
        max_epochs=config.training_params.n_epochs,
        callbacks=callbacks,
    )

    trainer.fit(model, dataloader)

    return model
