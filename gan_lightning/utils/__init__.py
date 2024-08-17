from gan_lightning.utils.config import *  # noqa F401
import logging
from omegaconf import OmegaConf
from torch.optim import (
    Adam,
    RMSprop,
    Rprop,
    SGD,
    AdamW
)
from torch.optim.lr_scheduler import StepLR

from gan_lightning.utils.losses import (
    BasicGenLoss,
    BasicDiscLoss,
    WDiscLoss,
    WGenLoss,
    FocalLoss,
    BCE,
)

import omegaconf
from omegaconf import OmegaConf


__optimizers__ = [
    Adam,
    RMSprop,
    Rprop,
    SGD,
    AdamW,
]


__schedulers__ = [StepLR]


all_losses = [BCE, BasicGenLoss, BasicDiscLoss, WDiscLoss, WGenLoss, FocalLoss]


def get_optimizer(model_params, optimizer_dict: str, **kwargs):
    scheduler_dict = optimizer_dict.get("scheduler", None)
    if optimizer_dict["optimizer_name"].lower() not in [
        optimizer.__name__.lower() for optimizer in __optimizers__
    ]:
        name = optimizer_dict["optimizer_name"]
        raise ValueError(f"Optimizer {name} not found")

    for optimizer in __optimizers__:
        if optimizer.__name__.lower() == optimizer_dict["optimizer_name"].lower():
            optimizer = optimizer
            break
    logging.info(
        "Using optimizer: {}".format(optimizer_dict["optimizer_name"].capitalize())
    )

    if scheduler_dict is None:
        logging.info("No scheduler found")
        return optimizer(lr=optimizer_dict["lr"], **kwargs)
    else:
        
        for scheduler in __schedulers__:
            if scheduler.__name__.lower() == scheduler_dict["scheduler_name"].lower():
                scheduler = scheduler
                break
        logging.info(f"Using scheduler: {scheduler_dict['scheduler_name']}")
        optimizer = optimizer(params=model_params, lr=optimizer_dict["lr"], **kwargs)

        return optimizer, scheduler(
            optimizer,
            step_size=scheduler_dict["step_size"],
            gamma=scheduler_dict["gamma"],
        )


def get_loss(loss_config: omegaconf.dictconfig.DictConfig):
    loss_dict = OmegaConf.to_container(loss_config)
    losses = {}
    for key, value in loss_dict.items():
        if value not in [loss.__name__ for loss in all_losses] and not value == "None":
            raise ValueError(f"Loss {value} not found")
        for loss in all_losses:
            if loss.__name__ == value:
                losses[key] = loss

    return losses
