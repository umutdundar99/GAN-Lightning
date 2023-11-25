import logging
from omegaconf import OmegaConf
from torch.optim import (
    Adam,
    RMSprop,
    Rprop,
    SGD,
)
from torch.optim.lr_scheduler import StepLR


__optimizers__ = [
    Adam,
    RMSprop,
    Rprop,
    SGD,
]


__schedulers__ = [StepLR]


def get_optimizer(model_params, optimizer_config: str, **kwargs):
    optimizer_dict = OmegaConf.to_container(optimizer_config)
    scheduler_dict = optimizer_dict.get("scheduler", None)
    if optimizer_config.name not in [optimizer.__name__.lower() for optimizer in __optimizers__]:
        raise ValueError(f"Optimizer {optimizer_config.name} not found")

    for optimizer in __optimizers__:
        if optimizer.__name__.lower() == optimizer_config.name.lower():
            optimizer = optimizer
            break
    logging.info("Using optimizer: {}".format(optimizer_config.name))

    if scheduler_dict is None:
        logging.info("No scheduler found")
        return optimizer(lr=optimizer_dict["lr"], **kwargs)
    else:
        if scheduler_dict["name"] not in [scheduler.__name__ for scheduler in __schedulers__]:
            raise ValueError("Scheduler {} not found".format(scheduler_dict["name"]))
        for scheduler in __schedulers__:
            if scheduler.__name__.lower() == scheduler_dict["name"].lower():
                scheduler = scheduler
                break
        logging.info(f"Using scheduler: {scheduler_dict['name']}")
        return scheduler(optimizer(model_params, lr=optimizer_dict["lr"], **kwargs), **scheduler_dict["params"])
