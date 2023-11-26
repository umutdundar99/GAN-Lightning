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


def get_optimizer(model_params, optimizer_dict: str, **kwargs):
    scheduler_dict = optimizer_dict.get("scheduler", None)
    if optimizer_dict["optimizer_name"] not in [
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
        if scheduler_dict["scheduler_name"] not in [
            scheduler.__name__ for scheduler in __schedulers__
        ]:
            raise ValueError(
                "Scheduler {} not found".format(scheduler_dict["scheduler_name"])
            )
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
