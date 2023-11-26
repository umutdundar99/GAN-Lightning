from omegaconf import OmegaConf
from typing import Dict
from gan_lightning.src.models.discriminator import *
from gan_lightning.src.models.generator import *
from gan_lightning.src.models.gans import *
from gan_lightning.src.models import registered_models


def get_model(config, loss, **kwargs):
    config = OmegaConf.to_container(config)
    model_architecture = config["model"]["architecture"]
    model = registered_models[model_architecture]

    return model(
        config=extract_model_params(config),
        losses=loss,
        optimizer_dict=config["optimizer"],
        **kwargs
    )


def extract_model_params(config: Dict, output_dict: Dict = {}):
    if output_dict is None:
        output_dict = {}

    for key, value in config.items():
        if isinstance(value, dict):
            extract_model_params(value, output_dict)
        elif isinstance(key, str):
            output_dict[key] = value

    return output_dict
