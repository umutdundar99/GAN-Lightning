from omegaconf import OmegaConf
from typing import Dict
from gan_lightning.src.models.discriminators import *
from gan_lightning.src.models.generators import *
from gan_lightning.src.models.gans import *
from gan_lightning.src.models.classifiers import *
from gan_lightning.src.models import registered_models


def get_model(training_config, dataset_config, loss, **kwargs):
    training_config = OmegaConf.to_container(training_config)
    dataset_config = OmegaConf.to_container(dataset_config)
    model_architecture = training_config["model"]["architecture"]
    model = registered_models[model_architecture]

    return model(
        training_config=extract_model_params(training_config),
        dataset_config=dataset_config,
        losses=loss,
        optimizer_dict=training_config["optimizer"],
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
