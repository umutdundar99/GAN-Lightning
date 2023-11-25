from omegaconf import OmegaConf
from gan_lightning.src.models.discriminator import *
from gan_lightning.src.models.generator import *
from gan_lightning.src.models.gans import *
from gan_lightning.src.models import registered_models


def get_model(config, loss, **kwargs):
    config = OmegaConf.to_container(config)
    model_architecture = config["model"]["architecture"]
    model = registered_models[model_architecture]

    return model(config=config, loss=loss, **kwargs)
