from omegaconf import OmegaConf
from gan_lightning.input.data import *
from gan_lightning.input import registered_dataloaders
import inspect

def get_dataloader(config, **kwargs):
    config = OmegaConf.to_container(config)
    dataloader = registered_dataloaders[config["name"].lower()]
    config = adjust_config(dataloader, config)
    dataloader = dataloader(**config, **kwargs)
    return dataloader
    
    
def adjust_config(loader, config: dict):
    constructor_info = inspect.getfullargspec(loader.__init__)
    config_iter = config.copy()
    args = constructor_info.args.copy()
    args.remove("self")
    for config_key, _ in config_iter.items():
        if not config_key in args:
            # remove config_key from config dict
            config.pop(config_key)
    return config