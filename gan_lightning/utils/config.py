from typing import Any, Dict
from omegaconf import OmegaConf


def read_config(config_path: str) -> Dict[str, Any]:
    """Reads the config file and returns a dictionary.

    Args:
        config_path (str): Path to the config file.

    Returns:
        Dict[str, Any]: Dictionary containing the config file.
    """
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)
    return config_dict
