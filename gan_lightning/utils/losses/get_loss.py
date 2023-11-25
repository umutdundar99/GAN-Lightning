from torch.nn import BCEWithLogitsLoss as BCE
from gan_lightning.utils.losses.losses import BasicGenLoss
from gan_lightning.utils.losses.losses import BasicDiscLoss
import omegaconf
from omegaconf import OmegaConf

all_losses = [BCE, BasicGenLoss, BasicDiscLoss]


def get_loss(loss_config: omegaconf.dictconfig.DictConfig):
    loss_dict = OmegaConf.to_container(loss_config)
    losses = {}
    for key, value in loss_dict.items():
        if value not in [loss.__name__ for loss in all_losses]:
            raise ValueError(f"Loss {value} not found")
        for loss in all_losses:
            if loss.__name__ == value:
                losses[key] = loss

    return losses
