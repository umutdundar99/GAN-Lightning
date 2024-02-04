import albumentations as A
from omegaconf import OmegaConf
from typing import Optional
import numpy as np


class GAN_Augmentation:
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir
        self.augmentations = []
        self.compose_list = []
        self.decompose_config()

    def decompose_config(self):
        config = OmegaConf.load(self.config_dir)
        augmentation_names = [aug for aug in config.keys()]
        assert len(augmentation_names) > 0, "No Augmentations Found"
        self.augmentations = [getattr(A, aug) for aug in augmentation_names]
        for aug in self.augmentations:
            aug_params = config[aug.__name__]
            aug_params = {k: v for k, v in aug_params.items() if v is not None}
            self.compose_list.append(aug(**aug_params))

    def __call__(self, image):
        return A.Compose(self.compose_list)(image=np.array(image))["image"]

    def __repr__(self):
        return f"GAN_Augmentation(config_dir={self.config_dir})"

    def __str__(self):
        return f"GAN_Augmentation(config_dir={self.config_dir})"

    def __len__(self):
        return len(self.compose_list)

    def __getitem__(self, idx):
        return self.compose_list[idx]
