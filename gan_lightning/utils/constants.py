from dataclasses import dataclass, field


@dataclass
class img_channel:
    MNIST: int = 1
    CIFAR10: int = 3
    FashionMNIST: int = 1
    CelebA: int = 3
    LSUN: int = 3


@dataclass
class num_classes:
    MNIST: int = 10
    CIFAR10: int = 10
    FashionMNIST: int = 10
    CelebA: int = 2
    LSUN: int = 2


@dataclass
class Constants:
    augment_config_dir: str = "gan_lightning/src/config/augment_config.yaml"
    img_channel: img_channel = field(default_factory=img_channel)
    num_classes: num_classes = field(default_factory=num_classes)
