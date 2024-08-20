from dataclasses import dataclass, field


@dataclass
class img_channel:
    MNIST: int = 1
    CIFAR10: int = 3
    FashionMNIST: int = 1
    CELEBA: int = 3
    LSUN: int = 3


@dataclass
class num_classes:
    MNIST: int = 10
    CIFAR10: int = 10
    FashionMNIST: int = 10
    CELEBA: int = 2
    LSUN: int = 2


@dataclass
class input_size:
    MNIST: int = 28
    CELEBA: int = 128


@dataclass
class Constants:
    AUGMENT_CONFIG_DIR: str = "gan_lightning/src/config/augment_configs"
    IMG_CHANNEL: img_channel = field(default_factory=img_channel)
    NUM_CLASSES: num_classes = field(default_factory=num_classes)
    INPUT_SIZE: input_size = field(default_factory=input_size)
