from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from gan_lightning.input import register_dataset
from gan_lightning.input.utils.augmentations import GAN_Augmentation
from gan_lightning.utils.constants import Constants


@register_dataset("mnist")
class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 12,
        download: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.transform = GAN_Augmentation(Constants.augment_config_dir)

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=self.download)
        MNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
