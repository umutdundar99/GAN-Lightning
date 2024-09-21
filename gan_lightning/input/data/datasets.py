import torch
import os
import cv2

import pandas as pd

from torch.utils.data import Dataset


class CelebaDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, subset=None, input_size=128
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.input_size = input_size
        with open(annotations_file, "r") as file:
            self.attributes = pd.read_csv(
                annotations_file, delim_whitespace=True, header=1, skiprows=0
            )
            self.attributes = self.attributes.replace(-1, 0)
            self.image_files = list(self.attributes.index.values)

        if subset == "train":
            self.image_files = self.image_files[2000:]
            self.attributes = self.attributes.iloc[2000:]
        if subset == "val":
            self.image_files = self.image_files[:2000]
            self.attributes = self.attributes.iloc[:2000]
        print(f"Lenght of {subset} dataset: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.input_size)

        if self.transform:
            image = self.transform(image)

        image = image.transpose((2, 0, 1))

        labels = torch.tensor(self.attributes.iloc[idx].values, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32)

        return image, labels
