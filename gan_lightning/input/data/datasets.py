import zipfile
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os

class CelebaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, subset="train"):
        
        self.img_dir = img_dir
        self.transform = transform
        
        # Read attributes
        with open(annotations_file, 'r') as file:
            lines = file.readlines()
            header = lines[1].strip().split()  # Second line contains attribute names
            self.attributes = pd.read_csv(annotations_file, delim_whitespace=True, header=1, skiprows=0)
            self.attributes = self.attributes.replace(-1, 0)  # Convert -1 to 0 for binary classification
            self.image_files = list(self.attributes.index.values)

        if subset == "val":
            self.image_files = self.image_files[:2000]
            self.attributes = self.attributes.iloc[:2000]
            
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Gerekli dönüşümleri uygulayın
        if self.transform:
            image = self.transform(image)

        # Label'leri al
        labels = torch.tensor(self.attributes.iloc[idx].values, dtype=torch.float32)

        return image, labels
