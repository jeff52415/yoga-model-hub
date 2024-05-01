import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, labels, transform=None, return_label=True):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        # Read the image using PIL
        image = Image.open(file_path)

        # Convert the image to RGB format (if it's not already)
        image = image.convert("RGB")

        # Convert the PIL image to a numpy array
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        if self.return_label:
            return image, label
        else:
            return image


def extract_data(file_path, root="Yoga82/dataset"):
    imgs = []
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            # Split the line on the comma
            parts = line.strip().split(",")
            # Get the image path and labels
            image_path = parts[0]
            image_path = os.path.join(root, image_path)
            label = ",".join(parts[1:])
            if os.path.exists(image_path):
                imgs.append(image_path)
                labels.append([int(word) for word in label.split(",")])
    return imgs, labels
