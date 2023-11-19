import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, target_size=(150, 150)):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.image_folder = os.path.join(root_dir, mode)
        self.label_file = os.path.join(root_dir, mode, 'labels.txt').replace('\\', '/')

        # Read the labels file
        self.labels = self._read_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.normpath(os.path.join(self.image_folder, self.labels[idx][0]))
        image = cv2.imread(img_name)

        label = float(self.labels[idx][1])  # Convert 'Car' to 1.0, 'NoCar' to 0.0

        image = cv2.resize(image, self.target_size)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        # Ensure label is a tensor with the correct shape
        label = int(self.labels[idx][1])  # Convert 'Car' to 1, 'NoCar' to 0

        return image, label

    def _read_labels(self):
        labels = []
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                labels.append((parts[0], int(parts[1]), parts[2]))
        return labels
