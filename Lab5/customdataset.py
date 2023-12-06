import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class PetNoseDataset(Dataset):
    def __init__(self, annotation_file, images_folder, transform=None):
        self.annotation_file = annotation_file
        self.images_folder = images_folder
        self.transform = transform

        # Read annotation file
        with open(annotation_file, 'r') as file:
            lines = file.readlines()

        # Extract image paths and labels
        self.image_paths = []
        self.labels = []

        for line in lines:
            # Split the line by commas
            parts = line.split(',')
            image_path = parts[0].strip()
            x_str = parts[1].replace('"(', '').replace(')"', '')
            y_str = parts[2].strip().replace(')"', '')
            label = (int(x_str),int(y_str))
            self.labels.append(label)
            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_paths[idx])

        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB') 
        old_width, old_height = image.size

        if self.transform:
            image = self.transform(image)

        # Ensure labels are tuples of two values
        label = self.labels[idx]

        # Resize coordinates proportionally
         # Corrected line
        new_width, new_height = 224, 224  # Target size

        width_scale = new_width / old_width
        height_scale = new_height / old_height

        label = (
            int(label[0] * width_scale),
            int(label[1] * height_scale)
        )

        label = torch.tensor(label, dtype=torch.float32)

        return image, label