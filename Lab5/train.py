import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
import os
import time

# Define a regression head
class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super(RegressionHead, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_outputs)

    def forward(self, x):
        x = self.global_avg_pooling(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

# Load pre-trained ResNet-18 model
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
in_channels = resnet18.fc.in_features

model = nn.Sequential(
    *list(resnet18.children())[:-2],  # Remove the last two layers (avgpool and fc)
    RegressionHead(in_channels, 2)  # Assuming 2 outputs for (x, y) coordinates
)

# Set your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define your dataset class and DataLoader
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
            parts = line.split(',')
            image_path = parts[0].strip()
            label_str = parts[1].replace('"', '').replace('(', '').replace(')', '').strip()
            label = [float(val) for val in label_str.split(',')]
            self.labels.append(tuple(label))  # Change to tuple
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

        if self.transform:
            image = self.transform(image)

        # Ensure labels are tuples of two values
        label_values = self.labels[idx]
        if len(label_values) == 1:
            label_values = label_values * 2  # Repeat the single value to create a tuple

        # Convert the label to a tensor and squeeze the extra dimension
        label = torch.tensor(label_values, dtype=torch.float32).squeeze()

        return image, label
# Set your hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as needed
    transforms.ToTensor(),
])

# Create datasets and dataloaders for training and testing
train_dataset = PetNoseDataset(annotation_file='train_noses3.txt', images_folder='images', transform=transform)
test_dataset = PetNoseDataset(annotation_file='test_noses.txt', images_folder='images', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("loading done")

# Set the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min')
criterion = nn.MSELoss()
model.train()
# Training loop with timer
train_losses = []
total_start_time = time.time()
for epoch in range(epochs):
    epoch_loss = 0.0
    start_time = time.time()

    for images, labels in train_dataloader:
        
        images, labels = images.to(device), labels.to(device)
        # print(images.size(), labels.size())

        optimizer.zero_grad()
        
        # Forward pass through the model
        outputs = model(images)

        # Assuming labels are (x, y) coordinates
        # Unpack the labels and reshape them
        labels = labels.view(-1, 2)  # Assuming each label is a tuple (x, y)
        
        # Check for batch size mismatch
        if outputs.size(0) != labels.size(0):
            print("Batch size mismatch. Skipping batch.")
            continue

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_dataloader)
    train_losses.append(average_epoch_loss)
    scheduler.step(average_epoch_loss)

    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')

total_end_time = time.time()
total_time_elapsed = total_end_time - total_start_time
print(f"Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")
