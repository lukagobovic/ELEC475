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
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

# Function to display image with annotations
# Function to display image with annotations
def display_image_with_annotations(image, true_labels, predicted_labels, title, distance):
    # Convert the PyTorch tensor to a numpy array
    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Convert the image to BGR format (OpenCV uses BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    true_nose = (int(true_labels[0]), int(true_labels[1]))
    predicted_nose = (int(predicted_labels[0]), int(predicted_labels[1]))

    # Draw circles on the image for true and predicted nose positions
    cv2.circle(image_bgr, true_nose, 8, (0, 255, 0), 1)
    cv2.circle(image_bgr, predicted_nose, 8, (255, 0, 0), 1)

    # Display the image
    cv2.imshow(title, image_bgr)
    print(f"True Nose: {true_nose}, Predicted Nose: {predicted_nose}, Distance: {distance}")
    key = cv2.waitKey(0)

    # Check if the window exists before trying to destroy it
    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(title)

    if key == ord('q'):
        exit(0)
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
# Set your hyperparameters
batch_size = 32
learning_rate = 0.0001
epochs = 30

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4789, 0.4476, 0.3948], std=[0.2321, 0.2291, 0.2318]),
])

# Data transformations
transform_test = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4789, 0.4476, 0.3948], std=[0.2321, 0.2291, 0.2318]),
])

def euclidean_distance(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))

# Create datasets and dataloaders for training and testing
train_dataset = PetNoseDataset(annotation_file='train_noses3.txt', images_folder='images', transform=transform_test)
test_dataset = PetNoseDataset(annotation_file='test_noses.txt', images_folder='images', transform=transform_test)
print("Train Dataset Length:", len(train_dataset))
print("Test Dataset Length:", len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("loading done")

# Set the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min')
criterion = nn.MSELoss()


train_losses = []
val_losses = []
val_accuracies = []
total_start_time = time.time()
for epoch in range(epochs):
    # Training
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.view(-1, 2)
        # labels = labels.long()  # Ensure labels are of type long for CrossEntropyLoss


        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_dataloader)
    train_losses.append(average_epoch_loss)
    scheduler.step(average_epoch_loss)

    # Validation
    val_loss = 0.0
    all_val_distances = []

   # Evaluate the model on the test dataset
    # Inside your validation loop
    # Inside your validation loop
    model.eval()
    all_distances = []
    worst_distance_idx = None
    best_distance_idx = None
    worst_distance = float('-inf')
    best_distance = float('inf')

    with torch.no_grad():
        for batch_idx, (val_images, val_labels) in enumerate(test_dataloader):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_labels = val_labels.view(-1, 2)

            # Calculate Euclidean distance
            val_distances = euclidean_distance(val_outputs, val_labels)
            all_distances.extend(val_distances.cpu().numpy())

            # Update indices for worst and best distances
            if torch.max(val_distances) > worst_distance:
                worst_distance = torch.max(val_distances)
                worst_distance_idx = batch_idx
            if torch.min(val_distances) < best_distance:
                best_distance = torch.min(val_distances)
                best_distance_idx = batch_idx

    # Print or visualize the images with the worst and best distances
    worst_images, worst_labels = test_dataset[worst_distance_idx]
    best_images, best_labels = test_dataset[best_distance_idx]

    # Get model predictions
    worst_predictions = model(worst_images.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
    best_predictions = model(best_images.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()

    # Print the indices and distances
    print(f"Worst Distance Index: {worst_distance_idx}, Worst Distance: {worst_distance}")
    print(f"Best Distance Index: {best_distance_idx}, Best Distance: {best_distance}")

    # Display the images with annotations
    display_image_with_annotations(worst_images, worst_labels, worst_predictions, "Worst Distance", worst_distance)
    display_image_with_annotations(best_images, best_labels, best_predictions, "Best Distance", best_distance)


    # Function to display image with annotations
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Training Epoch {epoch + 1}/{epochs},Train Loss: {average_epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')
    # print(f"Minimum Distance: {min_distance}")
    # print(f"Mean Distance: {mean_distance}")
    # print(f"Maximum Distance: {max_distance}")
    # print(f"Standard Deviation: {std_distance}")


total_end_time = time.time()
total_time_elapsed = total_end_time - total_start_time
print(f"Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot2.png')

# Save the trained model
torch.save(model.state_dict(), 'trained_model2.pth')
