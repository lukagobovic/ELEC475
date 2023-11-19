import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet18_Weights
import time  # Import the time module
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
from customdataset import CustomDataset  # Import the CustomDataset from the module

class YodaClassifier(nn.Module):
    def __init__(self, num_classes, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(YodaClassifier, self).__init__()
        resnet18 = models.resnet18(weights=weights)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        in_features = resnet18.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = self.resnet18(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x



def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, patience=6):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    early_stop_counter = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(average_epoch_loss)
        scheduler.step(average_epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            val_loss = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                val_loss += criterion(outputs, labels).item()


            accuracy = total_correct / total_samples
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
    
            end_time = time.time()
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_epoch_loss:.4f}, '
                f'Accuracy: {accuracy:.4f},'
                f'Validation Loss: {avg_val_loss:.4f}, Time: {end_time - start_time:.2f} seconds')

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs.')
                total_end_time = time.time()
                break

    total_end_time = time.time()
    print(f'Time: {((total_end_time - total_start_time) / 60.0):.2f} minutes')
    return train_losses, val_losses



if __name__ == '__main__':
    freeze_support()
    print('loading')
    # Set the root directory of your dataset
    dataset_root = 'data/Kitti8_ROIs'

    train_dataset = CustomDataset(
        root_dir=dataset_root,
        mode='train',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),  # Smaller rotation range
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        target_size=(150, 150)
    )

    test_dataset = CustomDataset(
        root_dir=dataset_root,
        mode='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        target_size=(150, 150)
    )


    # Use DataLoader for efficient batching with num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=3)

    print("done loading")

    # Initialize the model
    num_classes = 2  # Assuming binary classification (e.g., Car or NoCar)
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1)

    criterion = nn.CrossEntropyLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    num_epochs = 40  # or as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Training
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, patience=6)

    # Save the trained model
    torch.save(model.state_dict(), 'yoda_classifier_with_early_stopping_LR00001B192.pth')

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('yoda_classifier_with_early_stopping_LR00001B192')
    plt.show
