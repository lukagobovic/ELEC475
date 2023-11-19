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
            # Load the pre-trained ResNet-18 model
            resnet18 = models.resnet18(weights=weights)

            # Remove the last fully connected layer
            self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

            # Add a new fully connected layer with the desired number of output classes
            in_features = resnet18.fc.in_features
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            # Forward pass through the modified ResNet-18
            x = self.resnet18(x)
            
            # Global average pooling (GAP) to reduce spatial dimensions
            x = F.adaptive_avg_pool2d(x, (1, 1))

            # Reshape the tensor for the fully connected layer
            x = x.view(x.size(0), -1)

            # Forward pass through the fully connected layer
            x = self.fc(x)

            return x
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0  # Track the total loss for the epoch

        for batchnum, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        average_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(average_epoch_loss)
        end_time = time.time()
        scheduler.step()
        print(f'Time for epoch {epoch+1}: {end_time - start_time:.2f} seconds')

        # Plotting the training losses after each epoch
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label=f'Epoch {epoch+1} Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.savefig(f'yoda_classifier_train_epoch_{epoch+1}')
        plt.show()

    return model, train_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            #print(count)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            val_loss += criterion(outputs, labels).item()
            count += 1

    accuracy = total_correct / total_samples
    avg_val_loss = val_loss / len(dataloader)

    model.train()
    return accuracy, avg_val_loss

if __name__ == '__main__':
    freeze_support() 
    print('loading')
    dataset_root = 'data/Kitti8_ROIs'

    train_dataset = CustomDataset(root_dir=dataset_root, mode='train', transform=transforms.ToTensor(), target_size=(150, 150))
    test_dataset = CustomDataset(
            root_dir=dataset_root,
            mode='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]),
            target_size=(150, 150)
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=96, shuffle=False, num_workers=3)

    num_classes = 2
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    num_epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Train the model
    # model, train_losses = train(model, train_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Save the trained model
    # torch.save(model.state_dict(), 'yoda_classifier_test1.pth')

    # Evaluate the model
    model.load_state_dict(torch.load('yoda_classifier_with_early_stopping_part2.pth'))
    accuracy, avg_val_loss = evaluate(model, test_loader, criterion, device)

    print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {avg_val_loss:.4f}')
