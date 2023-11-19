import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from customdataset import CustomDataset  # Import your CustomDataset

from torchvision.models.resnet import ResNet18_Weights
from train import YodaClassifier
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    val_loss = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            val_loss += criterion(outputs, labels).item()


    accuracy = total_correct / total_samples
    avg_val_loss = val_loss / len(dataloader)

    model.train()  # Set the model back to training mode
    return accuracy, avg_val_loss

def main():
    # Set the root directory of your dataset
    dataset_root = 'data/Kitti8_ROIs'

    # Instantiate the dataset for test
    test_dataset = CustomDataset(root_dir=dataset_root, mode='test', transform=transforms.ToTensor(), target_size=(150, 150))

    # Use DataLoader for efficient batching
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Initialize the model
    num_classes = 2  # Assuming binary classification (e.g., Car or NoCar)
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1)

    # Load the trained weights
    model.load_state_dict(torch.load('yoda_classifier_test1.pth'))

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    accuracy, avg_val_loss = evaluate(model, test_loader, criterion, device)

    print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {avg_val_loss:.4f}')

if __name__ == "__main__":
    main()
