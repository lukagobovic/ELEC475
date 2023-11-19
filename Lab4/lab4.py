import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet18_Weights
from model import YodaClassifier
import time  
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
from customdataset import CustomDataset  
import argparse

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

def evaluate(model, dataloader, criterion, device):
    model.eval() 
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

    return accuracy, avg_val_loss

def main(args):
    freeze_support()
    print('loading')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    dataset_root = 'data/Kitti8_ROIs'

    train_dataset = CustomDataset(
        root_dir=dataset_root,
        mode='train',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

    print("done loading")
    num_classes = 2 
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')


    if args.mode == "train":
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.num_epochs, device, patience=5)
        torch.save(model.state_dict(), args.output_path)
        #plot the losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(args.loss_plot_path)
        plt.show()

    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.pth_file))
        accuracy, avg_val_loss = evaluate(model, test_loader, criterion, device)
        print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {avg_val_loss:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YODA Object Detection Test")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--pth_file", type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument("--loss_plot_path", type=str, default="lossplot", help="Path to save the loss plot")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode to train or evaluate the model")
    args = parser.parse_args()

    main(args)

