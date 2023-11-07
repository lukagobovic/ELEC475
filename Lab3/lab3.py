import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CustomClassifier
import time
import argparse

# Train the model
def train(model, train_loader, criterion, optimizer, num_epochs, scheduler, device, loss_path):
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        losses.append(running_loss / len(train_loader))
        scheduler.step(running_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60.0:.2f} minutes.")

    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(loss_path)
    plt.show()


def evaluate(model, test_loader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            _, predicted_top5 = outputs.topk(5, 1)
            for i in range(labels.size(0)):
                if labels[i] in predicted_top5[i]:
                    correct_top5 += 1

    top1_error = 1 - correct_top1 / total
    top5_error = 1 - correct_top5 / total
    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    

    print(f"Top-1 Error: {top1_error * 100:.2f}%")
    print(f"Top-5 Error: {top5_error * 100:.2f}%")
    print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = CustomClassifier(args.num_classes).to(device)

    if args.encoder_path:
        model.vgg_encoder.load_state_dict(torch.load(args.encoder_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    if args.mode == "train":
        train(model, train_loader, criterion, optimizer, args.num_epochs, scheduler, device, args.loss_plot_path)
        torch.save(model.state_dict(), args.output_path)
    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.output_path))
        model.eval()
        evaluate(model, test_loader,device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-100 Classifier")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--encoder_path", type=str, default="encoder.pth", help="Path to pre-trained encoder weights")
    parser.add_argument("--output_path", type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument("--loss_plot_path", type=str, default="lossplot", help="Path to save the loss plot")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes in CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode to train or evaluate the model")
    args = parser.parse_args()

    main(args)




