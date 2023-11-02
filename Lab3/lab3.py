import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


class encoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )

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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

class VGGCustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGCustomClassifier, self).__init__()
        self.vgg_encoder = encoder.encoder
        self.fc1 = nn.Linear(8192, 256)  # Adjusted for your specific VGG configuration
        self.dropout1 = nn.Dropout(0.5)  # Add dropout with a probability of 0.5
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.vgg_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

# Train the model
def train(model, train_loader, criterion, optimizer, num_epochs, scheduler):
    model.train()
    losses = [] 
    start_time = time.time()  # Start the timer
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        scheduler.step(running_loss)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60.0:.2f} minutes.")

    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('CIFAR100B256')
    plt.show()

num_classes = 100  # Number of classes in CIFAR-100

# Load the pre-trained encoder model
encoder_path = 'encoder.pth'  # Replace with the path to your pre-trained encoder model

# Initialize your model with the pre-trained encoder's architecture
model = VGGCustomClassifier(num_classes).to(device)

# Load the pre-trained weights into your model's encoder
model.vgg_encoder.load_state_dict(torch.load(encoder_path))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')  
num_epochs = 60 # You can adjust this value

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs, scheduler)

torch.save(model.state_dict(), 'CIFAR100B256.pth')

# Load the trained model for evaluation
model = VGGCustomClassifier(num_classes).to(device)
model.load_state_dict(torch.load('CIFAR100B256.pth'))
model.eval()

# Step 5: Evaluate the model and calculate top-1 and top-5 error rates
def evaluate(model, test_loader):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    #with torch.no_d各而寧配额貴。
    
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

evaluate(model, test_loader)