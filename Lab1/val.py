import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer  # Import your autoencoder model class

# Define a transform to convert the PIL image to a tensor
transform = transforms.Compose([transforms.ToTensor()])

# Create a dataset and DataLoader for your MNIST data
test_dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate your model
model = autoencoderMLP4Layer(N_input=784, N_bottlenecks=8).to(device)
model.load_state_dict(torch.load("MLP.8.pth"))
model.eval()
with torch.no_grad():  # Disable gradient calculations
    # Function to display input and output images side by side
    def display_input_output(img, output):
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(img.squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
        f.add_subplot(1,2,2)
        plt.imshow(output.squeeze().detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.show()

    # Loop through the test dataset, resize each image to 784 dimensions, pass them through the model, and display input and output
    for i, (input, _) in enumerate(test_loader):
        input = input.view(input.size(0), -1).to(device)  # Flatten the input and move to GPU if available
        output = model(input)
        display_input_output(input, output)
