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
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate your model
model = autoencoderMLP4Layer(N_input=784, N_bottlenecks=8).to(device)
model.load_state_dict(torch.load("MLP.8.pth"))
model.eval()

# Choose two images (e.g., a 1 and a 4)
image1 = test_dataset[2][0].to(device)  # Adjust the index as needed
image2 = test_dataset[13][0].to(device)  # Adjust the index as needed

# Encode the chosen images
z1 = model.encode(image1.view(1, -1))
z2 = model.encode(image2.view(1, -1))

# Number of interpolation steps
num_steps = 10

# Linearly interpolate between the two bottleneck tensors
interpolated_z = torch.zeros(num_steps, *z1.size(), device=device)
for i in range(num_steps):
    alpha = i / (num_steps - 1)  # Interpolation factor
    interpolated_z[i] = alpha * z1 + (1 - alpha) * z2

# Decode the interpolated bottleneck tensors and plot the results
plt.figure(figsize=(15, 5))
for i in range(num_steps):
    reconstructed_image = model.decode(interpolated_z[i].unsqueeze(0)).view(1, 28, 28)
    plt.subplot(1, num_steps, i + 1)
    plt.imshow(reconstructed_image.squeeze().cpu().detach().numpy(), cmap='gray')
    plt.axis('off')

plt.show()
