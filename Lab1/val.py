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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.2):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(),device=tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
with torch.no_grad():  # Disable gradient calculations
# Function to display input, noisy, and output images side by side
    def display_images(input, noisy, output):
        fig, axes = plt.subplots(ncols = 3, nrows = 3)
        for i in range(3):
            axes[i, 0].imshow(input[i].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title("Original")
            
            axes[i, 1].imshow(noisy[i].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 1].axis('off')
            axes[i, 1].set_title("Noisy")
            
            axes[i, 2].imshow(output[i].squeeze().detach().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 2].axis('off')
            axes[i, 2].set_title("Reconstructed")
        
        plt.show()

    # Loop through the test dataset, pass each image through the model, and display input, noisy, and output
    for batch in test_loader:
        inputs = batch[0].to(device)  # Move inputs to GPU if available
        noisy_inputs = AddGaussianNoise(0., 0.2)(inputs)
        output = model(noisy_inputs.view(3, -1))  # Pass the noisy inputs through the model
        display_images(inputs, noisy_inputs, output)


        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(inputs[0].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
        # axes[1].imshow(inputs[1].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
        # axes[2].imshow(inputs[2].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
        # plt.show()
        # print(type(test_loader))
        # print(inputs.shape)  # Concatenate inputs