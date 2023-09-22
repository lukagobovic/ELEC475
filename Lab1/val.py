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
    i = 0
    for batch in test_loader:
        if i == 2:
            break
        else:
            inputs = batch[0].to(device)  # Move inputs to GPU if available
            noisy_inputs = AddGaussianNoise(0., 0.2)(inputs).to(device)
            output = model(noisy_inputs.view(3, -1))  # Pass the noisy inputs through the model
            display_images(inputs, noisy_inputs, output)
        i+=1

with torch.no_grad():
    def interpolate_and_plot(model, image1, image2, num_steps):
        # Flatten and encode the input images
        z1 = model.encode(image1.view(1, -1)).to(device)
        z2 = model.encode(image2.view(1, -1)).to(device)

        # Linearly interpolate between the two bottleneck tensors
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_z = torch.lerp(z1, z2, alphas.unsqueeze(1))

        # Decode the interpolated bottleneck tensors and plot the results
        plt.figure(figsize=(15, 5))
        for i in range(num_steps):
            reconstructed_image = model.decode(interpolated_z[i].unsqueeze(0)).view(1, 28, 28)
            plt.subplot(1, num_steps, i + 1)
            plt.imshow(reconstructed_image.squeeze().cpu().detach().numpy(), cmap='gray')
            plt.axis('off')

        plt.show()


    example_images = [test_dataset[i][0].to(device) for i in range(2)]  # Adjust as needed
    interpolate_and_plot(model, example_images[0], example_images[1], num_steps=8)

