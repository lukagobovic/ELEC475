import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from model import autoencoderMLP4Layer
import argparse

parser = argparse.ArgumentParser(description='Run lab1 script with model weights.')
parser.add_argument('-l', '--model-weights', type=str, required=True,
                    help='Path to the model weights file.')
args = parser.parse_args()
model_weights_path = args.model_weights

transform = transforms.Compose([transforms.ToTensor()])

test_dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

total_images = len(test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = autoencoderMLP4Layer(N_input=784, N_bottlenecks=8).to(device)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():  
    def display_input_output(img, output):
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(img.squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
        f.add_subplot(1,2,2)
        plt.imshow(output.squeeze().detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.show()

    num_samples = len(test_loader.dataset)
    indices = random.sample(range(num_samples), 2)

    # Loop through the selected indices, resize the images, pass them through the model, and display input and output
    for i in indices:
        input, _ = test_loader.dataset[i]
        input = input.view(1, -1).to(device)  
        output = model(input)
        display_input_output(input, output)

def add_gaussian_noise(tensor, mean, std):
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

with torch.no_grad():  
# Function to display input, noisy, and output images side by side
    def display_images(inputs, noisy, outputs):
        fig, axes = plt.subplots(ncols = 3, nrows = 2)
        for i in range(2):
            axes[i, 0].imshow(inputs[i].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 0].set_title("Original")
            
            axes[i, 1].imshow(noisy[i].squeeze().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 1].axis('off')
            axes[i, 1].set_title("Noisy")
            
            axes[i, 2].imshow(outputs[i].squeeze().detach().cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, 2].axis('off')
            axes[i, 2].set_title("Reconstructed")
        
        plt.show()

    random_indices = random.sample(range(total_images), 2)
    random_images = [test_dataset[i][0].to(device) for i in random_indices]

    inputs = torch.stack(random_images)
    noisy_inputs = add_gaussian_noise(inputs, mean=0.0, std=0.2).to(device)
    output = model(noisy_inputs.view(2, -1))
    display_images(inputs, noisy_inputs, output)


def interpolate_and_plot_multiple(model, image_pairs_list, num_steps):
    num_pairs = len(image_pairs_list)
    num_rows = 3
    num_cols = num_steps + 2

    plt.figure(figsize=(15, 5 * num_pairs))

    for i, (image1, image2) in enumerate(image_pairs_list):
        z1 = model.encode(image1.view(1, -1)).to(image1.device)
        z2 = model.encode(image2.view(1, -1)).to(image2.device)

        alphas = torch.linspace(0, 1, num_steps, device=image1.device)
        interpolated_images = []

        for alpha in alphas:
            interpolated_z = alpha * z1 + (1 - alpha) * z2
            reconstructed_image = model.decode(interpolated_z.unsqueeze(0)).view(1, 28, 28)
            interpolated_images.append(reconstructed_image)

        plt.subplot(num_rows, num_cols, i * num_cols + 1)
        plt.imshow(image2.squeeze().cpu().detach().numpy(), cmap='gray')
        plt.axis('off')

        for j, alpha in enumerate(alphas):
            plt.subplot(num_rows, num_cols, i * num_cols + j + 2)
            plt.imshow(interpolated_images[j].squeeze().cpu().detach().numpy(), cmap='gray')
            plt.axis('off')

        plt.subplot(num_rows, num_cols, i * num_cols + num_steps + 2)
        plt.imshow(image1.squeeze().cpu().detach().numpy(), cmap='gray')
        plt.axis('off')

    plt.show()

image_pairs_list = []
for _ in range(3):
    random_indices = random.sample(range(total_images), 2)
    random_images = [test_dataset[i][0].to(device) for i in random_indices]
    image_pairs_list.append((random_images[0], random_images[1]))

interpolate_and_plot_multiple(model, image_pairs_list, num_steps=8)

