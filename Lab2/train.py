import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AdaIN_net import AdaIN_net, encoder_decoder
from custom_dataset import custom_dataset  # Import the custom dataset class
from torch.optim import Adam
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import torch.utils.data as data
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
    
def adjust_learning_rate(optimizer, iteration_count,learning_rate):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loss_plot(content_losses, style_losses,total_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label='Content + Style', color='blue')
    plt.plot(content_losses, label='Content', color='orange')
    plt.plot(style_losses, label='Style', color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def train(content_dir, style_dir, gamma, epochs, batch_size, encoder_path, decoder_path, preview_path, use_cuda):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    content_transform = train_transform()
    style_transform = train_transform()
    
    content_dataset = custom_dataset(content_dir,content_transform )
    content_loader = DataLoader(content_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    style_dataset = custom_dataset(style_dir,style_transform)
    style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Initialize the model
    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder = encoder_decoder.decoder

    model = AdaIN_net(encoder, decoder).to(device) 
    model.train() 

    optimizer = Adam(decoder.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, verbose = True, gamma = 0.85)  # Adjust the step_size and gamma as needed

    content_losses = []
    style_losses = []
    total_losses = []
  
    current_content_losses = 0.0
    current_style_losses = 0.0
    current_total_losses = 0.0  

    total_data_size = len(content_loader.dataset)  # Assuming content_dataloader is your data loader
    batch_size = content_loader.batch_size  # Assuming content_dataloader is your data loader

    n_batches = total_data_size // batch_size

    if total_data_size % batch_size != 0:
        n_batches += 1 
  # Set the model to training mode

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # print('epoch:',epoch)
        for batch in range(n_batches):  
            current_content_losses = 0.0
            current_style_losses = 0.0
            current_total_losses = 0.0

            content_images = next(iter(content_loader)).to(device)
            style_images = next(iter(style_loader)).to(device)

            # Perform style transfer
            content_loss, style_loss = model(content_images, style_images)

            style_loss = gamma  * style_loss

            # Calculate the total loss
            total_loss = content_loss + style_loss 

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()  # Update the model's parameters after backward pass


            current_content_losses += content_loss.item()
            current_style_losses += style_loss.item()
            current_total_losses += total_loss.item()
            #scheduler.step()  # Adjust the learning rate after each batch
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch + 1}/{n_batches}]")

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Calculate the estimated remaining time for training
        remaining_time = (epochs - epoch) * epoch_time

        # Print timing information
        print(f"Epoch [{epoch}/{epochs}] - Time taken: {epoch_time:.2f} seconds - Remaining time: {remaining_time:.2f} seconds")

        scheduler.step()
        # Save the final model at the end of each epoch if needed  
        # avg_content_loss = total_content_loss / len(content_loader)
        # avg_style_loss = total_style_loss / len(content_loader)
        # avg_total_loss = total_total_loss / len(content_loader)

        content_losses.append(current_content_losses / n_batches)
        style_losses.append(current_style_losses / n_batches)
        total_losses.append(current_total_losses / n_batches)

        print(f"Content Loss: {current_content_losses/n_batches:.2f}, Style Loss: {current_style_losses/n_batches:.2f}, Total Loss: {current_total_losses/n_batches:.2f}")

        torch.save(model.decoder.state_dict(), decoder_path) 

    loss_plot(content_losses,style_losses,total_losses, preview_path)           
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AdaIN style transfer model")
    parser.add_argument("-content_dir", type=str, help="Directory containing content images")
    parser.add_argument("-style_dir", type=str, help="Directory containing style images")
    parser.add_argument("-gamma", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("-e", type=int, help="Number of epochs")
    parser.add_argument("-b", type=int, help="Batch size")
    parser.add_argument("-l", "--encoder_path", type=str, help="Path to the encoder weights")
    parser.add_argument("-s", "--decoder_path", type=str, help="Path to save decoder weights")
    parser.add_argument("-p", "--preview_path", type=str, help="Path to save preview images")
    parser.add_argument("-cuda", type=str, help="Use CUDA (Y/N)")

    args = parser.parse_args()

    train(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        gamma=args.gamma,
        epochs=args.e,
        batch_size=args.b,
        encoder_path=args.encoder_path,
        decoder_path = args.decoder_path,
        preview_path=args.preview_path,
        use_cuda=args.cuda.lower() == 'y'
    )