import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from AdaIN_net import AdaIN_net, encoder_decoder
from custom_dataset import custom_dataset  # Import the custom dataset class
from torch.optim import Adam

def train(content_dir, style_dir, gamma, epochs, batch_size, encoder_path, preview_path, use_cuda):
    # Define the device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders for content and style images
    content_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    content_dataset = custom_dataset(content_dir, transform=content_transform)
    content_loader = DataLoader(content_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    style_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    style_dataset = custom_dataset(style_dir, transform=style_transform)
    style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder = encoder_decoder.decoder

    model = AdaIN_net(encoder, decoder).to(device)  # Initialize decoder weights using the function from AdaIN_net.py


    # Define the optimizer
    optimizer = Adam(model.decoder.parameters(), lr=0.001)

    # Training loop
    # ...

    # Training loop
    for epoch in range(1, epochs + 1):
        for batch, (content_batch, style_batch) in enumerate(zip(content_loader, style_loader)):  # Loop through content and style loaders together
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            optimizer.zero_grad()

            # Perform style transfer
            content_loss, style_loss = model(content_batch, style_batch)

            # Enable gradients for content_loss and style_loss
            content_loss.requires_grad = True
            style_loss.requires_grad = True

            # Calculate the total loss
            total_loss = gamma * content_loss + style_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch}/{epochs}] Batch [{batch + 1}/{len(content_loader)}] Content Loss: {content_loss.item():.4f} Style Loss: {style_loss.item():.4f}")

            if (batch + 1) % 5 == 0:
                # Save a preview image
                with torch.no_grad():
                    preview_image = model(content_batch, style_batch)
                save_path = os.path.join(preview_path, f"preview_epoch_{epoch}_batch_{batch + 1}.png")
                torchvision.utils.save_image(preview_image, save_path)

        # Save the final model at the end of each epoch if needed
        torch.save(model.encoder.state_dict(), f"final_encoder_epoch_{epoch}.pth")
        torch.save(model.decoder.state_dict(), f"final_decoder_epoch_{epoch}.pth")

    # ...

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AdaIN style transfer model")
    parser.add_argument("-content_dir", type=str, help="Directory containing content images")
    parser.add_argument("-style_dir", type=str, help="Directory containing style images")
    parser.add_argument("-gamma", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("-e", type=int, help="Number of epochs")
    parser.add_argument("-b", type=int, help="Batch size")
    parser.add_argument("-l", "--encoder_path", type=str, help="Path to the encoder weights")
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
        preview_path=args.preview_path,
        use_cuda=args.cuda.lower() == 'y'
    )