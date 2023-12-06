import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from customdataset import PetNoseDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
from torch.multiprocessing import freeze_support
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from torchsummary import summary


class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super(RegressionHead, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels, num_outputs)

    def forward(self, x):
        x = self.global_avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.fc(x)

# Load pre-trained ResNet-18 model
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
in_channels = resnet18.fc.in_features


model = nn.Sequential(
    *list(resnet18.children())[:-2],  # Remove the last two layers
    RegressionHead(in_channels, 2) 
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs,patience,plot_path,pthpath):
    print("training")
    train_losses = []
    val_losses = []
    total_start_time = time.time()

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        # Training
        model.train()
        epoch_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.view(-1, 2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_epoch_loss)
        scheduler.step(average_epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_images, val_labels in test_dataloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_labels = val_labels.view(-1, 2)

                val_loss += criterion(val_outputs, val_labels).item()

        average_val_loss = val_loss / len(test_dataloader)
        val_losses.append(average_val_loss)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Training Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_epoch_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Time: {epoch_time:.2f} seconds')

        # Check for validation loss rising
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered. Validation loss hasn't improved for {patience} consecutive epochs.")
            break

    total_end_time = time.time()
    total_time_elapsed = total_end_time - total_start_time
    print(f"Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path)

    # Save the trained model
    torch.save(model.state_dict(), pthpath)

def euclidean_distance(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1))


def evaluate_model(model, test_dataloader,printimages):
    print("evaluating")
    model.eval()
    all_distances = []

    total_images = 0
    total_time_taken = 0.0
    
    with torch.no_grad():
        for batch_idx, (val_images, val_labels) in enumerate(test_dataloader):
            batch_start_time = time.time()
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_labels = val_labels.view(-1, 2)

            # Iterate through all images in the batch
            for i in range(val_images.shape[0]):
                # Get model predictions from val_outputs
                total_images += 1
                predictions = val_outputs[i].squeeze().cpu().numpy()

                # Move the tensor to the CPU before converting to a numpy array
                image_np = (val_images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Convert the image to BGR format (OpenCV uses BGR)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Scale true and predicted nose positions based on the displayed image size
                true_nose_scaled = (
                    int(val_labels[i][0] * image_bgr.shape[1] / val_images.shape[3]),
                    int(val_labels[i][1] * image_bgr.shape[0] / val_images.shape[2])
                )
                predicted_nose_scaled = (
                    int(predictions[0] * image_bgr.shape[1] / val_images.shape[3]),
                    int(predictions[1] * image_bgr.shape[0] / val_images.shape[2])
                )

                # Calculate the Euclidean distance between predicted and true nose positions
                distance = np.sqrt((predicted_nose_scaled[0] - true_nose_scaled[0]) ** 2 +
                                   (predicted_nose_scaled[1] - true_nose_scaled[1]) ** 2)
                all_distances.append(distance)

                if printimages == "True" and i == 0: # Only draw the first image per batch
                    # Draw circles on the image for true and predicted nose positions
                    cv2.circle(image_bgr, true_nose_scaled, 8, (0, 255, 0), 1)  # Ground truth in green
                    cv2.circle(image_bgr, predicted_nose_scaled, 8, (0, 0, 255), 1)  # Prediction in red

                    # Display the image with ground truth and predictions
                    cv2.imshow(f"Batch {batch_idx}, Image {i}", image_bgr)
                    cv2.waitKey(10000)  # Display the image for 2 seconds (2000 milliseconds)

                    # Check if the window exists before trying to destroy it
                    if cv2.getWindowProperty(f"Batch {batch_idx}, Image {i}", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyAllWindows()

                    # if key == ord('q'):
                    #     exit(0)  # Break the loop if 'q' is pressed

            batch_end_time = time.time()
            batch_time_taken = batch_end_time - batch_start_time
            total_time_taken += batch_time_taken
            print(f"Batch {batch_idx} took {batch_time_taken:.2f} seconds for {val_images.shape[0]} images")
    
    
    avg_time_per_image = total_time_taken / total_images if total_images > 0 else 0.0
    print(f"Processed {total_images} images in {total_time_taken:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image * 1000:.2f} milliseconds")
    # Calculate and print the localization accuracy statistics for the entire dataset
    all_distances = np.array(all_distances)
    print("Localization Accuracy Statistics:")
    print(f"Minimum Distance: {np.min(all_distances):.4f}")
    print(f"Mean Distance: {np.mean(all_distances):.4f}")
    print(f"Maximum Distance: {np.max(all_distances):.4f}")
    print(f"Standard Deviation: {np.std(all_distances):.4f}")


def main(args):
    freeze_support()
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    train_dataset = PetNoseDataset(annotation_file='labels/train_noses3.txt', images_folder='images', transform=transform_train)
    test_dataset = PetNoseDataset(annotation_file='labels/test_noses.txt', images_folder='images', transform=transform_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set up training parameters
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')
    criterion = nn.SmoothL1Loss()
    
    if args.mode == "train":
        train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=args.num_epochs, patience = args.patience, plot_path = args.loss_plot_path, pthpath = args.pth_file)

    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.pth_file))
        evaluate_model(model, test_dataloader,printimages = args.display_images)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pet Nose Localization")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience")
    parser.add_argument("--pth_file", type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument("--loss_plot_path", type=str, default="lossplot.png", help="Path to save the loss plot")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode to train or evaluate the model")
    parser.add_argument("--display_images", type=str, default="False", help="To print images or not")
    args = parser.parse_args()

    main(args)

