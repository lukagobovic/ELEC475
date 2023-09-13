import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from torchsummary import summary
from model import autoencoderMLP4Layer
import numpy as np
import matplotlib.pyplot as plt
import torchsummary as ts
from torchvision import transforms
from torchvision.datasets import MNIST  # Import your autoencoder class from model.py
import datetime

def train(n_epochs, optimizer,model,loss_fn,train_loader,scheduler,device):
  print("training")
  model.train()
  losses_train = []

  for epoch in range(1, n_epochs+1):
    print('epoch ', epoch)
    loss_train = 0.0
    for imgs, _ in train_loader:
      imgs = imgs.view(imgs.size(0),-1)
      imgs = imgs.to(device=device)
      outputs = model(imgs)
      loss = loss_fn(outputs, imgs)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += loss.item()

  scheduler.step(loss_train)

  losses_train +=[loss_train/len(train_loader)]

  print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(),epoch,loss_train/len(train_loader)))

  plt.figure()
  plt.plot(losses_train)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.savefig("./outputs")
  plt.close()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('-z', type=int, default=8, help='Bottleneck size')
    parser.add_argument('-e', type=int, default=50, help='Number of epochs')
    parser.add_argument('-b', type=int, default=2048, help='Batch size')
    parser.add_argument('-s', type=str, default='MLP.8.pth', help='Model save path')
    parser.add_argument('-p', type=str, default='loss_MLP.8.png', help='Plot save path')
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model setup
    model = autoencoderMLP4Layer(N_bottlenecks=args.z).to(device)
    summary(model, (1, 28 * 28))  # Print model summary

    # Data loader setup (You need to define your data loader here)
    # train_loader = ...
    train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_set = MNIST('./data/mnist', train = True, download = True, transform = train_transform)


    train_loader = torch.utils.data.DataLoader(train_set,args.b,shuffle=True)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    # Training
    train(args.e, optimizer, model, loss_fn, train_loader, scheduler, device)

    # Save the trained model
    torch.save(model.state_dict(), args.s)







