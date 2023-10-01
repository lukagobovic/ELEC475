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
from torchvision.datasets import MNIST 
import datetime

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, savepath):
    print("training")
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device=device) 
            imgs = imgs.view(imgs.size(0), -1)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train.append(loss_train / len(train_loader))  

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))

    plt.figure()
    plt.plot(losses_train)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(savepath)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('-z', type=int, default=8, help='Bottleneck size')
    parser.add_argument('-e', type=int, default=50, help='Number of epochs')
    parser.add_argument('-b', type=int, default=2048, help='Batch size')
    parser.add_argument('-s', type=str, default='MLP.8.pth', help='Model save path')
    parser.add_argument('-p', type=str, default='loss_MLP.8.png', help='Plot save path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Print the device being used

    model = autoencoderMLP4Layer(N_bottlenecks=args.z).to(device)
    summary(model, (1, 784)) 

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train = True, download = True, transform = train_transform)
    test_dataset = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,args.b, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set,args.b,shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    train(args.e, optimizer, model, loss_fn, train_loader, scheduler, device,args.p)

    torch.save(model.state_dict(), args.s)

    








