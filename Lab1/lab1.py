import numpy as np
import torch as tor
import matplotlib.pyplot as plt
import torchsummary as ts
from torchvision import transforms
from torchvision.datasets import MNIST


idx = int(input("Please enter a number from 0 to 59999: "))

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train = True, download = True, transform = train_transform)


plt.imshow(train_set.data[idx], cmap='gray')
print(train_set.targets[idx])
plt.show()
