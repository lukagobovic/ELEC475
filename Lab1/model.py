import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary


class autoencoderMLP4Layer(nn.Module):
  
  def __init__(self, N_input = 784, N_bottlenecks = 8, N_output = 784):
    super(autoencoderMLP4Layer, self).__init__()
    N2 = 392
    self.fc1 = nn.Linear(N_input,N2)
    self.fc2 = nn.Linear(N2, N_bottlenecks)
    self.fc3 = nn.Linear(N_bottlenecks, N2)
    self.fc4 = nn.Linear(N2, N_output)
    self.type = "MLP4"
    self.input_shape = (1, 784) 

  def encode(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      return x

  def decode(self, z):
      z = self.fc3(z)
      z = F.relu(z)
      z = self.fc4(z)
      z = torch.sigmoid(z)
      return z

  def forward(self, x):
      return self.decode(self.encode(x)) 
