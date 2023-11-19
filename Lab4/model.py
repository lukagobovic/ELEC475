import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet18_Weights
import time  # Import the time module
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
from customdataset import CustomDataset  # Import the CustomDataset from the module



class YodaClassifier(nn.Module):
    def __init__(self, num_classes, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(YodaClassifier, self).__init__()
        resnet18 = models.resnet18(weights=weights)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        in_features = resnet18.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = self.resnet18(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x
