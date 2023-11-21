import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.nn.functional as F

class YodaClassifier(nn.Module):
    def __init__(self, num_classes, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(YodaClassifier, self).__init__()
        resnet18 = models.resnet18(weights=weights)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        in_features = resnet18.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.resnet18(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)  
        return x
