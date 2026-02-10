import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1A = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv1B = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2A = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2B = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dense3 = nn.Linear(128 * 7 * 7, 32)
        self.dense4 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1A(x))
        x = F.relu(self.conv1B(x))
        x = self.pool(x)
        x = F.relu(self.conv2A(x))
        x = F.relu(self.conv2B(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x


class AlphabetModel(nn.Module):
    def __init__(self, num_classes=26):
        super(AlphabetModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)