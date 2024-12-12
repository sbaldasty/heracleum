from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import BatchNorm2d

import torch.nn.functional as F


class SimpleCNN(Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ModifiedCNN(Module):
    """ CNN classifier model adapted from 'CIFAR10-CNN using PyTorch on Kaggle"""
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        self.conv1 = Conv2d(3, 32, 3, padding = 1)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, 3, stride = 1, padding = 1) # kernel size = 3
        self.bn2 = BatchNorm2d(64)
        self.pool1 = MaxPool2d(2, 2) # output = 64 * 16 * 16

        self.conv3 = Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.bn3 = BatchNorm2d(128)
        self.conv4 = Conv2d(128, 128, 3, stride = 1, padding = 1)
        #self.pool2 = MaxPool2d(2, 2) # output = 128 * 8 * 8

        self.conv5 = Conv2d(128, 256, 3, stride = 1, padding = 1)
        self.bn4 = BatchNorm2d(256)
        self.conv6 = Conv2d(256, 256, 3, stride = 1, padding = 1)
        # self.pool = MaxPool2d(2, 2) # output 256 * 4 * 4

        self.fc1 = Linear(256 * 4 * 4, 1024)
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = x.view(-1, 256 * 4 * 4) # flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
        

        