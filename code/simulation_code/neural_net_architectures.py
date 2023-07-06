import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# FMNIST classifier
class FashionMNIST_Classifier(nn.Module):
    def __init__(self):
        # Inherit the __init__() method from nn.Module (call __init__() from the parent class of FashionMNIST_Classifier)
        super(FashionMNIST_Classifier, self).__init__()

        # Define layers of the FashionMNIST_Classifier
        # padding = (k - 1) / 2 to preserve size for square kernel of size 3 and stride 1
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.relu0 = nn.ReLU()
        self.drop0 = nn.Dropout(p = 0.5)
        self.pool0 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Next add another layers batch: convolutional, batch normalization, relu, max pooling
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p = 0.5)
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        # Finally add last layers batch, that includes flattening the output, and 2 linear layers (only add linear here)
        self.lin0  = nn.Linear(in_features = 64 * 7 * 7, out_features = 128)
        self.drop2 = nn.Dropout(p = 0.5)
        self.lin1  = nn.Linear(in_features = 128, out_features = 10)

    # Push the input through the net
    def forward(self, x):
        # Reshape the input data to have desired shape (specific for given data, change as needed, here 1 channel for MNIST)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2]) 
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.drop0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1) # Flatten the tensor, except the batch dimension
        x = self.lin0(x)
        x = self.drop2(x)
        x = self.lin1(x)

        return x

# CIFAR10 classifer
class CIFAR10_Classifier(nn.Module):
    def __init__(self):
        # Inherit __init__ from nn.Module
        super(CIFAR10_Classifier, self).__init__()

        # Define layers for CIFAR10_Classifier
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Dropout(p = 0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Dropout(p = 0.5)
        )

        self.fcls = nn.Sequential(
            nn.Linear(in_features = 64 * 8 * 8, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 10)
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2]) 

        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # Flatten the tensor, except the batch dimension
        x = self.fcls(x)

        return x