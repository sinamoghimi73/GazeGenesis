#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


resnet_folds = {
    # layer_count, num_fold, stride_size
    "18": [(2, 1), (2, 2), (2, 2), (2, 2)],
    "34": [(3, 1), (4, 2), (6, 2), (3, 2)],
    "50": [(3, 1), (4, 2), (6, 2), (3, 2)],
    "101": [(3, 1), (4, 2), (23, 2), (3, 2)],
    "150": [(3, 1), (8, 2), (36, 2), (3, 2)],
}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, identity_downsample = None):
        super(Block, self).__init__()
        self.expansion = 4

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, stride = stride, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, stride = stride, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.core_layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity

        return x.ReLU()






class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes = 10, mode = 50):
        super(ResNet, self).__init__()
        print(f"MODEL: ResNet-{mode}")

        self.in_channels = in_channels

        self.head_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        
        self.layer_folds = resnet_folds[version]
        self.core_layers = self.makeLayers(in_channels, )


    def _make_layer(self):
        layers = []
        identity_downsample = None
        for num_fold, stride in self.layer_folds:
            layers.append([Block(in_channels=self.in_channels, out_channels=)])