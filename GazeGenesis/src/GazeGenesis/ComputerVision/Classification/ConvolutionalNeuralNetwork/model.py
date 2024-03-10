#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(CNN, self).__init__()
        print("MODEL: CNN")

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )

        self.tail_layers = nn.Linear(8*7*7, num_classes)


    def forward(self, x):
        x = self.core_layers(x)
        x = x.reshape(x.shape[0], -1)
        return self.tail_layers(x)

