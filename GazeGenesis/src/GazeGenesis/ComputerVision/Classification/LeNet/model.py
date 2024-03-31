#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class LeNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(LeNet, self).__init__()
        print("MODEL: LeNet")

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0)), # padding=(2,2) for MNIST
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            nn.ReLU(),
        )

        self.tail_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )


    def forward(self, x):
        x = self.core_layers(x)
        x = x.reshape(x.shape[0], -1)
        return self.tail_layers(x)

