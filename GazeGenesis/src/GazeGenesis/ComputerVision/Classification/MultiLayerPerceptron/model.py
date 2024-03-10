#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        print("MODEL: MLP")

        self.layers = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )


    def forward(self, x):
        return self.layers(x)

