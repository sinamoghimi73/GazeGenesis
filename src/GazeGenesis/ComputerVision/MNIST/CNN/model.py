#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        print("MODEL: CNN")

        self.layers = None


    def forward(self, x):
        return self.layers(x)

