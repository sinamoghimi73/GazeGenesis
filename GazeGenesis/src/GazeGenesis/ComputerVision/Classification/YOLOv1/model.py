#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name
from itertools import chain

conv_architecture = [
    # kernel_size, out_channel, stride, padding, max_pool
    # Block 1
    (7, 64, 2, 3, True), 

    # Block 2
    (3, 192, 1, 1, True), 

    # Block 3
    (1, 128, 1, 0, False),
    (3, 256, 1, 1, False),
    (1, 256, 1, 0, False),
    (3, 512, 1, 1, True),

    # Block 4
    [(1, 256, 1, 0, False), (3, 512, 1, 1, False)] * 4,
    (1, 512, 1, 0, False),
    (3, 1024, 1, 1, True),

    # Block 5
    [(1, 512, 1, 0, False), (3, 1024, 1, 1, False)] * 2,
    (3, 1024, 1, 1, True),

    # Block 6
    [(3, 1024, 1, 1, False)] * 2,

]

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channel: int = 3, kernel_size: int = 1, stride: int = 1, padding: int = 0, bias: bool = False, max_pool: bool = False):
        super(ConvBlock, self).__init__()

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            # nn.LeakyReLU(0.1),
            nn.SiLU() # the original paper is using ReLU
        )

        if max_pool:
            self.core_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.core_layers(x)
    

class YOLOv1(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, num_boxes: int = 2):
        super(YOLOv1, self).__init__()
        print("MODEL: YOLOv1")

        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.architecture_list = [item for sublist in conv_architecture for item in (sublist if isinstance(sublist, list) else [sublist])]
        self.architecture_blocks = []
        for block in self.architecture_list:
            kernel_size, out_channel, stride, padding, max_pool = block
            self.architecture_blocks.append(ConvBlock(in_channels=in_channels, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, max_pool=max_pool))
            in_channels = out_channel

        self.head_layers = nn.Sequential(*self.architecture_blocks)

        self.core_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * (num_classes + num_boxes * 5))
        )

    def forward(self, x):
        x = self.head_layers(x)
        x = self.core_layers(x)
        x = x.reshape(x.shape[0], (self.num_classes + self.num_boxes * 5), 7, 7)
        return x



# if __name__ == "__main__":
#     y = YOLOv1(in_channels = 3)

#     x = torch.randn(1, 3, 448, 448)

#     z = y(x).shape

#     print(z)
