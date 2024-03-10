#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class VGGNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, mode: int = 16):
        super(VGGNet, self).__init__()
        print(f"MODEL: VGG-{mode}")

        self.in_channels = in_channels

        self.vgg_dict = {
            11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        self.core_layers = self.create_conv_layers(self.vgg_dict[mode])

        self.tail_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.core_layers(x)
        x = x.reshape(x.shape[0], -1)
        return self.tail_layers(x)

    def create_conv_layers(self, arch):
        in_channels = self.in_channels
        layers = []
        for x in arch:
            if isinstance(x, int):
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    # nn.BatchNorm2d(x), # it introduced after VGG paper
                    nn.ReLU(),
                ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)
