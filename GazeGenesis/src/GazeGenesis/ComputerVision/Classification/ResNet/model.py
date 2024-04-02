#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name
import copy

resnet_folds = {
    # layer_count: kernel_size, num_features, stride

    50: [  [(1, 64, 1), (3, 64, 1), (1, 256, 1)] * 3,
           [(1, 128, 2), (3, 128, 2), (1, 512, 2)] * 4,
           [(1, 256, 2), (3, 256, 2), (1, 1024, 2)] * 6,
           [(1, 512, 2), (3, 512, 2), (1, 2048, 2)] * 3,
           ],

    18: [  [(3, 64, 1), (3, 64, 1)] * 2,
           [(3, 128, 2), (3, 128, 2)] * 2,
           [(3, 256, 2), (3, 256, 2)] * 2,
           [(3, 512, 2), (3, 512, 2)] * 2,
           ],

    34: [  [(3, 64, 1), (3, 64, 1)] * 3,
           [(3, 128, 2), (3, 128, 2)] * 4,
           [(3, 256, 2), (3, 256, 2)] * 6,
           [(3, 512, 2), (3, 512, 2)] * 3,
           ],

    101: [ [(1, 64, 1), (3, 64, 1), (1, 256, 1)] * 3,
           [(1, 128, 2), (3, 128, 2), (1, 512, 2)] * 4,
           [(1, 256, 2), (3, 256, 2), (1, 1024, 2)] * 23,
           [(1, 512, 2), (3, 512, 2), (1, 2048, 2)] * 3,
           ],

    150: [ [(1, 64, 1), (3, 64, 1), (1, 256, 1)] * 3,
           [(1, 128, 2), (3, 128, 2), (1, 512, 2)] * 8,
           [(1, 256, 2), (3, 256, 2), (1, 1024, 2)] * 36,
           [(1, 512, 2), (3, 512, 2), (1, 2048, 2)] * 3,
           ],
}

# resnet_expansion = {
#     18: 1,
#     34: 1,
#     50: 4,
#     101: 4,
#     152: 4
# }




class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 1, padding = 0):
        super(Block, self).__init__()

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.core_layers(x)

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mode = 18, conv_level = 0):
        super(ResidualBlock, self).__init__()
        self.resnet_architecture = resnet_folds[mode]
        self.conv_level = conv_level

        layers = []
        in_features = copy.copy(in_channels)
        for kernel_size, out_features, stride in self.resnet_architecture[self.conv_level]:
                padding = 1 if kernel_size == 3 else 0
                layers.append(Block(in_channels = in_features, out_channels = out_features, stride = stride, kernel_size = kernel_size, padding = padding))
                layers.append(nn.ReLU())
                in_features = out_features
        layers.pop()                

        self.identity = Block(in_channels = in_channels, out_channels = in_features, stride = stride, kernel_size = 1, padding = 0)
        self.out_channels = in_features

        self.core_layers = nn.Sequential(*layers)

        self.tail_layers = nn.ReLU()


    def forward(self, x):
        identity = self.identity(x)
        print("identity", identity.shape)
        x = self.core_layers(x)
        print("x", x.shape)
        x += identity
        return self.tail_layers(x)




class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes = 10, mode = 18):
        super(ResNet, self).__init__()
        print(f"MODEL: ResNet-{mode}")

        self.in_channels = in_channels
        self.head_out_channels = 64
        self.head_layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = self.head_out_channels, kernel_size=7, stride=2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        
        folds = []
        for i in range(4):
            residual_block = ResidualBlock(in_channels = self.head_out_channels, mode = mode, conv_level = i)
            folds.append(residual_block)
            self.head_out_channels = residual_block.out_channels
        
        self.core_layers = nn.Sequential(*folds)

        self.tail_layers_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.tail_layers_2 = nn.Sequential(
            nn.Linear(self.head_out_channels, num_classes),
        )

    def forward(self, x):
        x = self.head_layers(x)
        x = self.core_layers(x)
        x = self.tail_layers_1(x)
        x = x.flatten(1)
        x = self.tail_layers_2(x)
        return x

if __name__ == "__main__":
    # t = ResidualBlock(in_channels = 3, mode = 50, conv_level = 0)
    x = torch.randn(1, 3, 224, 224)

    # print(x.shape)
    # print(t(x).shape)

    t = ResNet(in_channels = 3, mode = 18)
    print(x.shape)
    print(t(x).shape)
