#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name
import copy


def create_fold(feature_list, kernel_list, fold_size, first_stride):
    return [(kernel_list[j], feature_list[j], first_stride if (i == 0 and j == 0) else 1) for i in range(fold_size) for j in range(len(feature_list))]



resnet_folds = {
    # layer_count: kernel_size, num_features, stride

    18: [   create_fold(kernel_list = [3, 3], feature_list = [64, 64], fold_size = 2, first_stride = 1),
            create_fold(kernel_list = [3, 3], feature_list = [128, 128], fold_size = 2, first_stride = 2),
            create_fold(kernel_list = [3, 3], feature_list = [256, 256], fold_size = 2, first_stride = 2),
            create_fold(kernel_list = [3, 3], feature_list = [512, 512], fold_size = 2, first_stride = 2),
        ],

    34: [   
            create_fold(kernel_list = [3, 3], feature_list = [64, 64], fold_size = 3, first_stride = 1),
            create_fold(kernel_list = [3, 3], feature_list = [128, 128], fold_size = 4, first_stride = 2),
            create_fold(kernel_list = [3, 3], feature_list = [256, 256], fold_size = 6, first_stride = 2),
            create_fold(kernel_list = [3, 3], feature_list = [512, 512], fold_size = 3, first_stride = 2),
        ],

    50: [   create_fold(kernel_list = [1, 3, 1], feature_list = [64, 64, 256], fold_size = 3, first_stride = 1),
            create_fold(kernel_list = [1, 3, 1], feature_list = [128, 128, 512], fold_size = 4, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [256, 256, 1024], fold_size = 6, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [512, 512, 2048], fold_size = 3, first_stride = 2),
        ],
           
    101: [ 
            create_fold(kernel_list = [1, 3, 1], feature_list = [64, 64, 256], fold_size = 3, first_stride = 1),
            create_fold(kernel_list = [1, 3, 1], feature_list = [128, 128, 512], fold_size = 4, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [256, 256, 1024], fold_size = 23, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [512, 512, 2048], fold_size = 3, first_stride = 2),
        ],

    152: [ 
            create_fold(kernel_list = [1, 3, 1], feature_list = [64, 64, 256], fold_size = 3, first_stride = 1),
            create_fold(kernel_list = [1, 3, 1], feature_list = [128, 128, 512], fold_size = 8, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [256, 256, 1024], fold_size = 36, first_stride = 2),
            create_fold(kernel_list = [1, 3, 1], feature_list = [512, 512, 2048], fold_size = 3, first_stride = 2),
        ],
}




class WeightLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 1, padding = 0):
        super(WeightLayer, self).__init__()

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
                layers.append(WeightLayer(in_channels = in_features, out_channels = out_features, stride = stride, kernel_size = kernel_size, padding = padding))
                layers.append(nn.ReLU())
                in_features = out_features
        layers.pop()                

        if conv_level:
            stride = 2
        self.identity = WeightLayer(in_channels = in_channels, out_channels = out_features, stride = stride, kernel_size = 1, padding = 0)
        self.out_channels = in_features

        self.core_layers = nn.Sequential(*layers)

        self.tail_layers = nn.ReLU()


    def forward(self, x):
        identity = self.identity(x)
        x = self.core_layers(x)
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
    x = torch.randn(1, 3, 224, 224)
    t = ResNet(in_channels = 3, num_classes = 10, mode = 150)
    print(x.shape)
    print(t(x).shape)
