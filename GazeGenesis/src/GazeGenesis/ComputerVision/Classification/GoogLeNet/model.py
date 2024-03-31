#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels), # this isn't in the original paper.
            nn.ReLU() 
        )
    
    def forward(self, x):
        return self.layers(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(InceptionBlock, self).__init__()
        self.branch_1 = ConvBlock(in_channels = in_channels, out_channels=out_1x1, kernel_size=1)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels = in_channels, out_channels=red_3x3, kernel_size=1),
            ConvBlock(in_channels = red_3x3, out_channels=out_3x3, kernel_size=3, stride = 1, padding = 1)
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels = in_channels, out_channels=red_5x5, kernel_size=1),
            ConvBlock(in_channels = red_5x5, out_channels=out_5x5, kernel_size=5, stride = 1, padding = 2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels = in_channels, out_channels=out_1x1_pool, kernel_size=1)
        )

    def forward(self, x):
        # x: (batch_size, filters, img_size, img_size)  
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], dim = 1)




class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes = 10):
        super(GoogLeNet, self).__init__()
        print("MODEL: GoogLeNet")

        
        self.head_layers = nn.Sequential(
            ConvBlock(in_channels = in_channels, out_channels=64, kernel_size=7, stride = 2, padding = 3), # conv_1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # maxpool_1
            ConvBlock(in_channels = 64, out_channels=192, kernel_size=3, stride = 1, padding = 1), # conv_2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # maxpool_2
        )

        self.core_layers = nn.Sequential(
            InceptionBlock(in_channels = 192, out_1x1 = 64, red_3x3 = 96, out_3x3 = 128, red_5x5 = 16, out_5x5 = 32, out_1x1_pool = 32), # Inception_3a
            InceptionBlock(in_channels = 256, out_1x1 = 128, red_3x3 = 128, out_3x3 = 192, red_5x5 = 32, out_5x5 = 96, out_1x1_pool = 64), # Inception_3b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # maxpool_3

            InceptionBlock(in_channels = 480, out_1x1 = 192, red_3x3 = 96, out_3x3 = 208, red_5x5 = 16, out_5x5 = 48, out_1x1_pool = 64), # inception_4a
            InceptionBlock(in_channels = 512, out_1x1 = 160, red_3x3 = 112, out_3x3 = 224, red_5x5 = 24, out_5x5 = 64, out_1x1_pool = 64), # inception_4b
            InceptionBlock(in_channels = 512, out_1x1 = 128, red_3x3 = 128, out_3x3 = 256, red_5x5 = 24, out_5x5 = 64, out_1x1_pool = 64), # inception_4c
            InceptionBlock(in_channels = 512, out_1x1 = 112, red_3x3 = 144, out_3x3 = 288, red_5x5 = 32, out_5x5 = 64, out_1x1_pool = 64), # inception_4d
            InceptionBlock(in_channels = 528, out_1x1 = 256, red_3x3 = 160, out_3x3 = 320, red_5x5 = 32, out_5x5 = 128, out_1x1_pool = 128), # inception_4e
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # maxpool_4

            InceptionBlock(in_channels = 832, out_1x1 = 256, red_3x3 = 160, out_3x3 = 320, red_5x5 = 32, out_5x5 = 128, out_1x1_pool = 128), # inception_5a
            InceptionBlock(in_channels = 832, out_1x1 = 384, red_3x3 = 192, out_3x3 = 384, red_5x5 = 48, out_5x5 = 128, out_1x1_pool = 128), # inception_5b
            nn.AvgPool2d(kernel_size=7, stride=1), # avg_pool
            nn.Dropout(p=0.4) # dropout
        )

        self.tail_layers = nn.Sequential(
            nn.Linear(1024, num_classes), # linear
            nn.Softmax(dim=-1) # Softmax
        )


    def forward(self, x):
        x = self.head_layers(x)
        x = self.core_layers(x)
        x = x.flatten(1)
        x = self.tail_layers(x)
        return x



if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    gn = GoogLeNet(in_channels = 3)

    print(gn(x).shape)