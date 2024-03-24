#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d): # features_d is the number of channels while we go deeper in the network.
        super(Discriminator, self).__init__()
        print("MODEL: Discriminator")
        
        self.layers = nn.Sequential(
            # Input: N * in_channels * 64 * 64
            nn.Conv2d(in_channels, features_d, kernel_size = 4, stride = 2, padding = 1), # 32*32
            nn.LeakyReLU(0.2),
            self.block_(in_channels = features_d * 1, out_channels = features_d  * 2, kernel_size = 4, stride = 2, padding = 1), # 16*16
            self.block_(in_channels = features_d * 2, out_channels = features_d  * 4, kernel_size = 4, stride = 2, padding = 1), # 8*8
            self.block_(in_channels = features_d * 4, out_channels = features_d  * 8, kernel_size = 4, stride = 2, padding = 1), # 4*4
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0) # 1*1
        )

        self.initialize_weights(self.layers)

        
    def block_(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),   
            )
    
    def forward(self, x):
        return self.layers(x)
    
    # we need to initialize the weights with mean=0 and std=0.02
    def initialize_weights(self, model):
        for m in  model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    


class Generator(nn.Module):
    def __init__(self, noise_dim, out_channels, features_g): # features_g is the number of channels while we go deeper in the network.
        super(Generator, self).__init__()
        print("MODEL: Generator")

        self.layers = nn.Sequential(
             # Input: N * noise_dim * 1 * 1
            self.block_(in_channels = noise_dim , out_channels = features_g  * 16, kernel_size = 4, stride = 1, padding = 0), # N * features_g * 16 * 4 * 4
            self.block_(in_channels = features_g * 16 , out_channels = features_g  * 8, kernel_size = 4, stride = 2, padding = 1), # N * features_g * 16 * 8 * 8
            self.block_(in_channels = features_g * 8 , out_channels = features_g  * 4, kernel_size = 4, stride = 2, padding = 1), # N * features_g * 16 * 16 * 16
            self.block_(in_channels = features_g * 4 , out_channels = features_g  * 2, kernel_size = 4, stride = 2, padding = 1), # N * features_g * 16 * 32 * 32
            nn.ConvTranspose2d(features_g  * 2, out_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh() # [-1, 1]
        )

        self.initialize_weights(self.layers)

    def block_(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),   
            )

    def forward(self, x):
        return self.layers(x)


    # we need to initialize the weights with mean=0 and std=0.02
    def initialize_weights(self, model):
        for m in  model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

# def test():
#     N, in_channels, H, W = 8, 3, 64, 64
#     noise_dim = 100
#     x = torch.randn((N, in_channels, H, W))
#     dicriminator = Discriminator(in_channels=in_channels, features_d = 8 )
#     # dicriminator.initialize_weights(dicriminator)
#     assert dicriminator(x).shape == (N, 1, 1, 1)

#     generator = Generator(noise_dim = noise_dim, out_channels = in_channels, features_g = 8)
#     # generator.initialize_weights(generator)
#     z = torch.randn((N, noise_dim, 1, 1))
#     assert generator(z).shape == (N, in_channels, H, W)
#     print("Success")


# test()