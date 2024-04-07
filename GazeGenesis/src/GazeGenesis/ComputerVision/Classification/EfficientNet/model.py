#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name



base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2), # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        super(CNNBlock, self).__init__()

        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding = padding, groups = groups, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.core_layers(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),
            nn.Sigmod()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction = 4, survaival_prob = 0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survaival_prob
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio
        self.expand = (in_channels != hidden_dim)
        reduced_dim = int(in_channels // reduction)

        if self.expand:
            self.expand_conv = CNNBlock(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        
        self.conv = nn.Sequential(
            CNNBlock(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim),
            SqueezeExcitation(in_channels=hidden_dim, reduced_dim=reduced_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device) < self.survival_prob 
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        
        



class EfficientNet(nn.Module):
    def __init__(self, mode, num_classes = 10):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(mode)

    def calculate_factors(self, mode, alpha=1.2, beta = 1.1):
        phi, res, drop_rate = phi_values[mode]
        