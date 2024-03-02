#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from GazeGenesis.ComputerVision.Models.MLP import MLP
from GazeGenesis.Utility.device import get_device_name

class User:
    def __init__(self, input_size = 784, num_classes = 10, learning_rate = 1e-3, dataset = "MNIST"):
        print("USER: MLP")
        print(f"DEVICE: {get_device_name()}")

        self.device = torch.device(get_device_name())
        self.model = MLP(num_classes = num_classes, input_size = input_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        self.dataset = dataset

    def train(self, epochs = 10):
        for epoch in range(epochs):
            pass




if __name__ == "__main__":
    user = User(input_size = 784, num_classes = 10, learning_rate = 1e-3)

    user.train(epochs = 2)