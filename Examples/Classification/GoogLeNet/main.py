#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.GoogLeNet.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.CIFAR10 import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()