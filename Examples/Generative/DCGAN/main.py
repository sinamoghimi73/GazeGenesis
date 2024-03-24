#!/usr/bin/python3
from GazeGenesis.ComputerVision.Generative.DCGAN.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER
import math, os

if __name__ == "__main__":
    img_channels = 1
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.Normalize(
                [0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]
                )
        ])
    loader = LOADER(validation_ratio=0.01, train_batch_size = 32, test_batch_size = 64, transform = transform)

    current_adress = os.path.dirname(__file__)
    user = User(in_channels = img_channels, noise_dim = 100, features = 64, learning_rate = 2e-4, loader=loader, summary_writer_address = current_adress + "/runs/DCGAN/")

    user.train(epochs = 5)
