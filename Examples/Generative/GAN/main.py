#!/usr/bin/python3
from GazeGenesis.ComputerVision.Generative.GAN.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER
import math, os

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081))
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 16, test_batch_size = 64, transform = transform)

    current_adress = os.path.dirname(__file__)
    user = User(input_dim = math.prod(loader.dimension), noise_dim = 64, learning_rate = 3e-4, loader=loader, summary_writer_address = current_adress + "/runs/GAN/")

    user.train(epochs = 10)
