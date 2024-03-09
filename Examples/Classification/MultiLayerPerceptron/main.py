#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.MultiLayerPerceptron.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(input_size = 28*28, num_classes = 10, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()