#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.YOLOv1.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.PASCAL import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
        ])
    loader = LOADER(train_batch_size = 20, mode = "2007", test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, num_boxes = 2, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()