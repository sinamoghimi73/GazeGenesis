#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.VisionTransformer.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081)),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 1, num_classes = 10, patch_size = 4, learning_rate = 1e-3, embedding_dim = 8, model_depth = 1, attention_heads = 4, loader=loader)

    user.train(epochs = 2)
    user.test()
