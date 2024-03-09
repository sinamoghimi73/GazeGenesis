#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.LeNet.user import User

if __name__ == "__main__":
    user = User(in_channels = 3, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64)

    user.train(epochs = 2)
    user.test()