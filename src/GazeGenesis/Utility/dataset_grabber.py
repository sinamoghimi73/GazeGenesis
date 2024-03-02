#!/usr/bin/python3

import torch
import os 
# from GazeGenesis.Datasets.mnist import MNIST


def grab(name: str = "MNIST", has_train_data = True, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    files = os.listdir(parent_directory + "/Datasets")
    files = [f.split(".")[0] for f in files if f.endswith(".py") and f != "__init__.py"]

    if name not in files:
        raise NameError(f'Dataset \"{name}\" not found. Available datasets are {files}.')
    else:
        if "MNIST" == name:
            from GazeGenesis.Datasets.MNIST import MNIST
            return MNIST(has_train_data = has_train_data, validation_ratio = validation_ratio, train_batch_size = train_batch_size, test_batch_size = test_batch_size)




if __name__ == "__main__":
    dataset = grab(name = "MNIST", has_train_data = True, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64)
