#!/usr/bin/python3

import torch, os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LOADER:
    def __init__(self, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64):
        if not (validation_ratio < 1 and validation_ratio >= 0):
            raise RuntimeError('Validation ratio should be >= 0 and < 1.')

        current_directory = os.path.dirname(os.path.abspath(__file__))

        self.train_dataset = datasets.MNIST(root = current_directory+"/DataSet/", train = True, transform = transforms.ToTensor(), download = True)
        self.test_dataset = datasets.MNIST(root = current_directory+"/DataSet/", train = False, transform = transforms.ToTensor(), download = True)
        
        total_train_count = len(self.train_dataset)
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.train_dataset, [int((1 - validation_ratio) * total_train_count), int(validation_ratio * total_train_count)])
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = train_batch_size, shuffle = True)
        self.valid_loader = DataLoader(dataset = self.validation_dataset, batch_size = test_batch_size, shuffle = True)
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = test_batch_size, shuffle = True)

        



        
        

