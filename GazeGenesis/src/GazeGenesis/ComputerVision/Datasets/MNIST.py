#!/usr/bin/python3

import torch, os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class LOADER:
    def __init__(self, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64, transform = None):
        if not (validation_ratio < 1 and validation_ratio >= 0):
            raise RuntimeError('Validation ratio should be >= 0 and < 1.')
        
        self.train_batch_size = train_batch_size
        
        print("Dataset: MNIST")
        self.dimension = (28,28)
        if transform is not None:
            self.custom_transformer = transform
        else:
            self.custom_transformer = transforms.ToTensor()

        current_directory = os.path.dirname(os.path.abspath(__file__))

        self.train_dataset = datasets.MNIST(root = current_directory, train = True, transform = self.custom_transformer, download = True)
        self.test_dataset = datasets.MNIST(root = current_directory, train = False, transform = self.custom_transformer, download = True)
        
        total_train_count = len(self.train_dataset)
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.train_dataset, [int((1 - validation_ratio) * total_train_count), int(validation_ratio * total_train_count)])
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = train_batch_size, shuffle = True)
        self.valid_loader = DataLoader(dataset = self.validation_dataset, batch_size = test_batch_size, shuffle = True)
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = test_batch_size, shuffle = True)

        



        
        

