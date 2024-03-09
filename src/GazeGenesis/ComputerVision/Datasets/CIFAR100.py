#!/usr/bin/python3

import torch, os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CustomTransformer:
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop((18,18)),
            # transforms.RandomGrayscale(p = 0.2),
            transforms.Resize((224,224)),
            # transforms.RandomRotation(degrees = 30),
            # transforms.ColorJitter(brightness = 0.5),
            # transforms.RandomHorizontalFlip(p = 0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.1], std = [1.0]), # the array size here is variable with channel count. For mean=0 and std=1, the data remains unchanged and not normalized. We should find those values first manually.
        ])

class LOADER:
    def __init__(self, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64):
        if not (validation_ratio < 1 and validation_ratio >= 0):
            raise RuntimeError('Validation ratio should be >= 0 and < 1.')
        
        print("Dataset: CIFAR100")
        self.dimension = (32,32)
        self.custom_transformer = CustomTransformer()

        current_directory = os.path.dirname(os.path.abspath(__file__))

        self.train_dataset = datasets.CIFAR100(root = current_directory+"/CIFAR100", train = True, transform = self.custom_transformer.transform, download = True)
        self.test_dataset = datasets.CIFAR100(root = current_directory+"/CIFAR100", train = False, transform = self.custom_transformer.transform, download = True)
        
        total_train_count = len(self.train_dataset)
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.train_dataset, [int((1 - validation_ratio) * total_train_count), int(validation_ratio * total_train_count)])
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = train_batch_size, shuffle = True)
        self.valid_loader = DataLoader(dataset = self.validation_dataset, batch_size = test_batch_size, shuffle = True)
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = test_batch_size, shuffle = True)

        



        
        

