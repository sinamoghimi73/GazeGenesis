#!/usr/bin/python3

import torch, os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.imagenet import parse_devkit_archive, parse_train_archive, parse_val_archive


class LOADER:
    def __init__(self, validation_ratio = 0.3, train_batch_size = 64, test_batch_size = 64, transform = None):
        if not (validation_ratio < 1 and validation_ratio >= 0):
            raise RuntimeError('Validation ratio should be >= 0 and < 1.')
        
        print("Dataset: ImageNet")
        self.dimension = (224,224)
        if transform is not None:
            self.custom_transformer = transform
        else:
            self.custom_transformer = transforms.ToTensor()

        current_directory = os.path.dirname(os.path.abspath(__file__))
        os.system(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate -P {current_directory}/ImageNet")
        os.system(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate -P {current_directory}/ImageNet")
        os.system(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate -P {current_directory}/ImageNet")

        self.train_dataset = datasets.ImageNet(root = current_directory+"/ImageNet", split = "train", transform = self.custom_transformer, download = False)
        self.validation_dataset = datasets.ImageNet(root = current_directory+"/ImageNet", split = "val", transform = self.custom_transformer, download = False)
        
        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = train_batch_size, shuffle = True)
        self.valid_loader = DataLoader(dataset = self.validation_dataset, batch_size = test_batch_size, shuffle = True)

        



        
        

