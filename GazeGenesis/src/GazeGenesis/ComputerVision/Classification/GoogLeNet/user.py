#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from GazeGenesis.Utility.device import get_device_name
from GazeGenesis.ComputerVision.Classification.GoogLeNet.model import GoogLeNet

from rich.progress import track

class User:
    def __init__(self, in_channels: int = 1, num_classes: int = 10, learning_rate: float = 1e-3, loader = None):
        print("USER: GoogLeNet")
        print(f"DEVICE: {get_device_name()}")

        self.device = torch.device(get_device_name())
        self.model = GoogLeNet(in_channels = in_channels, num_classes = num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        if loader is None: 
            raise ValueError("You should pass a 'loader' to the user.")
        self.dataset = loader

    def train(self, epochs = 10):
        if self.dataset is not None:
            digits = int(torch.log10(torch.tensor(epochs))) + 1
            for epoch in range(epochs):
                print()
                ls = []
                for batch_idx, (inputs, targets) in enumerate(track(self.dataset.train_loader, description=f"[TRAIN] {epoch+1:0{digits}d}/{epochs}")):

                    # Send to device
                    inputs = inputs.to(device = self.device)
                    targets = targets.to(device = self.device)

                    # Forward path
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    ls.append(loss)

                    # Backward prop
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update weights (gradient descent)
                    self.optimizer.step()

                train_accuracy = self.evaluate(self.dataset.train_loader, 'evaluate: train')
                valid_accuracy = self.evaluate(self.dataset.valid_loader, 'evaluate: validation')

                print(f"EPOCH: {epoch+1:0{digits}d}/{epochs}, LOSS: {torch.tensor(ls).mean():.4f}, TRAIN_ACC: {train_accuracy:.4f}, VAL_ACC: {valid_accuracy:.4f}")
        else:
            raise Exception("Dataset is None.")

    def evaluate(self, loader, name):
        if self.dataset is not None:
            self.model.eval() # This is evaluation mode

            with torch.no_grad():
                correct_predictions = total_samples = 0

                for batch_idx, (inputs, targets) in enumerate(track(loader, description=f"[{name.upper()}]")):

                    # Send to device
                    inputs = inputs.to(device = self.device)
                    targets = targets.to(device = self.device)

                    # Forward path
                    predictions = self.model(inputs)
                    _, predicted = torch.max(predictions, 1)

                    correct_predictions +=  (predicted == targets).sum().item()
                    total_samples += len(targets)

                accuracy = correct_predictions / total_samples

            self.model.train()
            return accuracy
        else:
            raise Exception("Dataset is None.")

    def test(self):
        test_accuracy = self.evaluate(self.dataset.test_loader, 'TEST')
        print(f"TEST_ACC: {test_accuracy:.4f}")

    def save(self):
        pass
