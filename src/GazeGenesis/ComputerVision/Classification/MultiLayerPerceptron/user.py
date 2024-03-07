#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from GazeGenesis.Utility.device import get_device_name
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER
from GazeGenesis.ComputerVision.Classification.MultiLayerPerceptron.model import MLP

from rich.progress import track

class User:
    def __init__(self, input_size = 784, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64):
        print("USER: MLP")
        print(f"DEVICE: {get_device_name()}")

        self.device = torch.device(get_device_name())
        self.model = MLP(num_classes = num_classes, input_size = input_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        self.dataset = LOADER(validation_ratio = 0.3, train_batch_size = train_batch_size, test_batch_size = test_batch_size)

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

                    # Get the inputs to correct shape
                    inputs = inputs.reshape(inputs.shape[0], -1) # -1 flattens the other dimensions together

                    
                    # Forward path
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    ls.append(loss)

                    # Backward prop
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update weights (gradient descent)
                    self.optimizer.step()

                train_accuracy = self.evaluate(self.dataset.train_loader, 'train')
                valid_accuracy = self.evaluate(self.dataset.valid_loader, 'validation')

                print(f"EPOCH: {epoch+1:0{digits}d}/{epochs}, LOSS: {torch.tensor(ls).mean():.4f}, TRAIN_ACC: {train_accuracy:.4f}, VAL_ACC: {valid_accuracy:.4f}")
        else:
            raise Exception("Dataset is None.")

    def evaluate(self, loader, name):
        if self.dataset is not None:
            self.model.eval() # This is evaluation mode

            with torch.no_grad():
                correct_predictions = total_samples = 0

                for batch_idx, (inputs, targets) in enumerate(track(loader, description=f"[EVALUATE]")):

                    # Send to device
                    inputs = inputs.to(device = self.device)
                    targets = targets.to(device = self.device)

                    # Get the inputs to correct shape
                    inputs = inputs.reshape(inputs.shape[0], -1) # -1 flattens the other dimensions together

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
        test_accuracy = self.evaluate(self.dataset.test_loader, 'validation')
        print(f"TEST_ACC: {test_accuracy:.4f}")

    def save(self):
        pass


