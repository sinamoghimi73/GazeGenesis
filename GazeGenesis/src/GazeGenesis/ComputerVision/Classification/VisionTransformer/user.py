#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from GazeGenesis.Utility.device import get_device_name
from GazeGenesis.ComputerVision.Classification.VisionTransformer.model import VisionTransformer

from rich.progress import track


class User:
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 4,
        learning_rate: float = 1e-3,
        embedding_dim: int = 8,
        model_depth: int = 1,
        attention_heads: int = 4,
        loader = None
    ):
        print(f"USER: ViT-{model_depth}")
        print(f"DEVICE: {get_device_name()}")

        self.device = torch.device(get_device_name())
        if loader is None: 
            raise ValueError("You should pass a 'loader' to the user.")
        self.dataset = loader

        self.model = VisionTransformer(
            img_size = self.dataset.dimension[0],
            patch_size= patch_size,
            in_channels = in_channels,
            n_classes = num_classes,
            embed_dim = embedding_dim,
            depth = model_depth,
            n_heads = attention_heads,
            mlp_ratio= 4.,
            qkv_bias= True,
            p = 0.1,
            attn_p = 0.1
            ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas = (0.9, 0.98), eps = 1e-9)
        

    def train(self, epochs=10):
        if self.dataset is not None:
            digits = int(torch.log10(torch.tensor(epochs))) + 1
            for epoch in range(epochs):
                print()
                ls = []
                for batch_idx, (inputs, targets) in enumerate(
                    track(
                        self.dataset.train_loader,
                        description=f"[TRAIN] {epoch+1:0{digits}d}/{epochs}",
                    )
                ):

                    # Send to device
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Forward path
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    ls.append(loss)

                    # Backward prop
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update weights (gradient descent)
                    self.optimizer.step()

                train_accuracy = self.evaluate(
                    self.dataset.train_loader, "evaluate: train"
                )
                valid_accuracy = self.evaluate(
                    self.dataset.valid_loader, "evaluate: validation"
                )

                print(
                    f"EPOCH: {epoch+1:0{digits}d}/{epochs}, LOSS: {torch.tensor(ls).mean():.4f}, TRAIN_ACC: {train_accuracy:.4f}, VAL_ACC: {valid_accuracy:.4f}"
                )
        else:
            raise Exception("Dataset is None.")

    def evaluate(self, loader, name):
        if self.dataset is not None:
            self.model.eval()  # This is evaluation mode

            with torch.no_grad():
                correct_predictions = total_samples = 0

                for batch_idx, (inputs, targets) in enumerate(
                    track(loader, description=f"[{name.upper()}]")
                ):

                    # Send to device
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Forward path
                    predictions = self.model(inputs)
                    _, predicted = torch.max(predictions, 1)

                    correct_predictions += (predicted == targets).sum().item()
                    total_samples += len(targets)

                accuracy = correct_predictions / total_samples

            self.model.train()
            return accuracy
        else:
            raise Exception("Dataset is None.")

    def test(self):
        test_accuracy = self.evaluate(self.dataset.test_loader, "TEST")
        print(f"TEST_ACC: {test_accuracy:.4f}")

    def save(self):
        pass
