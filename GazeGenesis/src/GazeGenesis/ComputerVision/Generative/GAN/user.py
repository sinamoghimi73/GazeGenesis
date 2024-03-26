#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from GazeGenesis.Utility.device import get_device_name
from GazeGenesis.ComputerVision.Generative.GAN.model import Discriminator, Generator
import torchvision
from torch.utils.tensorboard import SummaryWriter

from rich.progress import track

class User:
    def __init__(self, input_dim = 784, noise_dim = 64, learning_rate = 3e-4, loader = None, summary_writer_address = None):
        print("USER: GAN")
        print(f"DEVICE: {get_device_name()}")

        self.device = torch.device(get_device_name())
        self.discriminator = Discriminator(input_dim = input_dim).to(self.device)
        self.generator = Generator(noise_dim = noise_dim, output_dim = input_dim).to(self.device)
        self.fixed_noise = torch.randn((loader.train_batch_size, noise_dim)).to(self.device)
        self.noise_dim = noise_dim

        self.criterion = nn.BCELoss()
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr = learning_rate)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr = learning_rate)

        # Tensorboard settings
        self.summary_writer_address = summary_writer_address
        if self.summary_writer_address:
            self.writer_fake = SummaryWriter(self.summary_writer_address + "fake")
            self.writer_real = SummaryWriter(self.summary_writer_address + "real")
            self.step = 0

        if loader is None:
            raise ValueError("You should pass a 'loader' to the user.")
        self.dataset = loader

    def train(self, epochs = 10):
        if self.dataset is not None:
            digits = int(torch.log10(torch.tensor(epochs))) + 1
            for epoch in range(epochs):
                print()
                ls = []
                for batch_idx, (real, _) in enumerate(track(self.dataset.train_loader, description=f"[TRAIN] {epoch+1:0{digits}d}/{epochs}")):

                    # Send to device
                    real = real.to(device = self.device)

                    # Get the inputs to correct shape
                    real = real.reshape(real.shape[0], -1) # -1 flattens the other dimensions together
                    batch_size = real.shape[0]

                    # Train the discriminator -> max log(D(real)) + log(1 - D(G(noise)))
                    noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                    fake = self.generator(noise*0.5)

                    discriminator_on_real = self.discriminator(real).view(-1)
                    loss_discriminator_on_real = self.criterion(discriminator_on_real, torch.ones_like(discriminator_on_real))

                    discriminator_on_fake = self.discriminator(fake).view(-1)
                    loss_discriminator_on_fake = self.criterion(discriminator_on_fake, torch.zeros_like(discriminator_on_fake))

                    total_loss_discriminator = (loss_discriminator_on_real + loss_discriminator_on_fake) / 2

                    self.discriminator_optimizer.zero_grad()
                    total_loss_discriminator.backward(retain_graph = True) # we need the retain_graph = True to keep the gradients for the later use in Generator
                    self.discriminator_optimizer.step()

                    # Train the generator -> min log(1-D(G(noise))) <-> leads to saturating gradients
                    # instead we will have -> max log(D(G(noise)))

                    output = self.discriminator(fake).view(-1)
                    loss_generator = self.criterion(output, torch.ones_like(output))

                    self.generator_optimizer.zero_grad()
                    loss_generator.backward()
                    self.generator_optimizer.step()

                    if self.summary_writer_address:
                        if 0 == batch_idx:
                            print(f"EPOCH: {epoch+1:0{digits}d}/{epochs}, D_LOSS: {total_loss_discriminator:.4f}, G_LOSS: {loss_generator:.4f}")

                            with torch.no_grad():
                                fake = self.generator(self.fixed_noise*0.5).reshape(-1, 1, 28, 28)
                                data = real.reshape(-1, 1, 28,28)
                                img_grid_fake = torchvision.utils.make_grid(fake.detach(), normalize=True)
                                img_grid_real = torchvision.utils.make_grid(data.detach(), normalize=True)
                                self.writer_fake.add_image("Fake Images", img_grid_fake, global_step = self.step)
                                self.writer_real.add_image("Real Images", img_grid_real, global_step = self.step)
                                self.step += 1
                                self.writer_fake.flush()
                                self.writer_real.flush()


                # train_accuracy = self.evaluate(self.dataset.train_loader, 'evaluate: train')
                # valid_accuracy = self.evaluate(self.dataset.valid_loader, 'evaluate: validation')

                # print(f"EPOCH: {epoch+1:0{digits}d}/{epochs}, LOSS: {torch.tensor(ls).mean():.4f}, TRAIN_ACC: {train_accuracy:.4f}, VAL_ACC: {valid_accuracy:.4f}")
        if self.summary_writer_address:
            self.writer_fake.close()
            self.writer_real.close()
        else:
            raise Exception("Dataset is None.")

    def save(self):
        pass
