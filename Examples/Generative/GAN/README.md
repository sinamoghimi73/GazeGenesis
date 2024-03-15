# Generative Adversarial Network

**Script**
```python

from GazeGenesis.ComputerVision.Generative.GAN.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER
import math, os

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081))
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 16, test_batch_size = 64, transform = transform)

    current_adress = os.path.dirname(__file__)
    user = User(input_dim = math.prod(loader.dimension), noise_dim = 64, learning_rate = 3e-4, loader=loader, summary_writer_address = current_adress + "/runs/GAN/")

    user.train(epochs = 10)
```
**Parameters**
```python
input_dim = 784
noise_dim = 64
learning_rate = 0.0003
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
summary_writer_address = current_adress + "/runs/GAN/"
```
**Train**
```zsh
Dataset: MNIST
USER: GAN
DEVICE: mps
MODEL: Discriminator
MODEL: Generator

EPOCH: 01/10, D_LOSS: 0.6659, G_LOSS: 0.6969
[TRAIN] 01/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21

EPOCH: 02/10, D_LOSS: 0.0754, G_LOSS: 2.8331
[TRAIN] 02/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21

EPOCH: 03/10, D_LOSS: 0.0219, G_LOSS: 3.8983
[TRAIN] 03/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20

EPOCH: 04/10, D_LOSS: 0.0461, G_LOSS: 4.7629
[TRAIN] 04/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21

EPOCH: 05/10, D_LOSS: 0.0180, G_LOSS: 4.8466
[TRAIN] 05/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21

EPOCH: 06/10, D_LOSS: 0.0044, G_LOSS: 5.8503
[TRAIN] 06/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20

EPOCH: 07/10, D_LOSS: 0.0016, G_LOSS: 6.8584
[TRAIN] 07/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21

EPOCH: 08/10, D_LOSS: 0.0067, G_LOSS: 5.7062
[TRAIN] 08/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20

EPOCH: 09/10, D_LOSS: 0.0250, G_LOSS: 4.2842
[TRAIN] 09/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20

EPOCH: 10/10, D_LOSS: 0.0093, G_LOSS: 7.5974
[TRAIN] 10/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:21
```


Thanks to Aladdin Persson.