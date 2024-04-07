# Multi Layer Perceptron

**Script**
```python

from GazeGenesis.ComputerVision.Classification.MultiLayerPerceptron.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(input_size = 28*28, num_classes = 10, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
input_size = 28*28
num_classes = 10
learning_rate = 0.001
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```
**Train**
```zsh
Dataset: MNIST
USER: MLP
DEVICE: cuda
MODEL: MLP

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:06
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 1/2, LOSS: 0.4815, TRAIN_ACC: 0.9217, VAL_ACC: 0.9157

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 2/2, LOSS: 0.2410, TRAIN_ACC: 0.9427, VAL_ACC: 0.9356
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
TEST_ACC: 0.9372
```

Thanks to [Aladdin Persson](https://github.com/aladdinpersson).