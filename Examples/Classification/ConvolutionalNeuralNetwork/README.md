# Convolutional Neural Network

**Script**
```python

from GazeGenesis.ComputerVision.Classification.ConvolutionalNeuralNetwork.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform = transform)

    user = User(in_channels = 1, num_classes = 10, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
in_channels = 1
num_classes = 10
learning_rate = 0.001
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```
**Train**
```zsh
Dataset: MNIST
USER: CNN
DEVICE: cuda
MODEL: CNN

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:06
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 1/2, LOSS: 0.5848, TRAIN_ACC: 0.9257, VAL_ACC: 0.9186

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:06
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
EPOCH: 2/2, LOSS: 0.2045, TRAIN_ACC: 0.9451, VAL_ACC: 0.9403
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
TEST_ACC: 0.9557
```

Thanks to [Aladdin Persson](https://github.com/aladdinpersson).