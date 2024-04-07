# ResNet

**Script**
```python

from GazeGenesis.ComputerVision.Classification.ResNet.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.CIFAR10 import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, learning_rate = 1e-3, mode = 18, loader=loader)

    user.train(epochs = 2)
    user.test()

```
**Parameters**
```python
in_channels = 3
num_classes = 10
learning_rate = 0.001
mode = 18
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```
**Train**
```zsh
Dataset: CIFAR10
USER: ResNet-18
DEVICE: mps
MODEL: ResNet-18

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:27
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:04
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:28
EPOCH: 1/2, LOSS: 1.5589, TRAIN_ACC: 0.5111, VAL_ACC: 0.5015

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:17
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:05
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:29
EPOCH: 2/2, LOSS: 1.1188, TRAIN_ACC: 0.5014, VAL_ACC: 0.4913
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:20
TEST_ACC: 0.4943
```

Thanks to [Aladdin Persson](https://github.com/aladdinpersson).