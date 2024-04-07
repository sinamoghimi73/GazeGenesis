# GoogLeNet

**Script**
```python

#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.GoogLeNet.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.CIFAR10 import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
in_channels = 3
num_classes = 10
learning_rate = 0.001
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```

**Train**
```zsh
Dataset: CIFAR10
USER: GoogLeNet
DEVICE: mps
MODEL: GoogLeNet

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:04:33
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:12
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:31
EPOCH: 1/2, LOSS: 2.1385, TRAIN_ACC: 0.3029, VAL_ACC: 0.3013

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:04:22
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:17
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:34
EPOCH: 2/2, LOSS: 2.0772, TRAIN_ACC: 0.3959, VAL_ACC: 0.3941
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:23
TEST_ACC: 0.4000
```

Thanks to [Aladdin Persson](https://github.com/aladdinpersson)