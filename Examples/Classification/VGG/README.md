# VGG

**Script**
```python

from GazeGenesis.ComputerVision.Classification.VGG.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.CIFAR10 import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, learning_rate = 1e-3, mode = 11, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
in_channels = 3
num_classes = 10
learning_rate = 0.001
mode = 11
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```
**Train**
```zsh
Dataset: CIFAR10
USER: VGG-11
DEVICE: cuda
MODEL: VGG-11

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:04:06
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:57
EPOCH: 1/2, LOSS: 1.9281, TRAIN_ACC: 0.4064, VAL_ACC: 0.3985

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:04:05
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:57
EPOCH: 2/2, LOSS: 1.4066, TRAIN_ACC: 0.5220, VAL_ACC: 0.4929
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:35
TEST_ACC: 0.5094
```

Thanks to Aladdin Persson.