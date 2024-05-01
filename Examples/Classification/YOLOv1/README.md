# LeNEt

**Script**
```python

from GazeGenesis.ComputerVision.Classification.YOLOv1.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.CIFAR10 import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 20, test_batch_size = 64, transform=transform)

    user = User(in_channels = 3, num_classes = 10, num_boxes = 2, learning_rate = 1e-3, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
in_channels = 3
num_classes = 10
num_boxes = 2
learning_rate = 0.001
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```

**Train**
```zsh
Dataset: CIFAR10
USER: YOLOv1
DEVICE: mps
MODEL: YOLOv1

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:25:18
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:07:12
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:06
EPOCH: 1/2, LOSS: 1.9825, TRAIN_ACC: 0.4157, VAL_ACC: 0.4097
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
TEST_ACC: 0.4352
```

Thanks to [Aladdin Persson](https://github.com/aladdinpersson).