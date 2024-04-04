# Vision Transformer

**Script**
```python

#!/usr/bin/python3
from GazeGenesis.ComputerVision.Classification.VisionTransformer.user import User
import torchvision.transforms as transforms
from GazeGenesis.ComputerVision.Datasets.MNIST import LOADER

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])
    loader = LOADER(validation_ratio=0.3, train_batch_size = 64, test_batch_size = 64, transform=transform)

    user = User(in_channels = 1, num_classes = 10, patch_size = 4, learning_rate = 1e-3, embedding_dim = 8, model_depth = 1, attention_heads = 4, loader=loader)

    user.train(epochs = 2)
    user.test()
```
**Parameters**
```python
in_channels = 1
num_classes = 10
patch_size = 4
learning_rate = 0.001
embedding_dim = 8
model_depth = 1
attention_heads = 4
train_batch_size = 64
test_batch_size = 64
loader = <THE DATASET OF YOUR CHOICE WITH APPROPRIATE TRANSFORM>
```
**Train**
```zsh
Dataset: MNIST
USER: ViT-1
DEVICE: mps
Model: ViT-1
[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:10
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 1/2, LOSS: 2.0214, TRAIN_ACC: 0.3008, VAL_ACC: 0.3040

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:09
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 2/2, LOSS: 1.8785, TRAIN_ACC: 0.3399, VAL_ACC: 0.3372
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
TEST_ACC: 0.3419
```

Thanks to mildlyoverfitted.