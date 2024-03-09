# LeNEt

**Script**
```python

from GazeGenesis.ComputerVision.Classification.LeNet.user import User

if __name__ == "__main__":
    user = User(in_channels = 1, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64)

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
```
**Train**
```zsh
USER: LeNet
DEVICE: mps
MODEL: LeNet
Dataset: CIFAR10

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:06
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 1/2, LOSS: 1.8517, TRAIN_ACC: 0.3905, VAL_ACC: 0.3909

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
EPOCH: 2/2, LOSS: 1.5652, TRAIN_ACC: 0.4457, VAL_ACC: 0.4323
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
TEST_ACC: 0.9722
```

Thanks to Aladdin Persson.