# Multi Layer Perceptron

**Script**
```python

from GazeGenesis.ComputerVision.MNIST.MultiLayerPerceptron.user import User

if __name__ == "__main__":
    user = User(input_size = 28*28, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64)

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
```
**Train**
```zsh
USER: MLP
DEVICE: mps
MODEL: MLP

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
EPOCH: 1/2, LOSS: 0.4748, TRAIN_ACC: 0.9243, VAL_ACC: 0.9176

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
[EVALUATE: TRAIN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
[EVALUATE: VALIDATION] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
EPOCH: 2/2, LOSS: 0.2389, TRAIN_ACC: 0.9406, VAL_ACC: 0.9315
```

**Test**
```zsh
[TEST] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
TEST_ACC: 0.9380
```

Thanks to Aladdin Persson.