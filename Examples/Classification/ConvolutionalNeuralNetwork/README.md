# Convolutional Neural Network

**Script**
```python

from GazeGenesis.ComputerVision.MNIST.ConvolutionalNeuralNetwork.user import User

if __name__ == "__main__":
    user = User(in_channels = 1, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64)

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
```
**Train**
```zsh
USER: CNN
DEVICE: mps
MODEL: CNN

[TRAIN] 1/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
[EVALUATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
[EVALUATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
EPOCH: 1/2, LOSS: 0.6926, TRAIN_ACC: 0.9054, VAL_ACC: 0.9029

[TRAIN] 2/2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
[EVALUATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
[EVALUATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
EPOCH: 2/2, LOSS: 0.2462, TRAIN_ACC: 0.9424, VAL_ACC: 0.9389
```

**Test**
```zsh
[EVALUATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
TEST_ACC: 0.9442
```

Thanks to Aladdin Persson.