
from GazeGenesis.ComputerVision.MNIST.MultiLayerPerceptron.user import User

if __name__ == "__main__":
    user = User(input_size = 28*28, num_classes = 10, learning_rate = 1e-3, train_batch_size = 64, test_batch_size = 64)

    user.train(epochs = 2)
    user.test()