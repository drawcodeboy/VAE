from .mnist_dataset import MNIST_Dataset

def load_dataset(dataset="MNIST",
                 mode='train'):
    if dataset == "MNIST":
        return MNIST_Dataset(mode=mode)
    