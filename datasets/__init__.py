from .mnist_dataset import MNIST_Dataset
from .cifar10_dataset import CIFAR10_Dataset
from .celeba_dataset import CelebA_Dataset

def load_dataset(dataset="MNIST",
                 mode='train'):
    if dataset == "MNIST":
        return MNIST_Dataset(mode=mode)
    elif dataset == "CIFAR10":
        return CIFAR10_Dataset(mode=mode)
    elif dataset == 'CelebA':
        return CelebA_Dataset(mode=mode)