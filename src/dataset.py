from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from typing import Tuple


DATA_DIR = './data'
BATCH_SIZE = 32

test_set = None
cifar_train_loaders = None
cifar_test_loader = None

def cifar_test_set() -> CIFAR10:
    global test_set
    if test_set is None:
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    return test_set

def cifar_dataloaders(n_clients: int) -> Tuple[list[DataLoader], DataLoader]:
    global test_set
    global cifar_train_loaders
    global cifar_test_loader
    if cifar_test_loader is None:
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
        cifar_test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        train_set = CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        ideal_chunk_size = len(train_set) // n_clients
        chunk_sizes = [ideal_chunk_size] * n_clients
        leftovers = len(train_set) % ideal_chunk_size
        if leftovers:
            chunk_sizes[0] += leftovers
        chunks = random_split(train_set, chunk_sizes)
        cifar_train_loaders = [DataLoader(x, batch_size=BATCH_SIZE, shuffle=False) for x in chunks]
    return cifar_train_loaders, cifar_test_loader