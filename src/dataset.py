from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from typing import Tuple


DATA_DIR = './data'
BATCH_SIZE = 32


def cifar_dataloaders(n_clients: int) -> Tuple[list[DataLoader], DataLoader]:
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    train_set = CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    ideal_chunk_size = len(train_set) // n_clients
    chunk_sizes = [ideal_chunk_size] * n_clients
    leftovers = len(train_set) % ideal_chunk_size
    if leftovers:
        chunk_sizes[0] += leftovers
    chunks = random_split(train_set, chunk_sizes)
    chunk_loaders = [DataLoader(x, batch_size=BATCH_SIZE) for x in chunks]
    return chunk_loaders, test_loader
