import torch

from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from src.client import client_fn
from src.server import make_cifar_server
from task import test
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

N_CLIENTS = 10
N_ROUNDS = 10

if __name__ == '__main__':
    # TODO Share this devie across everywhere...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    server, model = make_cifar_server(n_clients=N_CLIENTS)
    start_simulation(
        client_fn=client_fn,
        num_clients=N_CLIENTS,
        server=server,
        config=ServerConfig(num_rounds=N_ROUNDS))

    # TODO Somehow reuse this for all the clients
    partitioner = IidPartitioner(num_partitions=N_CLIENTS)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    dataset = fds.load_split("test").with_transform(apply_transforms)
    model.to(device)
    loss, accuracy = test(model, DataLoader(dataset), device)
    print(f'THE ACCURACY IS {accuracy}')