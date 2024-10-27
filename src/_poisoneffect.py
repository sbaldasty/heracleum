from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from src.client import client_fn
from src.server import make_cifar_server
from task import get_device
from task import test
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

N_CLIENTS = 10
N_ROUNDS = 2

if __name__ == '__main__':
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
    print(f'THE DATASET IS {dataset}')
    loss, accuracy = test(model.to(get_device()), DataLoader(dataset))
    print(f'THE ACCURACY IS {accuracy}')