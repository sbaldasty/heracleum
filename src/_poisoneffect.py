from dataclass_csv import DataclassWriter
from dataclasses import dataclass
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


OUTPUT_FILE = './out/poisoneffect.csv'
N_CLIENTS = 10
N_ROUNDS = 20
N_CORRUPT_CLIENTS_START = 0
N_CORRUPT_CLIENTS_END = 10
N_CORRUPT_CLIENTS_STEP = 1


@dataclass
class Experiment:
    n_clients: int
    n_rounds: int
    n_corrupt_clients: int
    loss: float
    accuracy: float


if __name__ == '__main__':
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

    test_dataset = fds.load_split("test").with_transform(apply_transforms)
    test_loader = DataLoader(test_dataset)

    experiments = []

    for n_corrupt_clients in range(N_CORRUPT_CLIENTS_START, N_CORRUPT_CLIENTS_END + 1, N_CORRUPT_CLIENTS_STEP):
        server, model = make_cifar_server(n_clients=N_CLIENTS)
        start_simulation(
            client_fn=client_fn,
            num_clients=N_CLIENTS,
            server=server,
            config=ServerConfig(num_rounds=N_ROUNDS))

        loss, accuracy = test(model.to(get_device()), test_loader)
        experiments.append(Experiment(
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            n_corrupt_clients=n_corrupt_clients,
            loss=loss,
            accuracy=accuracy))

    with open(OUTPUT_FILE, 'w') as file:
        DataclassWriter(file, experiments, Experiment).write()