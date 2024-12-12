from collections import Counter
from dataclass_csv import DataclassWriter
from dataclasses import dataclass
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from src.model import SimpleCNN
from src.attack import NormBallCounterattack
from src.client import client_fn_fn
from src.defense import NormBallDefense
from src.server import make_cifar_server
from src.util import get_device
from src.util import test


OUTPUT_FILE = './out/normball.csv'
N_HONEST_CLIENTS = 10
N_ROUNDS = 30
N_CORRUPT_CLIENTS_START = 0
N_CORRUPT_CLIENTS_END = 4
N_CORRUPT_CLIENTS_STEP = 1


@dataclass
class Experiment:
    n_clients: int
    n_rounds: int
    n_corrupt_clients: int
    accusations: str
    loss: float
    accuracy: float


if __name__ == '__main__':
    max_clients = N_HONEST_CLIENTS + N_CORRUPT_CLIENTS_END
    corrupt_clients_range = range(N_CORRUPT_CLIENTS_START, N_CORRUPT_CLIENTS_END + 1, N_CORRUPT_CLIENTS_STEP)
    experiments = []
    for n_corrupt in corrupt_clients_range:
        n_clients = N_HONEST_CLIENTS + n_corrupt
        model = SimpleCNN()
        accusation_counter = Counter()
        corrupt_client_ids = [None] * n_corrupt
        defense = NormBallDefense(max_clients)
        attack = NormBallCounterattack(defense)
        server = make_cifar_server(model, attack, defense, accusation_counter, corrupt_client_ids, n_clients=n_clients)

        start_simulation(
            client_fn=client_fn_fn(model),
            num_clients=n_clients,
            server=server,
            config=ServerConfig(num_rounds=N_ROUNDS),
            client_resources={'num_cpus': 1, 'num_gpus': 1})

        loss, accuracy = test(model.to(get_device()), max_clients)

        experiments.append(Experiment(
            n_clients=n_clients,
            n_rounds=N_ROUNDS,
            n_corrupt_clients=n_corrupt,
            accusations=str(list(accusation_counter.values())),
            loss=loss,
            accuracy=accuracy))

    with open(OUTPUT_FILE, 'w') as file:
        DataclassWriter(file, experiments, Experiment).write()