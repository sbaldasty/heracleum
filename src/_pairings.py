from collections import Counter
from dataclass_csv import DataclassWriter
from dataclasses import dataclass
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from src.attack import NormBallCounterattack
from src.client import client_fn
from src.defense import NormBallDefense
from src.server import make_cifar_server
from task import get_device
from task import test


OUTPUT_FILE = './out/pairings/ballnorm.csv'
N_CLIENTS = 10
N_ROUNDS = 30
N_CORRUPT_CLIENTS_START = 0
N_CORRUPT_CLIENTS_END = 5
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
    corrupt_clients_range = range(N_CORRUPT_CLIENTS_START, N_CORRUPT_CLIENTS_END + 1, N_CORRUPT_CLIENTS_STEP)
    experiments = []
    for n_corrupt in corrupt_clients_range:
        accusation_counter = Counter()
        corrupt_client_ids = [None] * n_corrupt
        defense = NormBallDefense(N_CLIENTS)
        attack = NormBallCounterattack(defense)
        server, model = make_cifar_server(attack, defense, accusation_counter, corrupt_client_ids, n_clients=N_CLIENTS)

        start_simulation(
            client_fn=client_fn,
            num_clients=N_CLIENTS,
            server=server,
            config=ServerConfig(num_rounds=N_ROUNDS),
            client_resources={'num_cpus': 1, 'num_gpus': 1})

        loss, accuracy = test(model.to(get_device()), N_CLIENTS)

        experiments.append(Experiment(
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            n_corrupt_clients=n_corrupt,
            accusations=str(list(accusation_counter.values())),
            loss=loss,
            accuracy=accuracy))

    with open(OUTPUT_FILE, 'w') as file:
        DataclassWriter(file, experiments, Experiment).write()