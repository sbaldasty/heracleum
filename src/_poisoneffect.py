from collections import Counter
from dataclass_csv import DataclassWriter
from dataclasses import dataclass
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from itertools import product
from src.attack import AbsentAttack
from src.attack import GaussianNoiseAttack
from src.attack import ScalingAttack
from src.attack import SignFlipAttack
from src.client import client_fn
from src.defense import AbsentDefense
from src.defense import NormBallDefense
from src.server import make_cifar_server
from task import get_device
from task import test

OUTPUT_FILE = './out/poisoneffect/poisoneffect.csv'
N_CLIENTS = 3
N_ROUNDS = 2
N_CORRUPT_CLIENTS_START = 2
N_CORRUPT_CLIENTS_END = 2
N_CORRUPT_CLIENTS_STEP = 1

NOISE_ATTACK_MEAN = 0.0
NOISE_ATTACK_STDEV = 0.5
SCALING_ATTACK_FACTOR = 2.0


@dataclass
class Experiment:
    n_clients: int
    n_rounds: int
    n_corrupt_clients: int
    attack: str
    defense: str
    accusations: str
    loss: float
    accuracy: float


if __name__ == '__main__':
    attacks = [
#        ('No attack', AbsentAttack()),
#        ('Sign flipping', SignFlipAttack()),
        (f'Scaling (factor={SCALING_ATTACK_FACTOR})', ScalingAttack(SCALING_ATTACK_FACTOR))]
#        (f'Gaussian noise (mu={NOISE_ATTACK_MEAN}, sigma={NOISE_ATTACK_STDEV})', GaussianNoiseAttack(NOISE_ATTACK_MEAN, NOISE_ATTACK_STDEV))]

    defenses = [
#        ('No defense', AbsentDefense()),
        ('Norm ball', NormBallDefense(N_CLIENTS))]

    corrupt_clients_range = range(N_CORRUPT_CLIENTS_START, N_CORRUPT_CLIENTS_END + 1, N_CORRUPT_CLIENTS_STEP)
    experiments = []
    for (attack_name, attack_obj), (defense_name, defense_obj), n_corrupt in product(attacks, defenses, corrupt_clients_range):
        accusation_counter = Counter()
        corrupt_client_ids = [None] * n_corrupt
        server, model = make_cifar_server(attack_obj, defense_obj, accusation_counter, corrupt_client_ids, n_clients=N_CLIENTS)

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
            attack=attack_name,
            defense=defense_name,
            accusations=str(list(accusation_counter.values())),
            loss=loss,
            accuracy=accuracy))

    with open(OUTPUT_FILE, 'w') as file:
        DataclassWriter(file, experiments, Experiment).write()