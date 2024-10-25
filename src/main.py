from client import client_fn

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

from src.server import weighted_average
from src.server import make_cifar_server

from src.attack import SignFlipAttack
from src.task import Net, get_weights
from src.strategy import AdversarialScenarioStrategyDecorator, RaiseOnFailureStrategyDecorator

from flwr.simulation import start_simulation


if __name__ == "__main__":
    start_simulation(
        client_fn=client_fn,
        num_clients=10,
        server=make_cifar_server(),
        config=ServerConfig(num_rounds=3))