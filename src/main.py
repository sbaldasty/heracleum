from client import client_fn

from dataclasses import dataclass

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

import src


from src.attack import SignFlipAttack
from src.task import Net, get_weights
from src.strategy import AdversarialScenarioStrategyDecorator, ClientAwaitStrategyDecorator, RaiseOnFailureStrategyDecorator

from flwr.simulation import start_simulation


def default_fit_config():
    return {
        'batch-size': 32
    }

if __name__ == "__main__":
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    attack = SignFlipAttack()
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=src.server.weighted_average,
        initial_parameters=parameters,
    )
    strategy = AdversarialScenarioStrategyDecorator(strategy, attack, 2)
    strategy = RaiseOnFailureStrategyDecorator(strategy)
    strategy = ClientAwaitStrategyDecorator(strategy, 10)

    start_simulation(client_fn=client_fn, num_clients=10, server=Server(SimpleClientManager(), strategy), num_rounds=3, config=ServerConfig(num_rounds=3))