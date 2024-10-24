import threading
import flwr as fl

from client import client_fn

from dataclasses import dataclass
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

import src


from src.attack import SignFlipAttack
from src.task import Net, get_weights
from src.strategy import AdversarialScenarioStrategyDecorator, ClientAwaitStrategyDecorator, RaiseOnFailureStrategyDecorator

@dataclass
class Experiment:
    pass

def default_fit_config():
    return {
        'batch-size': 32
#num-server-rounds = 3
#fraction-evaluate = 0.5
#local-epochs = 1
#learning-rate = 0.1
#batch-size = 32
    }

# Start the Flower server in a separate thread
def start_server():
    strategy = fl.server.strategy.FedAvg(fit_config=default_fit_config)
    fl.server.start_server(
        server=Server(SimpleClientManager() ),
        server_address="localhost:8080",
        config=default_fit_config(),
        strategy=strategy
    )

# Start the client simulation in another thread
def start_simulation():
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,  # Simulate 10 clients
        config=ServerConfig(num_rounds=3),

    )

if __name__ == "__main__":
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    attack = SignFlipAttack()
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=src.server.weighted_average,
        initial_parameters=parameters,
    )
    strategy = AdversarialScenarioStrategyDecorator(strategy, attack, 2)
    strategy = RaiseOnFailureStrategyDecorator(strategy)
    strategy = ClientAwaitStrategyDecorator(strategy, 10)

    server = Server(SimpleClientManager(), strategy)
    # Start server and simulation concurrently
    server_thread = threading.Thread(target=lambda: fl.server.start_server(
        server=server,
        server_address="localhost:8080",
        config=default_fit_config(),
        strategy=strategy))

    client_thread = threading.Thread(target=lambda: fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,  # Simulate 10 clients
        config=ServerConfig(num_rounds=3)))

    # Start both threads
    server_thread.start()
    client_thread.start()

    # Wait for both to finish
    server_thread.join()
    client_thread.join()
    

# if __name__ == '__main__':
#     # Read from config
#     num_rounds = context.run_config["num-server-rounds"]

#     # Initialize model parameters
#     ndarrays = get_weights(Net())
#     parameters = ndarrays_to_parameters(ndarrays)

#     attack = SignFlipAttack()
#     strategy = FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=context.run_config["fraction-evaluate"],
#         min_available_clients=2,
#         evaluate_metrics_aggregation_fn=weighted_average,
#         initial_parameters=parameters,
#     )
#     strategy = AdversarialScenarioStrategyDecorator(strategy, attack, 2)
#     strategy = RaiseOnFailureStrategyDecorator(strategy)
#     strategy = ClientAwaitStrategyDecorator(strategy, 10)
#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(strategy=strategy, config=config)
