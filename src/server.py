from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from src.attack import Attack
from src.defense import Defense
from src.strategy import AttackStrategyDecorator
from src.strategy import DefenseStrategyDecorator
from src.strategy import ModelUpdateStrategyDecorator
from src.strategy import RaiseOnFailureStrategyDecorator
from src.task import Net
from src.task import get_weights
from typing import List
from typing import Tuple

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def make_cifar_server(
        attack: Attack,
        defense: Defense,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        n_clients=10,
        n_corrupt_clients=2):

    model = Net()
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=n_clients,
        initial_parameters=parameters)

    strategy = DefenseStrategyDecorator(strategy, defense)
    strategy = AttackStrategyDecorator(strategy, attack, n_corrupt_clients)
    strategy = ModelUpdateStrategyDecorator(strategy, model)
    strategy = RaiseOnFailureStrategyDecorator(strategy)

    return Server(client_manager=SimpleClientManager(), strategy=strategy), model