from typing import List, Tuple

from flwr.common import Metrics, ndarrays_to_parameters

from src.attack import SignFlipAttack
from src.task import Net, get_weights
from src.strategy import AdversarialScenarioStrategyDecorator, ClientAwaitStrategyDecorator, RaiseOnFailureStrategyDecorator
from flwr.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def make_cifar_server():
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    attack = SignFlipAttack()
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters)

    strategy = AdversarialScenarioStrategyDecorator(strategy, attack, 2)
    strategy = RaiseOnFailureStrategyDecorator(strategy)
    #strategy = ClientAwaitStrategyDecorator(strategy, 10)

    return Server(SimpleClientManager(), strategy)