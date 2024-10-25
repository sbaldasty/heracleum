from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from src.attack import SignFlipAttack
from src.strategy import AdversarialScenarioStrategyDecorator
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

    return Server(client_manager=SimpleClientManager(), strategy=strategy)