from collections import Counter
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
from src.util import get_weights
from torch.nn import Module


def make_cifar_server(
        model: Module,
        attack: Attack,
        defense: Defense,
        accusation_counter: Counter,
        corrupt_client_ids: list[str],
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        n_clients=10):

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)
    defense.on_model_update(model)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=n_clients,
        initial_parameters=parameters)

    strategy = DefenseStrategyDecorator(strategy, defense, model, accusation_counter)
    strategy = AttackStrategyDecorator(strategy, attack, corrupt_client_ids)
    strategy = ModelUpdateStrategyDecorator(strategy, model, [defense])
    strategy = RaiseOnFailureStrategyDecorator(strategy)

    return Server(client_manager=SimpleClientManager(), strategy=strategy)