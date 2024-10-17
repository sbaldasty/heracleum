from flwr.common import FitRes
from flwr.common import Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from typing import List
from typing import Tuple


class AdversarialScenarioStrategyDecorator(Strategy):


    def __init__(self, delegate: Strategy):
        self.delegate = delegate


    def initialize_parameters(self, client_manager: ClientManager):
        return self.delegate.initialize_parameters(client_manager)


    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return self.delegate.configure_fit(server_round, parameters, client_manager)


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        return self.delegate.aggregate_fit(server_round, results, failures)


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return self.delegate.configure_evaluate(server_round, parameters, client_manager)


    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        return self.delegate.aggregate_evaluate(server_round, results, failures)


    def evaluate(self, server_round: int, parameters: Parameters):
        return self.delegate.evaluate(server_round, parameters)
