from flwr.common import EvaluateIns
from flwr.common import EvaluateRes
from flwr.common import FitIns
from flwr.common import FitRes
from flwr.common import Parameters
from flwr.common import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class StrategyDecorator(Strategy):

    def __init__(self, delegate: Strategy):
        self.delegate = delegate

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.delegate.initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        return self.delegate.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        return self.delegate.aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.delegate.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.delegate.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.delegate.evaluate(server_round, parameters)


class AdversarialScenarioStrategyDecorator(StrategyDecorator):

    def __init__(self, delegate: Strategy):
        super().__init__(delegate)
        self.delegate = delegate
        self.client_ids = []

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        for s, proxy in client_manager.all().items():
            if s not in self.client_ids:
                self.client_ids.append(s)
                print(f'Found {s} / {proxy.node_id}')

        print(f'All available clients: {self.client_ids}')
        return super().configure_fit(server_round, parameters, client_manager)