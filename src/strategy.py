from flwr.common import EvaluateIns
from flwr.common import EvaluateRes
from flwr.common import FitIns
from flwr.common import FitRes
from flwr.common import Parameters
from flwr.common import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from random import sample
from src.attack import Attack
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


class RaiseOnFailureStrategyDecorator(StrategyDecorator):

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            raise Exception('A client responded with failure')
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if failures:
            raise Exception('A client responded with failure')
        return super().aggregate_evaluate(server_round, results, failures)


class AdversarialScenarioStrategyDecorator(StrategyDecorator):

    def __init__(self, delegate: Strategy, attack: Attack, n_corrupt_parties: int):
        super().__init__(delegate)
        self.attack = attack
        self.corrupt_party_ids = [None] * n_corrupt_parties
        
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        if server_round == 1:
            node_ids = [x.node_id for x in client_manager.all().values()]
            self.corrupt_party_ids = sample(node_ids, len(self.corrupt_party_ids))
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        poisoned_results = []
        for proxy, res in results:
            if proxy.node_id in self.corrupt_party_ids:
                res = self.attack.poison_gradients(proxy, results)
            poisoned_results.append((proxy, res))
        return super().aggregate_fit(server_round, poisoned_results, failures)