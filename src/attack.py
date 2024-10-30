from flwr.common import FitRes
from flwr.common import ndarrays_to_parameters
from flwr.common import parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from numpy.random import normal
from typing import List
from typing import Tuple


class Attack:

    def clean_response(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        for proxy, res in clean_results:
            if proxy == attacker:
                return res
        raise Exception('Attacker was not among the clients')

    def poison_data(self, some_params):
        raise Exception('Not implemented')

    def poison_gradients(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        raise Exception('Not implemented')


class SignFlipAttack(Attack):
    '''
    The adversary flips the signs of the local updates of a corrupt client.
    '''

    def poison_gradients(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        clean_response = self.clean_response(attacker, clean_results)
        clean_ndarrays = parameters_to_ndarrays(clean_response.parameters)
        poisoned_ndarrays = [x * -1 for x in clean_ndarrays]
        poisoned_parameters = ndarrays_to_parameters(poisoned_ndarrays)
        return FitRes(clean_response.status, poisoned_parameters, clean_response.num_examples, clean_response.metrics)


class ScalingAttack(Attack):
    '''
    The adversary scales the local updates of a corrupt client by some factor.
    '''

    def __init__(self, factor: float):
        self.factor = factor

    def poison_gradients(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        clean_response = self.clean_response(attacker, clean_results)
        clean_ndarrays = parameters_to_ndarrays(clean_response.parameters)
        poisoned_ndarrays = [x * self.factor for x in clean_ndarrays]
        poisoned_parameters = ndarrays_to_parameters(poisoned_ndarrays)
        return FitRes(clean_response.status, poisoned_parameters, clean_response.num_examples, clean_response.metrics)


class GaussianNoiseAttack(Attack):
    '''
    The adversary adds gaussian noise to the local updates of a corrupted client.
    '''
    def __init__(self, mean: float, stdev: float):
        self.mean = mean
        self.stdev = stdev

    def poison_gradients(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        clean_response = self.clean_response(attacker, clean_results)
        clean_ndarrays = parameters_to_ndarrays(clean_response.parameters)
        poisoned_ndarrays = [x + normal(self.mean, self.stdev, x.shape) for x in clean_ndarrays]
        poisoned_parameters = ndarrays_to_parameters(poisoned_ndarrays)
        return FitRes(clean_response.status, poisoned_parameters, clean_response.num_examples, clean_response.metrics)