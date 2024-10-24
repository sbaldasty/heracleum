from flwr.common import FitRes
from flwr.common import ndarrays_to_parameters
from flwr.common import parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
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
    A malicious client flips the sign of their local update.
    '''
    def poison_gradients(self, attacker: ClientProxy, clean_results: List[Tuple[ClientProxy, FitRes]]) -> FitRes:
        clean_response = self.clean_response(attacker, clean_results)
        clean_ndarrays = parameters_to_ndarrays(clean_response.parameters)
        poisoned_ndarrays = [x * -1 for x in clean_ndarrays]
        print(f'I am client {attacker.node_id} and I change for example {clean_ndarrays[0][0][0]} to {poisoned_ndarrays[0][0][0]}!')
        poisoned_parameters = ndarrays_to_parameters(poisoned_ndarrays)
        return FitRes(clean_response.status, poisoned_parameters, clean_response.num_examples, clean_response.metrics)


class ScalingAttack(Attack):
    '''
    A malicious client scales their local update by some value.
    '''
    pass