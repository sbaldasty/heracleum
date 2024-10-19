from flwr.common import FitRes
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
        # FIXME Just a demo for now!
        print(f'I am client {attacker.node_id} and I want to poison my gradients but for now I will just return them unharmed!')
        return self.clean_response(attacker, clean_results)


class ScalingAttack(Attack):
    '''
    A malicious client scales their local update by some value.
    '''
    pass