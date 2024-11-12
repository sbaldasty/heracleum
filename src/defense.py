from flwr.common import FitRes
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from torch.nn import Module
from typing import List
from typing import Tuple


class Defense:

    def detect_corrupt_clients(self, model: Module, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        raise Exception('Not implemented')


class AbsentDefense(Defense):

    def detect_corrupt_clients(self, model: Module, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        return []


class NormBallDefense(Defense):

    # TODO We probably have to initialize a model and other things?
    def __init__(self):
        pass

    # TODO Threshold updates need to happen here?
    def detect_corrupt_clients(self, model: Module, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        accused = []
        for client, response in results:
            ndarrays = parameters_to_ndarrays(response.parameters)
            if self.exceeds_threshold(ndarrays):
                accused.append(client)
        return accused

    # TODO Should threshold be passed in, or stored in the object?
    def exceeds_threshold(self, update: NDArrays) -> bool:
        return False