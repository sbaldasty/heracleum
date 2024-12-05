from flwr.common import FitRes
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from torch.nn import Module
import itertools
import torch
import numpy as np
from typing import List
from typing import Tuple
from dataset import cifar_dataloaders
from copy import deepcopy


PUBLIC_DATASET_BATCHES = 2


class Defense:

    def on_model_update(self, model: Module):
        raise Exception('Not implemented')

    def detect_corrupt_clients(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        raise Exception('Not implemented')


class AbsentDefense(Defense):

    def on_model_update(self, model: Module):
        pass

    def detect_corrupt_clients(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        return []


class NormBallDefense(Defense):

    def __init__(self, n_clients: int):
        self.n_clients = n_clients

    def on_model_update(self, model: Module):
        train_loaders, test_loader = cifar_dataloaders(self.n_clients)
        model = deepcopy(model)
        state = model.state_dict()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        public_model_updates = []
        for i, (inputs, labels) in itertools.islice(enumerate(test_loader), PUBLIC_DATASET_BATCHES):
            labels = labels.unsqueeze(1)
            for input, label in zip(inputs, labels):
                # for every data train the model on and get gradient 
                model.load_state_dict(state)
                optimizer.zero_grad()
                
                output = model(input)

                # compute the loss and its gradient 
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                flat_params = torch.cat([param.view(-1) for param in model.parameters()])
                public_model_updates.append([x.detach().numpy() for x in flat_params])

        # compute the mean of the model updates (centroid)
        centroid = np.mean(public_model_updates, axis = 0)

        # remove 30% of the gradients that are far from the centroid  
        frac_to_remove = 0.3
        dists = self.compute_dists_to_centroid(public_model_updates, centroid)

        self.latest_model = model
        self.centroid = torch.tensor(centroid)
        # compute the threshold (the radius of the sphere where the center is centroid)
        self.threshold = self.remove_and_compute_threshold(public_model_updates, dists, frac_to_remove)

    def detect_corrupt_clients(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        accused = []
        for client, fitres in results:
            # defense: distance betwwen an input data - the centroid > thershold => false   
            ndars = parameters_to_ndarrays(fitres.parameters)
            if self.accuses(ndars):
                accused.append(client)

        return accused


    def accuses(self, client_params: NDArrays) -> bool:
        flatndars = []
        for x in client_params:
            flatndars.append(torch.tensor(x).flatten())
        flatndars = torch.cat(flatndars)
        return np.linalg.norm(self.centroid - flatndars) > self.threshold


    def compute_dists_to_centroid(self, X, centroid):
        """
        for each class, 
        2) compute the distances from the data samples to the centroid
        3) return the distances  
        """
        return [np.linalg.norm(x - centroid) for x in X]


    def remove_and_compute_threshold(self, X, dists, frac_to_remove):
        assert frac_to_remove >= 0
        assert frac_to_remove <= 1
        frac_to_keep = 1.0 - frac_to_remove  
        idx_to_keep = []
        num_to_keep = int(np.round(frac_to_keep * np.shape(X)[0]))
        idx_to_keep.append([np.argsort(dists)[:num_to_keep]])
        return np.argsort(dists)[num_to_keep]
