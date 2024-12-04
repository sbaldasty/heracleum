from flwr.common import FitRes
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from torch.nn import Module
import itertools
import torch
import numpy as np
from typing import Iterable, List
from typing import Tuple
from dataset import cifar_dataloaders
from copy import deepcopy
from sklearn import metrics
from task import get_device


PUBLIC_DATASET_BATCHES = 2

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
    def detect_corrupt_clients(self, model: torch.nn.Module, results: List[Tuple[ClientProxy, FitRes]]) -> List[ClientProxy]:
        train_loaders, test_loader = cifar_dataloaders(len(results))
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
        centroid_tensor = torch.tensor(centroid)

        # remove 30% of the gradients that are far from the centroid  
        frac_to_remove = 0.3
        dists = self.compute_dists_to_centroid(public_model_updates, centroid)

        # compute the threshold (the radius of the sphere where the center is centroid)
        threshold = self.remove_and_compute_threshold(public_model_updates, dists, frac_to_remove)[1]        

        accused = []
        for client, fitres in results:
            
            # defense: distance betwwen an input data - the centroid > thershold => false   
            ndars = parameters_to_ndarrays(fitres.parameters)
            flatndars = []
            for x in ndars:
                flatndars.append(torch.tensor(x).flatten())
            flatndars = torch.cat(flatndars)
            if (np.linalg.norm(centroid_tensor - flatndars) > threshold):
                accused.append(client)

        return accused
                                        
 
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
        threshold = 0

        num_to_keep = int(np.round(frac_to_keep * np.shape(X)[0]))
        idx_to_keep.append([np.argsort(dists)[:num_to_keep]])
        threshold = np.argsort(dists)[num_to_keep]
            # np.where(Y==y)[0][np.argsort(dist[Y==y])[num_to_keep]]) 이렇게 하면 radius(threshold) 구할 수 있을 것 같음 
        #return X[idx_to_keep, :], threshold
        return 'asdf', threshold
    