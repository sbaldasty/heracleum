from flwr.common import FitRes
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from torch.nn import Module
import torch
import numpy as np
from typing import List
from typing import Tuple
from dataset import cifar_dataloaders
from copy import deepcopy
from sklearn import metrics

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

        public_model_updates = [] 
        loss_fn = torch.nn.CrossEntropyLoss()
        # for every data train the model on and get gradient 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for i, data in enumerate(test_loader):                    
            inputs, labels = data
            for input, label in zip(inputs, labels):
                # initialize model parameters
                reset_model = deepcopy(model) 
                # initialize gradient 
                optimizer.zero_grad()
                
                # train the model on the input 
                output = model(input)

                # compute the loss and its gradient 
                loss = loss_fn(output, label)
                loss.backward()
            
                public_model_updates.append(reset_model.parameters())

        # compute the mean of the model updates (centroid)
        centroid = np.mean(public_model_updates, axis = 0)

        # remove 30% of the gradients that are far from the centroid  
        frac_to_remove = 0.3
        dists = self.compute_dists_to_centroid(public_model_updates, centroid)

        # compute the threshold (the radius of the sphere where the center is centroid)
        threshold = self.remove_and_compute_threshold(public_model_updates, dists, frac_to_remove)[1]        

        accused = []
        for client, fitres in results:
            
            # defense: distance betwwen an input data - the centroid > thershold => false   
            
            if (metrics.pairwise_distances(centroid, parameters_to_ndarrays(fitres.parameters), metric = 'euclidean') > threshold):
                accused.append(client)
        return accused
                                        
 
    def compute_dists_to_centroid(self, X, centroid):
        """
        for each class, 
        2) compute the distances from the data samples to the centroid
        3) return the distances  
        """
        dists = np.zeros(X.shape[0])

        for data_point in X:
            dists.append(metrics.pairwise.pairwise_distances(data_point, centroid, metric = 'euclidean'))
        
        return dists 

    def remove_and_compute_threshold(X, dists, frac_to_remove):
        assert frac_to_remove >= 0
        assert frac_to_remove <= 1
        frac_to_keep = 1.0 - frac_to_remove  
        idx_to_keep = []
        threshold = 0

        num_to_keep = int(np.round(frac_to_keep * np.shape(X)[0]))
        idx_to_keep.append([np.argsort(dists)[:num_to_keep]])
        threshold = np.argsort(dists)[num_to_keep]
            # np.where(Y==y)[0][np.argsort(dist[Y==y])[num_to_keep]]) 이렇게 하면 radius(threshold) 구할 수 있을 것 같음 
        return X[idx_to_keep, :], threshold
    