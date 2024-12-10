from dataset import cifar_dataloaders
from flwr.client import NumPyClient
from flwr.common import Context
from util import get_weights, set_weights
from util import get_device
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


class FlowerClient(NumPyClient):
    def __init__(self, module, trainloader, local_epochs, learning_rate):
        self.net = module
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.lr = learning_rate

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        self.net.to(get_device())
        criterion = CrossEntropyLoss().to(get_device())
        optimizer = SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.net.train()
        for _ in range(self.local_epochs):
            for batch in self.trainloader:
                #print(f'BATCH BATCH {batch.}')
                images, labels = batch
                optimizer.zero_grad()
                criterion(self.net(images.to(get_device())), labels.to(get_device())).backward()
                optimizer.step()

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.trainloader.dataset), {}


def client_fn_fn(module: Module):
    def client_fn(context: Context):
        partition_id = int(context.node_config['partition-id'])
        train_loaders, test_loader = cifar_dataloaders(int(context.node_config['num-partitions']))
        flower_client = FlowerClient(module, train_loaders[partition_id], local_epochs=1, learning_rate=0.01)
        return flower_client.to_client()
    
    return client_fn