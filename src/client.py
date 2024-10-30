from flwr.client import NumPyClient
from flwr.common import Context

from src.task import Net, get_weights, set_weights
from src.task import get_device
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


train_loaders = None


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, local_epochs, learning_rate):
        self.net = Net()
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
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                criterion(self.net(images.to(get_device())), labels.to(get_device())).backward()
                optimizer.step()

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.trainloader.dataset), {}


def client_fn(context: Context):
    global train_loaders
    partition_id = int(context.node_config["partition-id"])
    flower_client = FlowerClient(train_loaders[partition_id], local_epochs=1, learning_rate=0.01)
    return flower_client.to_client()