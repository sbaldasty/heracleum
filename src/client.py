"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.task import Net, get_weights, load_data, set_weights, test, train


client_id_counter = 0

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, client_id, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.client_id = client_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    print(f'ASDF ASDF {partition_id=} and {num_partitions=}')
    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = 32
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = 1
    learning_rate = 0.1

    # Return Client instance
    global client_id_counter
    flower_client = FlowerClient(client_id_counter, trainloader, valloader, local_epochs, learning_rate)
    client_id_counter += 1

    return flower_client.to_client()


# Flower ClientApp
app = ClientApp(client_fn)
