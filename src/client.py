from flwr.client import NumPyClient
from flwr.common import Context

from src.task import Net, get_weights, load_data, set_weights, train

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.lr
        )
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.trainloader.dataset), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = 32
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = 1
    learning_rate = 0.01

    # Return Client instance
    flower_client = FlowerClient(trainloader, valloader, local_epochs, learning_rate)
    return flower_client.to_client()