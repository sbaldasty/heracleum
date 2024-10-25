from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from src.client import client_fn
from src.server import make_cifar_server


if __name__ == "__main__":
    start_simulation(
        client_fn=client_fn,
        num_clients=10,
        server=make_cifar_server(),
        config=ServerConfig(num_rounds=3))