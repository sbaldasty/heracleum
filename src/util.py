from collections import OrderedDict
from dataset import cifar_dataloaders

import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def test(net, n_clients):
    """Validate the model on the test set."""
    train_loaders, test_loader = cifar_dataloaders(n_clients)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(get_device())
            labels = labels.to(get_device())
            net.to(get_device())
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    loss = loss / len(test_loader)
    return loss, accuracy