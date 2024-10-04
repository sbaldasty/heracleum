# Heracleum

Introduction goes here...

## Setup

# Links and notes
* [Installing anaconda on linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent)
* [Creating conda projects](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/creating-projects.html)
* [Introduction to federated learning](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html)

Google SearchLabs generated code for modifying gradients with a flower client:
```
import flwr as fl

class MyClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # ... your training code ...

        # Modify gradients here
        for i in range(len(gradients)):
            gradients[i] *= 2  # Double the gradients

        return gradients, len(self.x_train), {}
```

It looks like we could do poison detection with a [custom strategy implementation](https://flower.ai/docs/framework/how-to-implement-strategies.html) on the server.

# Tasks
- [x] Create github repository
- [x] Run a conda demo
- [x] Research flower capabilities
- [ ] Run a flower demo
- [ ] Script to run the flower demo on the vacc
- [ ] Demonstrate loss of model performance from a data poisoning attack
- [ ] Demonstrate loss of model performance from a gradient poisoning attack
- [ ] Figure out how to hook into gradient aggregation serverside
- [ ] What model architectures do the paper use, should we explore different ones?
- [ ] Build out abstractions for data poisoning attacks, gradient poisoning attacks, and poison detection - should attacks be dataset-specific?
- [ ] Identify and collect data about attacks, detection, success, model performance, etc.
- [ ] Identify which attacks and detection methods we want to use
- [ ] Implement them (expand into multiple tasks)
- [ ] Present gathered data in an interesting way