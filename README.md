# Heracleum

Introduction goes here...

## Setup

# Links and notes

### Premise from Joe Near

> The defenses usually output a binary decision: "ok" or "not ok." Often,
the defenses determine the binary output by comparing some statistic
against a threshold. If the statistic is above the threshold, the
defense outputs "not ok" and the update is considered malicious (and
thus excluded from the aggregated model). The attacks they evaluate
against are usually designed to produce *maximal* reduction in accuracy,
which means it's usually easy to choose a threshold that includes all
the honest updates but excludes the malicious ones.
>
>I suspect that if the adversary knows the threshold, they could poison
their data *just enough* to fall right below the threshold, and pass the
defense check while still reducing accuracy of the final model.

### Links

* [EIFFL: Ensuring Integrity For Federated Learning](https://arxiv.org/pdf/2112.12727)
* [Installing anaconda on linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent)
* [Creating conda projects](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/creating-projects.html)
* [Introduction to federated learning](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html)

### Notes

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

Here are [flower baselines](https://flower.ai/docs/baselines/how-to-contribute-baselines.html). They are reproductions of the results of papers. We should consider contributing one for EIFFeL since we will be reproducing a portion of their work.

# Tasks
- [x] Create github repository
- [x] Run a conda demo
- [x] Research flower capabilities
- [x] Run a flower demo
- [x] Script to run the flower demo on the vacc
- [x] Evaluate performance of models
- [ ] Collect experiment data in a csv file
- [ ] Demonstrate loss of model performance from a gradient poisoning attack
- [ ] Figure out how to hook into gradient aggregation serverside, demonstrate poison defense
- [ ] Present gathered data in an interesting way
- [ ] Prepare project proposal presentation
- [ ] Demonstrate loss of model performance from a data poisoning attack
- [ ] What model architectures do the paper use, should we explore different ones?
- [ ] Build out abstractions for data poisoning attacks, gradient poisoning attacks, and poison detection - should attacks be dataset-specific?
- [ ] Identify and collect data about attacks, detection, success, model performance, etc.
- [ ] Identify which attacks and detection methods we want to use
- [ ] Implement them (expand into multiple tasks)
