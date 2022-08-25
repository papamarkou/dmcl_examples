# %% Import packages

import numpy as np

from eeyore.samplers import Gibbs

from dmcl_examples.experiments.mlp.fashion_mnist.setting5.mcmc.dataloaders import training_dataloader
from dmcl_examples.experiments.mlp.fashion_mnist.setting5.model import model

# %% Setup Gibbs sampler

scales = [np.sqrt(0.01) for _ in range(10)] + \
    [np.sqrt(0.0001) for _ in range(10)] + \
    [np.sqrt(0.0001) for _ in range(10)] + \
    [np.sqrt(0.00001) for _ in range(10)]

node_subblock_size = [10 for _ in range(10)] + [None for _ in range(30)]

# node_subblock_size = [10 for _ in range(10)] + [None for _ in range(10)] + [5 for _ in range(20)]

sampler = Gibbs(
    model,
    theta0=model.prior.sample(),
    dataloader=training_dataloader,
    scales=scales,
    node_subblock_size=node_subblock_size
)
