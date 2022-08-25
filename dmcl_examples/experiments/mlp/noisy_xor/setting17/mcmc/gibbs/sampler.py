# %% Import packages

import numpy as np

from eeyore.samplers import Gibbs

from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.dataloaders import training_dataloader
from dmcl_examples.experiments.mlp.noisy_xor.setting17.model import model

# %% Setup Gibbs sampler

scales = [np.sqrt(0.001) for _ in range(13)]

node_subblock_size = [None for _ in range(13)]

sampler = Gibbs(
    model,
    theta0=model.prior.sample(),
    dataloader=training_dataloader,
    scales=scales,
    node_subblock_size=node_subblock_size
)
