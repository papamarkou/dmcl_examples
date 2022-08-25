# %% Import packages

import matplotlib.pyplot as plt
# import numpy as np
import random
import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.models import mlp

from dmcl_examples.experiments.mlp.mnist.setting7.mcmc.load_datasets import training_dataset
from dmcl_examples.experiments.mlp.mnist.setting7.constants import dtype
from dmcl_examples.experiments.mlp.mnist.setting7.model import hparams, prior_scale
from dmcl_examples.experiments.mlp.mnist.setting7.mcmc.batch_sampler import BatchSampler

# %% Set seeds for reproducibility

random.seed(0)
# np.random.seed(0)
torch.manual_seed(0)

# %% Setup MLP model with binary cross entropy likelihood using mean reduction

model = mlp.MLP(loss=lambda x, y: nn.CrossEntropyLoss(reduction='mean')(x, torch.argmax(y, 1)), hparams=hparams, dtype=dtype)

model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype), torch.full([model.num_params()], prior_scale, dtype=model.dtype)
)

# %% Sample parameters from prior

param_vals = model.prior.sample()

# %% Set model parameters

model.set_params(param_vals.clone().detach())

# %% Evaluate log-likelihood for varying batch size

first_batch_size = 5000 # 500
batch_step = 5000 # 1000

num_samples = (len(training_dataset) - first_batch_size) // batch_step + 1

batch_sizes = range(first_batch_size, first_batch_size+num_samples*batch_step, batch_step)

num_iters = 10

batch_log_lik_vals = torch.empty(len(batch_sizes), num_iters, dtype=dtype)

for (i, batch_size) in enumerate(batch_sizes):
    batch_sampler = BatchSampler(training_dataset, batch_size)

    batch_dataloader = DataLoader(training_dataset, batch_sampler=batch_sampler)

    for j in range(num_iters):
        for x, y in batch_dataloader:
            batch_log_lik_vals[i, j] = model.log_lik(x, y).clone().detach()

batch_log_lik_means = batch_log_lik_vals.mean(axis=1)

batch_log_lik_vars = batch_log_lik_vals.var(axis=1)
print("Variances per batch size: {}".format(batch_log_lik_vars))

# %% Evaluate log-likelihood (reduction set to 'mean') for the whole dataset

full_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset))
x, y = next(iter(full_dataloader))
full_log_lik_val = model.log_lik(x, y).clone().detach()

# %% Generate boxplots of log-likelihood (divded by batch size) per batch size

plt.figure(figsize=(16, 8))

plt.boxplot(batch_log_lik_vals, labels=batch_sizes, medianprops=dict(color="orange"))

plt.scatter(range(1, len(batch_log_lik_means)+1), batch_log_lik_means, color="red", marker="*")

plt.axhline(y=full_log_lik_val, color="green", linestyle='-')

plt.xticks(rotation=90)

font_size = 16

plt.rc('font', size=font_size)         # controls default text sizes
plt.rc('axes', titlesize=font_size)    # fontsize of the axes title
plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)   # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)   # fontsize of the tick labels
# plt.rc('legend', fontsize=font_size) # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
