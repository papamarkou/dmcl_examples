# %% Load packages

import numpy as np

from kanga.chains import ChainArrays

from dmcl_examples.experiments.mlp.mnist.setting7.mcmc.constants import (
    num_chains, diagnostic_iter_lower_thres, diagnostic_iter_upper_thres
)
from dmcl_examples.experiments.mlp.mnist.setting7.mcmc.gibbs.constants import sampler_output_run_paths
from dmcl_examples.experiments.mlp.mnist.setting7.mcmc.gibbs.sampler import node_subblock_size, sampler

# %% Load acceptance rates

acceptance_rates = []

for i in range(num_chains):
    acceptance_rates.append(np.loadtxt(sampler_output_run_paths[i].joinpath('acceptance_rates.txt')))

# %% Compute acceptance rate summaries per layer

layer_ranges = [0, 780, 790, 800, 810]

means = np.empty([len(layer_ranges)-1, num_chains])
medians = np.empty([len(layer_ranges)-1, num_chains])

for i in range(num_chains):
    means[:, i] = [np.mean(acceptance_rates[i][layer_ranges[j]:layer_ranges[j+1]]) for j in range(len(layer_ranges)-1)]
    medians[:, i] = [np.median(acceptance_rates[i][layer_ranges[j]:layer_ranges[j+1]]) for j in range(len(layer_ranges)-1)]

# %% Save acceptance rates

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('acceptance_rate_means_per_layer.txt'), means[:, i])
    np.savetxt(sampler_output_run_paths[i].joinpath('acceptance_rate_medians_per_layer.txt'), medians[:, i])
