# %% Load packages

import numpy as np

from kanga.chains import ChainArrays

from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.gibbs.constants import (
    sampler_output_path, sampler_output_run_paths
)

# %% Load acceptance rates

acceptance_rates = []

for i in range(num_chains):
    acceptance_rates.append(np.loadtxt(sampler_output_run_paths[i].joinpath('acceptance_rates.txt')))

acceptance_rates = np.stack(acceptance_rates)

# %% Compute acceptance rate summaries per layer

means = np.mean(acceptance_rates, axis=0)
medians = np.median(acceptance_rates, axis=0)

# %% Save acceptance rates

np.savetxt(sampler_output_path.joinpath('acceptance_rate_means_per_layer.txt'), means)
np.savetxt(sampler_output_path.joinpath('acceptance_rate_medians_per_layer.txt'), medians)
