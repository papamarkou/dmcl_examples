# %% Load packages

import numpy as np

from kanga.chains import ChainArrays

from dmcl_examples.experiments.mlp.fashion_mnist.setting8.mcmc.constants import (
    num_chains, diagnostic_iter_lower_thres, diagnostic_iter_upper_thres
)
from dmcl_examples.experiments.mlp.fashion_mnist.setting8.mcmc.gibbs.constants import sampler_output_run_paths

# %% Load chain lists

chain_arrays = ChainArrays.from_file(sampler_output_run_paths, keys=['accepted'])

# %% Drop burn-in samples

chain_arrays.vals['accepted'] = np.array(
    [chain_arrays.vals['accepted'][i][diagnostic_iter_lower_thres:diagnostic_iter_upper_thres] for i in range(num_chains)]
)

# %% Compute acceptance rates

acceptance_rates = chain_arrays.block_acceptance_rate()

# %% Save acceptance rates

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('acceptance_rates.txt'), acceptance_rates[i])
