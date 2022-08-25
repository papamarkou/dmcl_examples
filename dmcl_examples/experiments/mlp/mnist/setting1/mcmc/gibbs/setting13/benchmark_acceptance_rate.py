# %% Load packages

import numpy as np

from kanga.chains import ChainArrays

from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.constants import sampler_output_run_paths
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.setting13.constants import (
    diagnostic_iter_thres, summary_output_run_paths
)

# %% Load chain lists

chain_arrays = ChainArrays.from_file(sampler_output_run_paths, keys=['accepted'])

# %% Drop burn-in samples

chain_arrays.vals['accepted'] = np.array(
    [chain_arrays.vals['accepted'][i][diagnostic_iter_thres:] for i in range(num_chains)]
)

# %% Compute acceptance rates

acceptance_rates = chain_arrays.block_acceptance_rate()

# %% Save acceptance rates

for i in range(num_chains):
    np.savetxt(summary_output_run_paths[i].joinpath('acceptance_rates.txt'), acceptance_rates[i])
