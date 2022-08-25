# %% Load packages

import pandas as pd

from dmcl_examples.experiments.mlp.fashion_mnist.setting7.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.fashion_mnist.setting7.mcmc.gibbs.constants import sampler_output_run_paths

# %% Concatenate predictions across test subsets

for k in range(num_chains):
    filenames = [
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_subset01.csv'),
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_subset02.csv'),
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_subset03.csv'),
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_subset04.csv'),
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_subset05.csv')
    ]

    combined_csv_data = pd.concat([pd.read_csv(f, header=None, dtype='str') for f in filenames])

# %% Save predictions across test subsets

combined_csv_data.to_csv(sampler_output_run_paths[k].joinpath('pred_posterior_on_test.csv'), index=None, header=None)
