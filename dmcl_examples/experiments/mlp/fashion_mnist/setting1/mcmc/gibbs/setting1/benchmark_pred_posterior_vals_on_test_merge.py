# %% Load packages

import pandas as pd

from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.gibbs.setting1.constants import summary_output_run_paths

# %% Concatenate predictions across test subsets

for k in range(num_chains):
    filenames = [
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_subset01.csv'),
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_subset02.csv'),
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_subset03.csv'),
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_subset04.csv'),
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_subset05.csv')
    ]

    combined_csv_data = pd.concat([pd.read_csv(f, header=None, dtype='str') for f in filenames])

# %% Save predictions across test subsets

combined_csv_data.to_csv(summary_output_run_paths[k].joinpath('pred_posterior_on_test.csv'), index=None, header=None)
