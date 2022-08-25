# %% Import packages

from dmcl_examples.experiments.mlp.mnist.setting1.constants import output_path
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.constants import num_chains, num_epochs
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.dataloaders import training_dataloader
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.constants import sampler_output_run_paths

# %% Define constants

iter_thres = 100000

diagnostic_iter_thres = iter_thres

pred_iter_thres = iter_thres

iter_range_str = '{:07d}_{:07d}'.format(iter_thres, num_epochs * len(training_dataloader))

# %% Define summary-specific output directories

summary_output_run_paths = sampler_output_run_paths.copy()

for i in range(num_chains):
    summary_output_run_paths[i] = sampler_output_run_paths[i].joinpath(iter_range_str)
