# %% Import packages

from dmcl_examples.experiments.mlp.fashion_mnist.setting4.constants import output_path
from dmcl_examples.experiments.mlp.fashion_mnist.setting4.mcmc.constants import num_chains

# %% Define sampler-specific output directories

sampler_output_path = output_path.joinpath('gibbs')
sampler_output_pilot_path = sampler_output_path.joinpath('pilot_run')
sampler_output_run_paths = [
    sampler_output_path.joinpath('run'+str(i+1).zfill(len(str(num_chains)))) for i in range(num_chains)
]
