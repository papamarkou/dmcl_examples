# %% Import packages

import torch

from dmcl_examples.experiments.mlp.noisy_xor.setting18.mcmc.constants import (
    num_chains, num_epochs, num_burnin_epochs, verbose, verbose_step
)
from dmcl_examples.experiments.mlp.noisy_xor.setting18.mcmc.gibbs.constants import sampler_output_path
from dmcl_examples.experiments.mlp.noisy_xor.setting18.mcmc.gibbs.sampler import sampler

# %% Set number of threads

torch.set_num_threads(4)

# %% Benchmark Gibbs sampler

sampler.benchmark(
    num_chains=num_chains,
    num_epochs=num_epochs,
    num_burnin_epochs=num_burnin_epochs,
    path=sampler_output_path,
    check_conditions=lambda chain, runtime : torch.all(0.05 <= chain.acceptance_rate()),
    verbose=verbose,
    verbose_step=verbose_step,
    print_acceptance=False,
    print_runtime=True
)
