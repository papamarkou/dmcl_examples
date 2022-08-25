# %% Import packages

from dmcl_examples.experiments.mlp.fashion_mnist.setting4.mcmc.constants import (
    num_chains, num_epochs, num_burnin_epochs, verbose, verbose_step
)
from dmcl_examples.experiments.mlp.fashion_mnist.setting4.mcmc.gibbs.constants import sampler_output_path
from dmcl_examples.experiments.mlp.fashion_mnist.setting4.mcmc.gibbs.sampler import sampler

# %% Benchmark Gibbs sampler

sampler.benchmark(
    num_chains=num_chains,
    num_epochs=num_epochs,
    num_burnin_epochs=num_burnin_epochs,
    path=sampler_output_path,
    check_conditions=None, # lambda chain, runtime : 0.05 <= chain.acceptance_rate() <= 0.99,
    verbose=verbose,
    verbose_step=verbose_step,
    print_acceptance=False,
    print_runtime=True
)
