# %% Import packages

from datetime import timedelta
from timeit import default_timer as timer

from dmcl_examples.experiments.mlp.fashion_mnist.setting5.mcmc.constants import (
    num_burnin_epochs, num_epochs, verbose, verbose_step
)
from dmcl_examples.experiments.mlp.fashion_mnist.setting5.mcmc.gibbs.constants import sampler_output_pilot_path
from dmcl_examples.experiments.mlp.fashion_mnist.setting5.mcmc.gibbs.sampler import sampler

# %% Run Gibbs sampler

start_time = timer()

sampler.run(num_epochs=num_epochs, num_burnin_epochs=num_burnin_epochs, verbose=verbose, verbose_step=verbose_step)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Save chain array

sampler.get_chain().to_chainfile(keys=['sample', 'accepted'], path=sampler_output_pilot_path, mode='w')
