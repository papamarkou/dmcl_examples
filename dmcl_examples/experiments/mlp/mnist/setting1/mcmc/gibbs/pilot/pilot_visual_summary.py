# %% Import packages

import random

import kanga.plots as ps

from kanga.chains import ChainArray

from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.constants import sampler_output_pilot_path
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.pilot.constants import diagnostic_iter_thres

# %% Load chain array

chain_array = ChainArray.from_file(keys=['sample'], path=sampler_output_pilot_path)

# %% Drop burn-in samples

chain_array.vals['sample'] = chain_array.vals['sample'][diagnostic_iter_thres:, :]

# %% Set parameters for which visualizations will be generated

selected_params = sorted(random.sample(range(7850), 10) + random.sample(range(7850, 8179), 20))

# %% Plot traces of simulated chain

for i in selected_params:
    ps.trace(
        chain_array.get_param(i),
        title=r'Traceplot of $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Parameter value'
    )

# %% Plot running means of simulated chain

for i in selected_params:
    ps.running_mean(
        chain_array.get_param(i),
        title=r'Running mean plot of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Running mean'
    )

# %% Plot histograms of marginals of simulated chain

for i in selected_params:
    ps.hist(
        chain_array.get_param(i),
        bins=30,
        density=True,
        title=r'Histogram of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Parameter value',
        ylabel='Parameter relative frequency'
    )
