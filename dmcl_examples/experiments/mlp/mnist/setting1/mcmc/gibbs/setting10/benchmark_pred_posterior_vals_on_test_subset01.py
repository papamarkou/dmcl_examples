# %% Load packages

import numpy as np
import torch

from eeyore.chains import ChainLists

from dmcl_examples.experiments.mlp.mnist.setting1.constants import dtype, num_classes
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.constants import num_chains, pred_iter_thres
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.datascanners import test_dataloader_subset01
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.constants import sampler_output_run_paths
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.setting10.constants import summary_output_run_paths
from dmcl_examples.experiments.mlp.mnist.setting1.model import model

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][pred_iter_thres:]

# %% Compute and save predictive posteriors

verbose_msg = 'Evaluating predictive posterior based on chain {:' \
    + str(len(str(num_chains))) \
    + '} out of ' \
    + str(num_chains) \
    + ' at test point {:' \
    + str(len(str(len(test_dataloader_subset01)))) \
    + '} out of ' \
    + str(len(test_dataloader_subset01)) \
    + '...'

for k in range(num_chains):
    test_pred_probs = np.empty([len(test_dataloader_subset01), num_classes])
    nums_dropped_samples = np.empty([len(test_dataloader_subset01), num_classes], dtype=np.int64)

    for i, (x, _) in enumerate(test_dataloader_subset01):
        print(verbose_msg.format(k+1, i+1))

        for j in range(num_classes):
            y = torch.zeros([1, num_classes], dtype=dtype)
            y[0, j] = 1.
            integral, num_dropped_samples = model.predictive_posterior(chain_lists.vals['sample'][k], x, y)
            test_pred_probs[i, j] = integral.item()
            nums_dropped_samples[i, j] = num_dropped_samples

    np.savetxt(summary_output_run_paths[k].joinpath('pred_posterior_on_test.csv'), test_pred_probs, delimiter=',')
    np.savetxt(
        summary_output_run_paths[k].joinpath('pred_posterior_on_test_num_dropped_samples.csv'),
        nums_dropped_samples,
        fmt='%d',
        delimiter=','
    )
