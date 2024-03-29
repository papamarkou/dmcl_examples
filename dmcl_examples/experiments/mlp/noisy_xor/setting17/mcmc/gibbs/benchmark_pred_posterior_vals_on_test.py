# %% Load packages

import numpy as np
import torch

from eeyore.chains import ChainLists

from dmcl_examples.experiments.mlp.noisy_xor.setting17.constants import dtype
from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.constants import (
    num_chains, pred_iter_lower_thres, pred_iter_upper_thres
)
from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.datascanners import test_dataloader
from dmcl_examples.experiments.mlp.noisy_xor.setting17.mcmc.gibbs.constants import sampler_output_run_paths
from dmcl_examples.experiments.mlp.noisy_xor.setting17.model import model

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][pred_iter_lower_thres:pred_iter_upper_thres]

# %% Compute and save predictive posteriors

verbose_msg = 'Evaluating predictive posterior based on chain {:' \
    + str(len(str(num_chains))) \
    + '} out of ' \
    + str(num_chains) \
    + ' at test point {:' \
    + str(len(str(len(test_dataloader)))) \
    + '} out of ' \
    + str(len(test_dataloader)) \
    + '...'

for k in range(num_chains):
    test_pred_probs = np.empty([len(test_dataloader), 2])
    nums_dropped_samples = np.empty([len(test_dataloader)], dtype=np.int64)

    for i, (x, _) in enumerate(test_dataloader):
        print(verbose_msg.format(k+1, i+1))

        integral, num_dropped_samples = model.predictive_posterior(
            chain_lists.vals['sample'][k], x, torch.tensor([[1.]], dtype=dtype)
        )
        test_pred_probs[i, 1] = integral.item()
        test_pred_probs[i, 0] = 1. - test_pred_probs[i, 1]
        nums_dropped_samples[i] = num_dropped_samples

    np.savetxt(sampler_output_run_paths[k].joinpath('pred_posterior_on_test.csv'), test_pred_probs, delimiter=',')
    np.savetxt(
        sampler_output_run_paths[k].joinpath('pred_posterior_on_test_num_dropped_samples.txt'),
        nums_dropped_samples,
        fmt='%d'
    )
