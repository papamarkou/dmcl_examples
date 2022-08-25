# %% Load packages

import numpy as np
import torch

from sklearn.metrics import accuracy_score

from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.dataloaders import test_dataloader
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.gibbs.constants import sampler_output_path
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.gibbs.setting1.constants import (
    iter_range_str, summary_output_run_paths
)

# %% Load test data and labels

_, test_labels = next(iter(test_dataloader))

# %% Compute predictive accuracies

accuracies = np.empty(num_chains)

for i in range(num_chains):
    test_preds = np.loadtxt(summary_output_run_paths[i].joinpath('preds_via_bm.txt'), skiprows=0)

    accuracies[i] = accuracy_score(test_preds, torch.argmax(test_labels, 1))

# %% Save predictive accuracies

np.savetxt(sampler_output_path.joinpath('_'.join([iter_range_str, 'accuracies_via_bm.txt'])), accuracies)
