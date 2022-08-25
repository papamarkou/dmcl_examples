# %% Load packages

import numpy as np

from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.constants import num_chains
from dmcl_examples.experiments.mlp.mnist.setting1.mcmc.gibbs.setting1.constants import summary_output_run_paths

# %% Make and save predictions

for i in range(num_chains):
    test_pred_probs = np.loadtxt(
        summary_output_run_paths[i].joinpath('pred_posterior_on_test.csv'), delimiter=',', skiprows=0
    )

    test_preds = np.argmax(test_pred_probs, axis=1)

    np.savetxt(summary_output_run_paths[i].joinpath('preds_via_bm.txt'), test_preds, fmt='%d')
