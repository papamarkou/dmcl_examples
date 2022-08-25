# %% Load packages

import numpy as np
import torch

from pathlib import Path

from eeyore.constants import torch_to_np_types

from dmcl_examples.experiments.mlp.mnist.setting4.constants import dtype, num_classes
from dmcl_examples.experiments.mlp.mnist.setting4.mcmc.constants import (
    batch_size, num_epochs, pred_iter_thres, pred_iter_step
)
from dmcl_examples.experiments.mlp.mnist.setting4.mcmc.dataloaders import training_dataloader
from dmcl_examples.experiments.mlp.mnist.setting4.mcmc.datascanners import test_dataloader
from dmcl_examples.experiments.mlp.mnist.setting4.mcmc.gibbs.constants import sampler_output_run_paths
from dmcl_examples.experiments.mlp.mnist.setting4.model import model

# %%

num_preds = int((num_epochs * len(training_dataloader.dataset) / batch_size - pred_iter_thres) / pred_iter_step)
# int((5500 * 60000 / 3000 - 10000) / 1000)
l = 0

for path in sampler_output_run_paths:
    integral = torch.zeros([len(test_dataloader), num_classes], dtype=dtype)
    num_kept_samples = torch.ones([len(test_dataloader), num_classes], dtype=torch.int64)

    accuracies = torch.zeros(num_preds)
    num_correct_preds = 0

    with open(Path(path, 'sample.csv')) as file:
        for i, line in enumerate(file):
            if i >= pred_iter_thres:
                theta = torch.from_numpy(np.fromstring(line, dtype=torch_to_np_types[dtype], sep=','))
                model.set_params(theta.clone().detach())

                for j, (x, label) in enumerate(test_dataloader):
                    print(i, j)
                    for k in range(num_classes):
                        y = torch.zeros([1, num_classes], dtype=dtype)
                        y[0, k] = 1.

                        integrand = model.log_lik(x, y)

                        if not torch.isnan(integrand):
                            integral[j, k] = (
                                ((num_kept_samples[j, k] - 1) * integral[j, k] + integrand) / num_kept_samples[j, k]
                            )
                            num_kept_samples[j, k] = num_kept_samples[j, k] + 1

                    pred = torch.argmax(integral[j, :])

                    if pred == torch.argmax(label):
                        num_correct_preds = num_correct_preds + 1

            if ((i+1) % pred_iter_step) == 0:
                accuracies[l] = num_correct_preds
                l = l+1
