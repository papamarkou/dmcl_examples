# %% Import packages

import numpy as np
import torch

from torch.distributions import Normal

from eeyore.models import mlp
from eeyore.stats import binary_cross_entropy

from dmcl_examples.experiments.mlp.noisy_xor.setting16.constants import dtype, mlp_dims

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=mlp_dims)

model = mlp.MLP(loss=lambda x, y: binary_cross_entropy(x, y, reduction='sum'), hparams=hparams, dtype=dtype)

prior_scale = np.sqrt(10.)

model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.full([model.num_params()], prior_scale, dtype=model.dtype)
)
