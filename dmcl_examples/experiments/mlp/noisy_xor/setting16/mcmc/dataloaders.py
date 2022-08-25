# %% Import packages

from torch.utils.data import DataLoader

from dmcl_examples.datasets import load_xydataset_from_file
from dmcl_examples.datasets.noisy_xor.data2.constants import test_data_path, training_data_path
from dmcl_examples.experiments.mlp.noisy_xor.setting16.constants import dtype
from dmcl_examples.experiments.mlp.noisy_xor.setting16.mcmc.constants import shuffle

# %% Load training dataloader

training_dataset, training_dataloader = load_xydataset_from_file(training_data_path, dtype=dtype, shuffle=shuffle)

# %% Load test dataloader

test_dataset, test_dataloader = load_xydataset_from_file(test_data_path, dtype=dtype)
