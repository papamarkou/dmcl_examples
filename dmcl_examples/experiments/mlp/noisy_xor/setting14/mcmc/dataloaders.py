# %% Import packages

from torch.utils.data import DataLoader

from dmcl_examples.datasets import load_xydataset_from_file
from dmcl_examples.datasets.noisy_xor.data2.constants import test_data_path, training_data_path
from dmcl_examples.experiments.mlp.noisy_xor.setting14.constants import dtype
from dmcl_examples.experiments.mlp.noisy_xor.setting14.mcmc.batch_sampler import BatchSampler
from dmcl_examples.experiments.mlp.noisy_xor.setting14.mcmc.constants import batch_size, shuffle

# %% Load training dataloader

training_dataset, _ = load_xydataset_from_file(training_data_path, dtype=dtype, shuffle=shuffle)

training_batch_sampler = BatchSampler(training_dataset, batch_size)

training_dataloader = DataLoader(training_dataset, batch_sampler=training_batch_sampler)

# %% Load test dataloader

test_dataset, test_dataloader = load_xydataset_from_file(test_data_path, dtype=dtype)
