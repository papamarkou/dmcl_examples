# %% Import packages

from torch.utils.data import DataLoader

from dmcl_examples.experiments.mlp.fashion_mnist.setting9.mcmc.batch_sampler import BatchSampler
from dmcl_examples.experiments.mlp.fashion_mnist.setting9.mcmc.constants import batch_size
from dmcl_examples.experiments.mlp.fashion_mnist.setting9.mcmc.load_datasets import (
    training_dataset, test_dataset,
    test_dataset_subset01, test_dataset_subset02, test_dataset_subset03, test_dataset_subset04, test_dataset_subset05
)

# %% Create training dataloader

training_batch_sampler = BatchSampler(training_dataset, batch_size)

training_dataloader = DataLoader(training_dataset, batch_sampler=training_batch_sampler)

# %% Create test dataloader

test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# %% Create test subset dataloaders

test_dataloader_subset01 = DataLoader(test_dataset_subset01, batch_size=len(test_dataset_subset01), shuffle=False)
test_dataloader_subset02 = DataLoader(test_dataset_subset02, batch_size=len(test_dataset_subset02), shuffle=False)
test_dataloader_subset03 = DataLoader(test_dataset_subset03, batch_size=len(test_dataset_subset03), shuffle=False)
test_dataloader_subset04 = DataLoader(test_dataset_subset04, batch_size=len(test_dataset_subset04), shuffle=False)
test_dataloader_subset05 = DataLoader(test_dataset_subset05, batch_size=len(test_dataset_subset05), shuffle=False)
