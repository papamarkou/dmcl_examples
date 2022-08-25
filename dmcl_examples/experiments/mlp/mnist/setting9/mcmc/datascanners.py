# %% Import packages

from torch.utils.data import DataLoader

from dmcl_examples.experiments.mlp.mnist.setting9.mcmc.load_datasets import (
    test_dataset,
    test_dataset_subset01, test_dataset_subset02, test_dataset_subset03, test_dataset_subset04, test_dataset_subset05
)

# %% Load test dataloader with batch size of 1

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% Load test subset dataloader with batch size of 1

test_dataloader_subset01 = DataLoader(test_dataset_subset01, batch_size=1, shuffle=False)
test_dataloader_subset02 = DataLoader(test_dataset_subset02, batch_size=1, shuffle=False)
test_dataloader_subset03 = DataLoader(test_dataset_subset03, batch_size=1, shuffle=False)
test_dataloader_subset04 = DataLoader(test_dataset_subset04, batch_size=1, shuffle=False)
test_dataloader_subset05 = DataLoader(test_dataset_subset05, batch_size=1, shuffle=False)
