# %% Import packages

import numpy as np
import torch
from torchvision import datasets

from torch.nn.functional import one_hot

from eeyore.datasets import XYDataset

from dmcl_examples.datasets import data_root
from dmcl_examples.experiments.mlp.fashion_mnist.setting4.constants import dtype, output_path

# %% Set data paths

output_img_dirname = output_path.joinpath('data', 'images')
output_target_dirname = output_path.joinpath('data', 'targets')

# %% Load augmented data from files

training_data = torch.empty([60000, 28, 28], dtype=torch.uint8)
training_targets = torch.empty([60000], dtype=torch.int64)

for idx, _ in enumerate(training_data):
    training_data[idx, :, :] = torch.from_numpy(np.array(torch.load(output_img_dirname.joinpath(f"{idx}.pt"))))
    training_targets[idx] = torch.load(output_target_dirname.joinpath(f"{idx}.pt"))

# %% Create training dataset

training_dataset = XYDataset(
    training_data.to(dtype).reshape(training_data.shape[0], training_data.shape[1]*training_data.shape[2]),
    training_targets.to(dtype)[:, None]
)

training_dataset.x = (training_dataset.x - training_dataset.x.mean()) / training_dataset.x.std()

training_dataset.y = one_hot(training_dataset.y.squeeze(-1).long()).to(training_dataset.y.dtype)

# %% Create test dataset

test_dataset = datasets.MNIST(root=data_root, train=False, download=False)

test_dataset = XYDataset(
    test_dataset.data.to(dtype).reshape(
        test_dataset.data.shape[0],
        test_dataset.data.shape[1]*test_dataset.data.shape[2]
    ),
    test_dataset.targets.to(dtype)[:, None]
)

test_dataset.x = (test_dataset.x - test_dataset.x.mean()) / test_dataset.x.std()

test_dataset.y = one_hot(test_dataset.y.squeeze(-1).long()).to(test_dataset.y.dtype)

# %% Create test dataset subsets

test_dataset_subset01 = torch.utils.data.Subset(test_dataset, range(   0,  2000))
test_dataset_subset02 = torch.utils.data.Subset(test_dataset, range(2000,  4000))
test_dataset_subset03 = torch.utils.data.Subset(test_dataset, range(4000,  6000))
test_dataset_subset04 = torch.utils.data.Subset(test_dataset, range(6000,  8000))
test_dataset_subset05 = torch.utils.data.Subset(test_dataset, range(8000, 10000))
