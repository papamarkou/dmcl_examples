# %% Import packages

import torch.nn

from torchvision import datasets, transforms

from torch.nn.functional import one_hot

from eeyore.datasets import XYDataset

from dmcl_examples.datasets import data_root
from dmcl_examples.experiments.mlp.mnist.setting8.constants import dtype

# %% Create training dataset

# https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/27
# https://discuss.pytorch.org/t/mnist-normalization/49080/2
# https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a

training_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=None)

# print("Mean = ", training_dataset.data.float().mean() / 255)
# print("Std = ", training_dataset.data.float().std() / 255)
# Mean =  tensor(0.1307)
# Std =  tensor(0.3081)

training_dataset = XYDataset(
    training_dataset.data.to(dtype).reshape(
        training_dataset.data.shape[0],
        training_dataset.data.shape[1]*training_dataset.data.shape[2]
    ),
    training_dataset.targets.to(dtype)[:, None]
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
