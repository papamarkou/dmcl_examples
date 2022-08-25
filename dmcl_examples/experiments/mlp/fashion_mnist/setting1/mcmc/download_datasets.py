# %% Import packages

import torchvision.datasets as datasets

from dmcl_examples.datasets import data_root

# %% Download training dataset

datasets.FashionMNIST(root=data_root, train=True, download=True, transform=None)

# %% Download test dataset

datasets.FashionMNIST(root=data_root, train=False, download=True, transform=None)
