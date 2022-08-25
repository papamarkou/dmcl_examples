# %% Import packages

import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dmcl_examples.datasets import data_root
from dmcl_examples.experiments.mlp.mnist.setting6.constants import output_path

# %% Set random seed

torch.manual_seed(6)

# %% Create training dataset

training_dataset = datasets.MNIST(
    root=data_root,
    train=True,
    download=False,
    transform=transforms.RandomInvert(p=0.5)
)

# %% Define paths

output_img_dirname = output_path.joinpath('data', 'images')
output_target_dirname = output_path.joinpath('data', 'targets')

# %% Create output directories if they do not exist

output_img_dirname.mkdir(parents=True, exist_ok=True)
output_target_dirname.mkdir(parents=True, exist_ok=True)

# %% Save augmented data in files

for idx, (img, target) in enumerate(training_dataset):
    torch.save(img, output_img_dirname.joinpath(f"{idx}.pt"))
    torch.save(target, output_target_dirname.joinpath(f"{idx}.pt"))
