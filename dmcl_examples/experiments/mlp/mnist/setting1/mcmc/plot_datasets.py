# %% Import packages

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from dmcl_examples.datasets import data_root

# %% Create training dataset

training_dataset_original = datasets.MNIST(root=data_root, train=True, download=False, transform=None)

# %%

plt.imshow(training_dataset_original[0][0], cmap='gray')

# %%

fig = plt.figure()
for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.tight_layout()
  # plt.imshow(training_dataset_original[i][0], cmap='gray', interpolation='none')
  plt.imshow(training_dataset_original[i][0], interpolation='none')
  plt.title("Ground Truth: {}".format(training_dataset_original[i][1]))
  plt.xticks([])
  plt.yticks([])
fig

# %% Create transformed training dataset 01

training_dataset_tsfm01 = datasets.MNIST(
    root=data_root, train=True, download=False, transform=transforms.RandomRotation(degrees=(0, 30))
)

# %%

fig = plt.figure()
for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.tight_layout()
  plt.imshow(training_dataset_tsfm01[i][0], interpolation='none')
  plt.title("Ground Truth: {}".format(training_dataset_tsfm01[i][1]))
  plt.xticks([])
  plt.yticks([])
fig
