# %% Import packages

import matplotlib.pyplot as plt
import torch

from dmcl_examples.experiments.mlp.mnist.setting5.constants import output_path

# %% Set data paths

output_img_dirname = output_path.joinpath('data', 'images')
output_target_dirname = output_path.joinpath('data', 'targets')

# %% Load augmented data from files

training_data = []
training_targets = torch.empty([60000], dtype=torch.int64)

for idx in range(60000):
    training_data.append(torch.load(output_img_dirname.joinpath(f"{idx}.pt")))
    training_targets[idx] = torch.load(output_target_dirname.joinpath(f"{idx}.pt"))

# %% Load a single image

plt.imshow(training_data[0], interpolation='none')

# %% Long six images

plt.figure()
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.tight_layout()
  plt.imshow(training_data[i], interpolation='none')
  plt.title("Ground Truth: {}".format(training_targets[i]))
  plt.xticks([])
  plt.yticks([])
