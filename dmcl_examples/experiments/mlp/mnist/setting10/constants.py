# %% Import packages

import torch

from pathlib import Path

# %% Define constants

# output_path = Path.home().joinpath('output', 'dmcl_examples', 'mlp', 'mnist', 'setting10')
output_path = Path.home().joinpath('scratch', 'output', 'dmcl_examples', 'mlp', 'mnist', 'setting10')
# output_path = Path('/work', 'tc030', 'tc030', 'theodore', 'output', 'dmcl_examples', 'mlp', 'mnist', 'setting10')

num_features = 784 # 28*28
num_classes = 10

mlp_dims = [num_features, 10, 10, 10, num_classes]
mlp_bias = [True, True, True, True]
mlp_activations = [torch.sigmoid, torch.sigmoid, torch.sigmoid, None]

dtype = torch.float32
