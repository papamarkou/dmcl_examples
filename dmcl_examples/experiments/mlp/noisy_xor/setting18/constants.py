# %% Import packages

import torch

from pathlib import Path

# %% Define constants

# output_path = Path.home().joinpath('output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting18')
output_path = Path.home().joinpath('scratch', 'output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting18')
# output_path = Path('/work', 'tc030', 'tc030', 'theodore', 'output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting18')

num_features = 2

mlp_dims = [num_features, 2, 2, 2, 2, 2, 2, 1]
# mlp_dims = [num_features, 2, 2, 2, 2, 2, 1]
# mlp_dims = [num_features, 2, 2, 2, 2, 1]

mlp_bias = 7*[True]
# mlp_bias = 6*[True]
# mlp_bias = 5*[True]

mlp_activations = 7*[torch.sigmoid]
# mlp_activations = 6*[torch.sigmoid]
# mlp_activations = 5*[torch.sigmoid]

dtype = torch.float32
