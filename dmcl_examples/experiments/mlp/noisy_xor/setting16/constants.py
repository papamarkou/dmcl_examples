# %% Import packages

import torch

from pathlib import Path

# %% Define constants

# output_path = Path.home().joinpath('output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting16')
output_path = Path.home().joinpath('scratch', 'output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting16')
# output_path = Path('/work', 'tc030', 'tc030', 'theodore', 'output', 'dmcl_examples', 'mlp', 'noisy_xor', 'setting16')

num_features = 2

mlp_dims = [num_features, 2, 1]

dtype = torch.float32
