# %% Import packages

from dmcl_examples.datasets import load_xydataset_from_file
from dmcl_examples.datasets.noisy_xor.data2.constants import test_data_path
from dmcl_examples.experiments.mlp.noisy_xor.setting13.constants import dtype

# %% Load test dataloader with batch size of 1

test_dataset, test_dataloader = load_xydataset_from_file(test_data_path, dtype=dtype, batch_size=1)
