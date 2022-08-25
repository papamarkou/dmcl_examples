# %% Import packages

from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.constants import num_epochs, num_burnin_epochs
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.dataloaders import training_dataloader
from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.gibbs.constants import sampler_output_run_paths
# from dmcl_examples.experiments.mlp.fashion_mnist.setting1.mcmc.gibbs.sampler import sampler
from dmcl_examples.io import extract_cols_from_csv

# %% Notes on how the columns have been chosen

# model.num_params()
# 8180

# blocks = sampler.get_blocks()
# len(blocks)
# 40

# Each of the first 10 blocks are split into 78 sub-blocks
# So the total number of blocks becomes 78 * 10 + 10 + 10 + 10 = 810

# awk "NR == 100" acceptance_rates.txt 
# 7.912799999999999834e-01
# awk "NR == 783" acceptance_rates.txt 
# 4.632100000000000106e-01
# awk "NR == 795" acceptance_rates.txt 
# 4.465199999999999725e-01
# awk "NR == 804" acceptance_rates.txt 
# 5.111900000000000333e-01

# len(blocks[0][2])
# 78
# len(blocks[1][2])
# 78

# blocks[1][2][21] # Pick one chain from block 100
# [999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]

# blocks[12] # Pick one chain from block 783
# [1, 2, [[7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7952]]]

# blocks[24] # Pick one chain from block 795
# [2, 4, [[8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8064]]]

# blocks[33] # Pick one chain from block 804
# [3, 3, [[8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8173]]]

# For example, pick
# 1004, 7873, 8007, 8106

# %% Extract and save columns

extract_cols_from_csv(
    sampler_output_run_paths[0].joinpath('sample.csv'),
    sampler_output_run_paths[0].joinpath('sample_extract.csv'),
    # cols=[1004, 7873, 8007, 8106],
    cols=[
        999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
        7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7952,
        8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8064,
        8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8173
    ],
    header=None,
    dtype='str',
    mode='a',
    chunk_size=100,
    verbose=True,
    num_lines=(num_epochs - num_burnin_epochs) * len(training_dataloader)
)
