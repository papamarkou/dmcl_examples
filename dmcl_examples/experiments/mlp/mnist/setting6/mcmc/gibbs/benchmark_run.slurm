#!/bin/bash

#SBATCH --job-name=mnist_setting6
#SBATCH --time=95:59:00
#SBATCH --ntasks=10
#SBATCH --account=tc030-theodore
#SBATCH --partition=standard
#SBATCH --qos=standard

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Enforce threading to 1 in case underlying libraries are threaded
export OMP_NUM_THREADS=1

source /work/tc030/tc030/theodore/opt/continuum/miniconda/miniconda3/etc/profile.d/conda.sh

# /work/tc030/tc030/theodore/opt/continuum/miniconda/miniconda3/bin/conda activate dmcl

conda activate dmcl

srun --cpu-bind=cores --ntasks=1 python /work/tc030/tc030/theodore/opt/python/packages/dmcl_examples/dmcl_examples/experiments/mlp/mnist/setting6/mcmc/gibbs/benchmark_run.py
