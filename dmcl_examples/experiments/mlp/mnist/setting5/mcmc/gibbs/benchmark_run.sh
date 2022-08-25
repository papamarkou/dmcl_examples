#!/bin/bash --login

# Job will run in the current directory (where you ran qsub)
#$ -cwd

# Reserve an 512GB RAM node
#$ -l mem512

# Indicates parallel job, but it is used to reserve 64GBB RAM
#$ -pe smp.pe 2

# Activate conda dmcl environment
conda activate dmcl

# Now the commands to be run by the job
python benchmark_run.py
