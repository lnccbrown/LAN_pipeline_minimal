#!/bin/bash

# specifies sbatch resources needed for the job

#SBATCH --account=default
#SBATCH -p batch --gres=gpu:0
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -J ddm_generate_sbatch
#SBATCH --time=24:00:00
#SBATCH --output=ddm_generate_sbatch.out
#SBATCH --error=ddm_generate_sbatch.err
#SBATCH --array=1-1

# Your commands here

module load python
module load gcc

python -m uv run generate --config-path /path/to/config_data_generation.yaml --log-level WARNING --output my_generated_data
