#!/bin/bash

# specifies sbatch resources needed for the job

#SBATCH --account=default
#SBATCH -p gpu --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -J ddm_jaxtrain_sbatch
#SBATCH --time=24:00:00
#SBATCH --output=ddm_jaxtrain_sbatch.out
#SBATCH --error=ddm_jaxtrain_sbatch.err
#SBATCH --array=1-1

# Your commands here

module load python
module load gcc

python -m uv run jaxtrain --config-path /path/to/config_network_train.yaml --log-level WARNING --networks-path-base my_trained_network --training-data-folder my_generated_data/data/training_data/lan/training_data_n_samples_2000_dt_0.001/ddm --network-id 0 --dl-workers 2
