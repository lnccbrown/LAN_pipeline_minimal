#!/bin/bash

# Example SBATCH script produced by sbatch_scripts/gen_sbatch.py
# (paths shown as placeholders). Regenerate with the matching gen_sbatch.py command.

#SBATCH --account=default
#SBATCH -p gpu --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -J ddm_generate_sbatch
#SBATCH --time=00:30:00
#SBATCH --output=ddm_generate_sbatch.out
#SBATCH --error=ddm_generate_sbatch.err
#SBATCH --array=1-10

module load python
module load gcc

# MLflow environment variables
export MLFLOW_EXPERIMENT_NAME=ddm-data-generation
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

pip install uv
python -m uv run generate --config-path configs/examples/data_generation.yaml --log-level WARNING --output /path/to/output --mlflow-run-name ddm-worker-$SLURM_ARRAY_TASK_ID --mlflow-experiment-name ddm-data-generation
