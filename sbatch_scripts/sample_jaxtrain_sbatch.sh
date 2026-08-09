#!/bin/bash

# Example SBATCH script produced by sbatch_scripts/gen_sbatch.py
# (paths shown as placeholders). Regenerate with the matching gen_sbatch.py command.

#SBATCH --account=default
#SBATCH -p gpu --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -J ddm_jaxtrain_sbatch
#SBATCH --time=00:30:00
#SBATCH --output=ddm_jaxtrain_sbatch_%A_%a.out
#SBATCH --error=ddm_jaxtrain_sbatch_%A_%a.err
#SBATCH --array=1-1

module load python
module load gcc

# MLflow environment variables
export MLFLOW_EXPERIMENT_NAME=ddm-training
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

if ! python -m uv --version >/dev/null 2>&1; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    python -m pip install uv
  else
    python -m pip install --user uv
  fi
fi
python -m uv run jaxtrain --config-path configs/examples/network_training_lan.yaml --log-level WARNING --networks-path-base /path/to/networks --training-data-folder /path/to/data --network-id 0 --dl-workers 1 --mlflow-run-id PARENT_RUN_ID
