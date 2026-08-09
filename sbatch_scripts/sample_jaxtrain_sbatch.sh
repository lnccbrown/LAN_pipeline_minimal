#!/bin/bash

# Example SBATCH script produced by sbatch_scripts/gen_sbatch.py
# (paths shown as placeholders). Regenerate with the matching gen_sbatch.py
# command; the real file is written to <output-path>/runs/<timestamp>_<job>.sh.

#SBATCH --account=carney-frankmj-condo
#SBATCH -p batch --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -J ddm_jaxtrain_sbatch
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/networks/runs/ddm_jaxtrain_sbatch_%A_%a.out
#SBATCH --error=/path/to/networks/runs/ddm_jaxtrain_sbatch_%A_%a.err
#SBATCH --array=1-1

module load python
module load gcc

# The uv project is resolved from the working directory, and SLURM starts the
# job in whatever directory sbatch was invoked from — $HOME for a driver that
# submits over ssh. Pin it to the checkout that generated this script.
cd /path/to/LAN_pipeline_minimal || exit 1

# MLflow environment variables
export MLFLOW_EXPERIMENT_NAME=ddm-training
export MLFLOW_TRACKING_URI=sqlite:////shared/storage/mlflow/tracking.db

if ! python -m uv --version >/dev/null 2>&1; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    python -m pip install uv
  else
    python -m pip install --user uv
  fi
fi
python -m uv run jaxtrain --config-path configs/examples/network_training_lan.yaml --log-level WARNING --networks-path-base /path/to/networks --training-data-folder /path/to/data --network-id 0 --dl-workers 1 --mlflow-run-id PARENT_RUN_ID
