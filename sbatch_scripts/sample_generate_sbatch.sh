#!/bin/bash

# Example SBATCH script produced by sbatch_scripts/gen_sbatch.py
# (paths shown as placeholders). Regenerate with the matching gen_sbatch.py command.

#SBATCH --account=default
#SBATCH -p batch --gres=gpu:0
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -J ddm_generate_sbatch
#SBATCH --time=02:00:00
#SBATCH --output=ddm_generate_sbatch_%A_%a.out
#SBATCH --error=ddm_generate_sbatch_%A_%a.err
#SBATCH --array=1-10

module load python
module load gcc

# MLflow environment variables
export MLFLOW_EXPERIMENT_NAME=ddm-data-generation
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

if ! python -m uv --version >/dev/null 2>&1; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    python -m pip install uv
  else
    python -m pip install --user uv
  fi
fi
python -m uv run generate --config-path configs/examples/data_generation.yaml --log-level WARNING --output /path/to/output --mlflow-run-name ddm-worker-$SLURM_ARRAY_TASK_ID --mlflow-experiment-name ddm-data-generation
