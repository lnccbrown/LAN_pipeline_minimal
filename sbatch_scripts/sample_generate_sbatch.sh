#!/bin/bash

# Example SBATCH script produced by sbatch_scripts/gen_sbatch.py
# (paths shown as placeholders). Regenerate with the matching gen_sbatch.py
# command; the real file is written to <output-path>/runs/<timestamp>_<job>.sh.

#SBATCH --account=carney-frankmj-condo
#SBATCH -p batch --gres=gpu:0
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -J ddm_generate_sbatch
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/output/runs/ddm_generate_sbatch_%A_%a.out
#SBATCH --error=/path/to/output/runs/ddm_generate_sbatch_%A_%a.err
#SBATCH --array=1-10

module load python
module load gcc

# The uv project is resolved from the working directory, and SLURM starts the
# job in whatever directory sbatch was invoked from — $HOME for a driver that
# submits over ssh. Pin it to the checkout that generated this script.
cd /path/to/LAN_pipeline_minimal || exit 1

# MLflow environment variables
export MLFLOW_EXPERIMENT_NAME=ddm-data-generation
export MLFLOW_TRACKING_URI=sqlite:////shared/storage/mlflow/tracking.db

if ! python -m uv --version >/dev/null 2>&1; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    python -m pip install uv
  else
    python -m pip install --user uv
  fi
fi
python -m uv run generate --config-path configs/examples/data_generation.yaml --log-level WARNING --output /path/to/output --mlflow-run-name ddm-worker-"$SLURM_ARRAY_TASK_ID" --mlflow-experiment-name ddm-data-generation
