#!/bin/bash

# Local test script for MLflow integration
# Simulates the end-to-end pipeline without submitting to Slurm

set -e

# Setup paths
TEST_DIR="local_test_data"
CONFIG_GEN="user_configs_examples/config_data_generation.yaml"
CONFIG_TRAIN="user_configs_examples/config_network_training_lan.yaml"

echo "=== Starting Local MLflow Integration Test ==="

# 1. Data Generation Phase
echo "[Phase 1] Generating Data..."
# We manually trigger what sbatch_scripts/gen_sbatch.py would generate inside the sbatch script
# BUT, we use gen_sbatch.py --sh-only to get the MLflow Run ID created by the orchestrator

# Create a dummy run via the orchestrator just to get a Run ID (simulating the submission)
# Note: In a real local run, we might just want to run the python commands directly.
# However, to test the linkage, let's try to use the CLI.

# Since gen_sbatch.py submits to sbatch, we can't use it easily for local execution unless we mock sbatch.
# Instead, we will invoke the underlying CLIs directly, which corresponds to what the "worker node" does.

# Start a parent MLflow run for the "Workflow"
export MLFLOW_RUN_ID=$(python3 -c "import mlflow; print(mlflow.start_run(run_name='local_test_workflow').info.run_id)")
echo "Workflow Run ID: $MLFLOW_RUN_ID"

echo "Running ssm-simulators generate..."
# Run generation (this should log to the workflow run or a nested run if we configured it that way)
# Based on implementation: generate.py takes --mlflow-run-id
uv run ssm-simulators/ssms/cli/generate.py \
    --config-path "$CONFIG_GEN" \
    --output "$TEST_DIR" \
    --n-files 1 \
    --log-level INFO \
    --mlflow-run-id "$MLFLOW_RUN_ID"

# 2. Network Training Phase
echo "[Phase 2] Training Network..."

# Find the generated data path (hacky for bash, but predictable from config)
# Assuming default config structure: data/training_data/lan/training_data_n_samples_.../ddm
# Let's find it dynamically
DATA_PATH=$(find "$TEST_DIR" -type d -name "ddm" | head -n 1)
echo "Training on data at: $DATA_PATH"

echo "Running lanfactory jaxtrain..."
uv run LANfactory/src/lanfactory/cli/jax_train.py \
    --config-path "$CONFIG_TRAIN" \
    --training-data-folder "$DATA_PATH" \
    --networks-path-base "$TEST_DIR/networks" \
    --network-id 0 \
    --dl-workers 0 \
    --log-level INFO \
    --mlflow-run-id "$MLFLOW_RUN_ID"

echo "=== Test Complete ==="
echo "Inspect results with: uv run mlflow ui"

