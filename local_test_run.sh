#!/bin/bash

# Local test script for MLflow integration
# Simulates the end-to-end pipeline without submitting to Slurm
#
# This script demonstrates the complete workflow:
# 1. Data generation with ssm-simulators (creates MLflow runs)
# 2. Network training with LANfactory (links to data generation via experiment ID)

set -e

# Setup paths - use quick_test configs for fast local testing
TEST_DIR="local_test_data"
CONFIG_GEN="configs/quick_test/data_generation.yaml"
CONFIG_TRAIN="configs/quick_test/network_training.yaml"

# MLflow configuration (uses SQLite by default)
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"

echo "=== Starting Local MLflow Integration Test ==="
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow Artifacts: $MLFLOW_ARTIFACT_LOCATION"

# Clean up previous test data (optional)
if [ -d "$TEST_DIR" ]; then
    echo "Removing previous test data..."
    rm -rf "$TEST_DIR"
fi

# 1. Data Generation Phase
echo ""
echo "[Phase 1] Generating Data..."
echo "Using config: $CONFIG_GEN"

# Read model name from config for experiment naming
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_GEN'))['MODEL'])")
EXPERIMENT_NAME="${MODEL_NAME}-data-generation"
echo "Experiment name: $EXPERIMENT_NAME"

# Run data generation using ssm-simulators CLI
# Each run gets a unique name; in local testing we simulate a single worker
uv run generate \
    --config-path "$CONFIG_GEN" \
    --output "$TEST_DIR" \
    --n-files 2 \
    --log-level INFO \
    --mlflow-run-name "local-worker-1" \
    --mlflow-experiment-name "$EXPERIMENT_NAME"

# Get the experiment ID for linking to training
EXPERIMENT_ID=$(python3 -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
exp = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
print(exp.experiment_id if exp else '')
")

echo ""
echo "Data generation experiment ID: $EXPERIMENT_ID"

# 2. Network Training Phase
echo ""
echo "[Phase 2] Training Network..."
echo "Using config: $CONFIG_TRAIN"

# Find the generated data path
# The structure is: {output}/{data_subdir}/{model_name}
DATA_PATH=$(find "$TEST_DIR" -type d -name "$MODEL_NAME" | head -n 1)

if [ -z "$DATA_PATH" ]; then
    echo "ERROR: Could not find generated data for model '$MODEL_NAME' in '$TEST_DIR'"
    echo "Available directories:"
    find "$TEST_DIR" -type d
    exit 1
fi

echo "Training on data at: $DATA_PATH"

# Run network training using LANfactory CLI
# Link to data generation experiment for lineage tracking
uv run jaxtrain \
    --config-path "$CONFIG_TRAIN" \
    --training-data-folder "$DATA_PATH" \
    --networks-path-base "$TEST_DIR/networks" \
    --network-id 0 \
    --dl-workers 0 \
    --log-level INFO \
    --data-generation-experiment-id "$EXPERIMENT_ID"

echo ""
echo "=== Test Complete ==="
echo ""
echo "To view results in MLflow UI:"
echo "  uv run mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI"
echo ""
echo "Then open http://localhost:5000 in your browser"
