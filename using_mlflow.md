# MLflow Integration Summary

## Overview

This document summarizes the MLflow integration across the LAN pipeline ecosystem:

| Package | MLflow Status | Role |
|---------|--------------|------|
| `ssm-simulators` | **Optional** | Data generation with tracking |
| `LANfactory` | **Optional** | Network training with tracking and data lineage |
| `LAN_pipeline_minimal` | **Required** | Orchestration and experiment management |

**Tracking Backend**: SQLite (default: `sqlite:///mlflow.db`)
- Production-ready, future-proof tracking backend
- Configurable via environment variables or CLI arguments
- Supports shared filesystem storage for cluster environments

## Installation

### For Individual Packages (Optional MLflow)

```bash
# ssm-simulators with MLflow support
pip install ssm-simulators[mlflow]

# LANfactory with MLflow support
pip install lanfactory[mlflow]
```

### For Orchestrator (MLflow Required)

```bash
# Using uv (recommended)
cd LAN_pipeline_minimal
uv sync

# Or pip
pip install lan-pipeline-minimal
```

This design allows:
- ✅ Individual packages work standalone without MLflow
- ✅ Orchestrator manages MLflow for coordinated workflows
- ✅ Users choose when to enable tracking features

## Architecture

### Experiment Organization

```
SQLite Database (sqlite:///mlflow.db or custom path)
│
├── Metadata Storage:
│   ├── Experiments
│   ├── Runs
│   ├── Parameters
│   ├── Metrics
│   └── Tags
│
└── Artifacts (filesystem or cloud storage):
    │
    ├── {model_name}-data-generation/         # Experiment for data generation
    │   ├── Run 1: {model}-worker-1           # Individual distributed worker
    │   ├── Run 2: {model}-worker-2
    │   └── Run N: {model}-worker-N
    │       └── Artifacts:
    │           └── generated_files_inventory.json  # List of files created
    │
    └── {model_name}-training/                # Experiment for training
        └── Run: jaxtrain_network_0
            ├── Tags:
            │   └── data_generation_experiment_id: {exp_id}  # Links to data gen
            ├── Artifacts:
            │   ├── training_output/
            │   │   ├── training_history.csv
            │   │   ├── train_state_*.pickle
            │   │   ├── train_config.json
            │   │   └── network_config.json
            │   └── training_data_lineage.json  # Detailed file tracking
            └── Metrics:
                ├── train_loss (per epoch)
                └── test_loss (per epoch)
```

### Data Lineage Model

**Key Concept**: Training runs link to the **data generation experiment ID**, not individual run IDs.

This allows:
1. Multiple distributed data generation runs → Single training run
2. Automatic aggregation of all files from the data generation experiment
3. Verification that expected files are present in training folder
4. Detection of missing or extra files

## CLI Reference

### ssm-simulators: `generate` command

```bash
uv run generate --help
```

**MLflow-related arguments:**

| Argument | Description |
|----------|-------------|
| `--mlflow-run-name` | Human-readable name for this run. Enables MLflow tracking. |
| `--mlflow-experiment-name` | Experiment name to log to. |
| `--mlflow-tracking-uri` | MLflow tracking URI. Defaults to `MLFLOW_TRACKING_URI` env var or `sqlite:///mlflow.db`. |
| `--mlflow-artifact-location` | Root directory for artifacts. Defaults to `MLFLOW_ARTIFACT_LOCATION` env var. |
| `--dry-run` | Validate pipeline without saving data. |

**Example:**
```bash
uv run generate \
    --config-path config.yaml \
    --output ./data/output \
    --mlflow-run-name "worker-1" \
    --mlflow-experiment-name "ddm-data-generation"
```

### LANfactory: `jaxtrain` / `torchtrain` commands

```bash
uv run jaxtrain --help
uv run torchtrain --help
```

**MLflow-related arguments:**

| Argument | Description |
|----------|-------------|
| `--mlflow-run-name` | Human-readable name for a NEW run. Enables MLflow tracking. |
| `--mlflow-experiment-name` | Experiment name. Defaults to `MLFLOW_EXPERIMENT_NAME` env var. |
| `--mlflow-run-id` | (Advanced) Resume an existing run by UUID. |
| `--data-generation-experiment-id` | Link to data generation experiment for lineage tracking. |
| `--dry-run` | Validate pipeline without training. |

**Example:**
```bash
uv run jaxtrain \
    --config-path config.yaml \
    --training-data-folder ./data/output \
    --networks-path-base ./networks \
    --data-generation-experiment-id "123456789"
```

### LAN_pipeline_minimal: `gen_sbatch.py`

The orchestrator creates SBATCH scripts and manages MLflow experiments.

**For data generation:**
```bash
uv run sbatch_scripts/gen_sbatch.py generate \
    --config-path config.yaml \
    --output-path ./data \
    --n-jobs-in-array 10
```

**For training:**
```bash
uv run sbatch_scripts/gen_sbatch.py jaxtrain \
    --config-path config.yaml \
    --output-path ./networks \
    --training-data-folder ./data \
    --data-generation-experiment-id "123456789"
```

## Configuration

### Environment Variables

All components respect the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | Where MLflow stores metadata | `sqlite:///mlflow.db` |
| `MLFLOW_ARTIFACT_LOCATION` | Where MLflow stores artifacts | MLflow-managed |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name for the current run | Set by orchestrator |

### Configuration Priority

For each setting, the priority order is:
1. CLI argument (highest priority)
2. Environment variable
3. Default value (lowest priority)

### Cluster Configuration Example

For distributed computation on a Slurm cluster with shared filesystem:

```bash
# In your environment setup or batch script
export MLFLOW_TRACKING_URI="sqlite:////shared/storage/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/storage/mlflow/artifacts"

# Then run orchestrator
uv run sbatch_scripts/gen_sbatch.py generate \
    --config-path config.yaml \
    --output-path /shared/data
```

The `gen_sbatch.py` orchestrator will:
1. Read these environment variables
2. Create experiments with the specified artifact location
3. Inject these settings into generated SBATCH scripts
4. Pass tracking URI to all worker processes

## Workflow: How the Pieces Fit Together

### Data Generation Flow

```
┌─────────────────────┐
│   gen_sbatch.py     │  1. Creates experiment "{model}-data-generation"
│   (orchestrator)    │  2. Does NOT create a parent run
└─────────┬───────────┘  3. Injects env vars + CLI args into SBATCH
          │
          ▼
┌─────────────────────┐
│   SBATCH Worker 1   │  4. Each worker calls: generate --mlflow-run-name "worker-$ID"
│   SBATCH Worker 2   │  5. Each worker creates its OWN MLflow run
│   SBATCH Worker N   │  6. Each logs its generated files to artifacts
└─────────────────────┘
```

### Training Flow

```
┌─────────────────────┐
│   gen_sbatch.py     │  1. Creates experiment "{model}-training"
│   (orchestrator)    │  2. Creates parent run, gets run_id
└─────────┬───────────┘  3. Injects --mlflow-run-id into SBATCH
          │
          ▼
┌─────────────────────┐
│   SBATCH Worker     │  4. Calls: jaxtrain --mlflow-run-id {run_id}
│                     │              --data-generation-experiment-id {exp_id}
│                     │  5. Continues logging to parent run
│                     │  6. Logs training metrics + data lineage
└─────────────────────┘
```

### Key Difference: `--mlflow-run-name` vs `--mlflow-run-id`

| Argument | Creates New Run? | Use Case |
|----------|------------------|----------|
| `--mlflow-run-name` | ✅ Yes | Data generation workers (each creates own run) |
| `--mlflow-run-id` | ❌ No (resumes) | Training (continues orchestrator's parent run) |

## Workflow Examples

### Example 1: Local Development (Single Machine)

```bash
# 1. Set local SQLite tracking
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"

# 2. Generate data
uv run generate \
    --config-path user_configs_examples/config_data_generation.yaml \
    --output ./data/output \
    --n-files 5 \
    --mlflow-experiment-name "ddm-data-generation" \
    --mlflow-run-name "local-worker-1"

# Note the experiment ID from output (e.g., "123456789")

# 3. Train network with lineage tracking
uv run jaxtrain \
    --config-path user_configs_examples/config_network_training_lan.yaml \
    --networks-path-base ./networks \
    --training-data-folder ./data/output/training_data/lan/.../ddm \
    --data-generation-experiment-id "123456789"
```

### Example 2: Using Orchestrator (Slurm Cluster)

```bash
cd LAN_pipeline_minimal

# Set shared SQLite tracking (absolute path required for cluster)
export MLFLOW_TRACKING_URI="sqlite:////shared/storage/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/storage/mlflow/artifacts"

# 1. Generate data (10 distributed workers)
uv run sbatch_scripts/gen_sbatch.py generate \
    --config-path user_configs_examples/config_data_generation.yaml \
    --output-path /shared/data/output \
    --n-jobs-in-array 10 \
    --partition gpu \
    --num-gpus 1

# Output shows:
# ============================================================
# DATA GENERATION EXPERIMENT ID: 123456789
# Use this ID with training commands via --data-generation-experiment-id
# ============================================================

# 2. Train network (after data generation completes)
uv run sbatch_scripts/gen_sbatch.py jaxtrain \
    --config-path user_configs_examples/config_network_training_lan.yaml \
    --output-path /shared/networks/output \
    --training-data-folder /shared/data/output/training_data/lan/.../ddm \
    --data-generation-experiment-id 123456789 \
    --partition gpu \
    --num-gpus 1
```

### Example 3: Viewing Results

```bash
# Start MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open http://localhost:5000 in browser
```

### Example 4: Querying Best Models (Python)

```python
import mlflow
import pickle

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Find best model by test loss
runs = mlflow.search_runs(
    experiment_names=["ddm-training"],
    order_by=["metrics.test_loss ASC"],
    max_results=1
)
best_run_id = runs.iloc[0]["run_id"]

# Download model artifacts
client = mlflow.MlflowClient()
artifact_path = client.download_artifacts(
    run_id=best_run_id,
    path="training_output/train_state_best.pickle"
)
with open(artifact_path, 'rb') as f:
    model = pickle.load(f)

# Check data lineage
run = client.get_run(best_run_id)
data_gen_exp_id = run.data.tags.get("data_generation_experiment_id")
print(f"Model trained from data generation experiment: {data_gen_exp_id}")
```

## Testing

### Run Local Test Script

```bash
cd LAN_pipeline_minimal
./local_test_run.sh
```

### Run Package Test Suites

```bash
# ssm-simulators
cd ssm-simulators
pytest tests/test_mlflow_integration.py -v

# LANfactory
cd LANfactory
pytest tests/test_mlflow_integration.py -v
```

## Troubleshooting

### Issue: "mlflow-run-id" not recognized by ssm-simulators

**Cause**: `ssm-simulators` uses `--mlflow-run-name` (creates new run), not `--mlflow-run-id`.

**Solution**: Use `--mlflow-run-name "worker-name"` for data generation.

### Issue: Runs not showing up in experiment

**Cause**: Experiment not set before starting run.

**Solution**: Ensure environment variables are set:
```bash
export MLFLOW_EXPERIMENT_NAME="my-experiment"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

### Issue: Workers can't find each other's runs (cluster)

**Cause**: Workers using different tracking URIs.

**Solution**: Use absolute path for shared SQLite:
```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/path/mlflow.db"  # Note: 4 slashes for absolute path
```

### Issue: Missing files in lineage

**Solution**:
1. Check that all data generation workers completed successfully
2. Verify `generated_files_inventory.json` exists in each run's artifacts
3. Review logs for warnings about missing files

### Issue: "No data source provided" error in training

**Cause**: Neither `--training-data-folder` nor `--data-generation-experiment-id` provided.

**Solution**: Provide at least one:
```bash
# Option 1: Direct path
uv run jaxtrain --training-data-folder ./data/path ...

# Option 2: MLflow-derived (requires proper lineage setup)
uv run jaxtrain --data-generation-experiment-id "123456789" ...
```

## Benefits of This Integration

1. **Complete Provenance**: Track exactly which data files went into each model
2. **Distributed Coordination**: Multiple workers → single coherent experiment
3. **Reproducibility**: All parameters, configs, and metrics logged
4. **Verification**: Automatic detection of missing/extra data files
5. **Simplified Management**: Single tool for tracking
6. **Queryable**: Use MLflow API/UI to find best models, compare runs
7. **Artifact Storage**: Centralized storage of models, configs, histories
8. **Flexible Activation**: MLflow enabled implicitly when needed (no explicit `--mlflow-on` flag)
