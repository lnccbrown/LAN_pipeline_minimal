# LAN_pipeline_minimal — Project Context for Claude

## What is LAN_pipeline_minimal?

Lightweight orchestration layer for the HSSM ecosystem's network training pipeline. Chains ssm-simulators (data generation) → LANfactory (network training) using YAML configs, with MLflow for experiment tracking. Can run locally or generate SLURM batch scripts for cluster submission. This repo contains no simulation or training code itself — it wraps the CLIs of ssm-simulators and LANfactory. For ecosystem-wide context, see the HSSMSpine repo.

## Project Structure

```
configs/                           # YAML pipeline configs
  examples/                        # Production configs (large-scale runs)
  quick_test/                      # Fast test configs (~1-2 min)
  legacy/                          # Deprecated bash workflow configs
sbatch_scripts/
  gen_sbatch.py                    # Main orchestrator (the only Python file in the repo)
local_test_run.sh                  # End-to-end local test script
md_files/                          # Development planning docs
using_mlflow.md                    # MLflow integration guide
local_test_data/                   # Generated test outputs (gitignored)
```

## Build & Tooling

- **Package manager:** uv (with `uv.lock`)
- **Python:** >3.10, <3.13
- **Dependencies:** ssm-simulators + lanfactory (both from GitHub main branch, not PyPI) + mlflow
- **Dev dependency:** ruff only
- **No CI workflows** — this is an orchestration tool, not a library
- **No pre-commit hooks**

## Common Commands

```bash
# Install dependencies
uv sync

# Run end-to-end local test (~2-3 min)
bash local_test_run.sh

# Generate SLURM scripts for cluster (data generation)
uv run python sbatch_scripts/gen_sbatch.py generate \
  --config configs/examples/data_generation.yaml \
  --output-folder /path/to/output \
  --account <slurm-account> --partition <partition>

# Generate SLURM scripts for cluster (training)
uv run python sbatch_scripts/gen_sbatch.py jaxtrain \
  --config configs/examples/network_training_lan.yaml \
  --training-data-folder /path/to/data \
  --networks-path-base /path/to/output \
  --account <slurm-account> --partition gpu --gpus 1

# View MLflow experiments
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Pipeline Flow

```
YAML config
    │
    ▼
gen_sbatch.py generate          →  ssm-simulators `generate` CLI
    │                               (produces training data with all data keys:
    │                                lan_data, cpn_data, opn_data)
    │
    │  MLflow tracks: {model}-data-generation experiment
    │  passes data_generation_experiment_id downstream
    │
    ▼
gen_sbatch.py jaxtrain/torchtrain  →  LANfactory `jaxtrain`/`torchtrain` CLI
    │                                   (trains networks from generated data)
    │
    │  MLflow tracks: {model}-training experiment
    │  linked to data generation via experiment ID (lineage)
    │
    ▼
Trained networks (.pt, .onnx)
    │
    ▼
upload-hf (LANfactory CLI)      →  HuggingFace franklab/HSSM
```

## Config System

Two config types drive the pipeline:

### Data Generation Configs

```yaml
MODEL: 'ddm'                    # SSM model to simulate
GENERATOR_APPROACH: 'lan'       # Data generation strategy
PIPELINE:
  N_PARAMETER_SETS: 1000        # Parameter combinations
  N_SUBRUNS: 20                 # Parallel sims per parameter set
SIMULATOR:
  N_SAMPLES: 20000              # Samples per simulation
  DELTA_T: 0.001
TRAINING:
  N_SAMPLES_PER_PARAM: 2000
ESTIMATOR:
  TYPE: 'kde'
```

### Network Training Configs

```yaml
NETWORK_TYPE: "lan"              # lan, cpn, or opn
MODEL: "ddm"
N_EPOCHS: 20
LAYER_SIZES: [[100, 100, 100, 1], ...]   # Multiple architectures
ACTIVATIONS: [['tanh', 'tanh', 'tanh'], ...]
CPU_BATCH_SIZE: 1000
GPU_BATCH_SIZE: 50000
LEARNING_RATE: 0.001
LR_SCHEDULER: 'reduce_on_plateau'
```

For deadline models (e.g., `ddm_deadline`), use the `*_deadline.yaml` config variants.

## Key Conventions

- A single data generation run produces all data keys (lan, cpn, opn) simultaneously.
  Different network types consume different keys from the same data.
- `quick_test/` configs are for fast local validation (~1-2 min).
  `examples/` configs are for production cluster runs.
- MLflow is required (not optional like in ssm-simulators and LANfactory).
  Default backend: SQLite (`sqlite:///mlflow.db`), artifacts in `./mlflow_artifacts/`.
- The `gen_sbatch.py` orchestrator supports three commands: `generate`, `jaxtrain`, `torchtrain`.

## Compaction

When compacting, preserve: the pipeline flow diagram, config structure,
gen_sbatch.py commands, and the MLflow experiment naming convention
(`{model}-data-generation`, `{model}-training`).
