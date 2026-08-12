# LAN_pipeline_minimal

Minimal version of the LAN pipeline for generating training data and training likelihood approximation networks.

## Installation

We recommend using this pipeline with `uv`. Find installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone and setup
git clone <repo-url>
cd LAN_pipeline_minimal
uv sync
```

This installs the required dependencies:
- `ssm-simulators` - Data generation
- `lanfactory` - Network training
- `mlflow` - Experiment tracking

## Project Structure

```
LAN_pipeline_minimal/
├── configs/
│   ├── examples/           # Production-ready configs
│   │   ├── data_generation.yaml
│   │   ├── network_training_lan.yaml
│   │   └── network_training_cpn.yaml
│   ├── quick_test/         # Fast testing configs (~1-2 min)
│   │   ├── data_generation.yaml
│   │   └── network_training.yaml
│   ├── cluster/            # Cluster resource inventories
│   │   └── oscar.yaml      # Condos, limits, per-job-kind defaults
│   └── README.md           # Config documentation
├── sbatch_scripts/
│   └── gen_sbatch.py       # Main orchestrator script
│                           # (for an example script, use --script-only)
├── tests/                  # pytest suite for gen_sbatch + validation gate
├── local_test_run.sh       # Local end-to-end test script
├── using_mlflow.md         # MLflow integration guide
└── pyproject.toml          # Dependencies (from GitHub main branches)
```

## Quick Start

### Local Testing

Run a quick end-to-end test locally (~2-3 min):

```bash
./local_test_run.sh
```

This will:
1. Generate test data with `ssm-simulators`
2. Train a network with `lanfactory`
3. Track everything in MLflow

### Generate SBATCH Scripts

The `gen_sbatch.py` script creates SBATCH scripts for Slurm clusters:

```bash
# View available commands
uv run python sbatch_scripts/gen_sbatch.py --help
uv run python sbatch_scripts/gen_sbatch.py generate --help
uv run python sbatch_scripts/gen_sbatch.py jaxtrain --help
uv run python sbatch_scripts/gen_sbatch.py torchtrain --help
```

## Usage

### Data Generation

```bash
# Generate SBATCH script for data generation (Slurm cluster)
uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path /path/to/output \
    --n-jobs-in-array 10 \
    --n-files 20 \
    --cluster-config configs/cluster/oscar.yaml

# Or run directly (local)
uv run generate \
    --config-path configs/quick_test/data_generation.yaml \
    --output ./data \
    --n-files 5
```

**Where things land, and what comes back.** The generated script and the
SLURM `.out`/`.err` files are written to `<output-path>/runs/`, timestamped —
repeated invocations never overwrite each other. Each invocation prints
exactly one JSON line on stdout (all logging goes to stderr, so this holds at
any `--log-level`):

```json
{"command": "generate --config-path ...", "job_id": 9876543,
 "mlflow_experiment_id": "42", "mlflow_run_id": null,
 "sbatch_script": "/path/to/output/runs/20260809T101500_ddm_generate_sbatch.sh",
 "output_path": "/path/to/output", "account": "carney-mjfrank-condo2",
 "partition": "batch"}
```

That line is the interface for scripted use — `jq -r .job_id` to poll with
`sacct`, `jq -r .mlflow_experiment_id` to chain into training. A failed
submission exits non-zero with `"job_id": null`. `--script-only` writes the
script and prints the same line without submitting, and creates no MLflow
experiment or run — it is also the way to see an example script, since one
generated on demand cannot be out of date the way a checked-in copy can.

**Cluster resources.** `--cluster-config` reads per-job-kind defaults
(account, partition, cores, memory, GPUs, wall time) from a cluster
inventory; see `configs/README.md`. Precedence is built-in fallbacks <
cluster config < explicit flags, so any flag below still wins.

### Network Training

```bash
# Generate SBATCH script for training (Slurm cluster)
uv run python sbatch_scripts/gen_sbatch.py jaxtrain \
    --config-path configs/examples/network_training_lan.yaml \
    --output-path /path/to/networks \
    --training-data-folder /path/to/data \
    --data-generation-experiment-id <exp-id> \
    --cluster-config configs/cluster/oscar.yaml

# Or run directly (local)
uv run jaxtrain \
    --config-path configs/quick_test/network_training.yaml \
    --training-data-folder ./data/ddm \
    --networks-path-base ./networks
```

## Configuration Files

See `configs/README.md` for detailed documentation on configuration options.

### Data Generation Config

```yaml
MODEL: 'ddm'
GENERATOR_APPROACH: 'lan'

PIPELINE:
  N_PARAMETER_SETS: 1000
  N_SUBRUNS: 20

SIMULATOR:
  N_SAMPLES: 20000
  DELTA_T: 0.001

TRAINING:
  N_SAMPLES_PER_PARAM: 2000

ESTIMATOR:
  TYPE: 'kde'
```

### Network Training Config

```yaml
NETWORK_TYPE: "lan"
MODEL: "ddm"
N_EPOCHS: 20
LAYER_SIZES: [[100, 100, 100, 1]]
ACTIVATIONS: [['tanh', 'tanh', 'tanh']]
CPU_BATCH_SIZE: 1000
GPU_BATCH_SIZE: 50000
TRAINING_DATA_FOLDER: ""  # Set via CLI
```

## MLflow Integration

This pipeline includes MLflow integration for experiment tracking. See `using_mlflow.md` for details.

Key features:
- Automatic experiment organization by model name (`{model}-data-generation`, `{model}-training`)
- Data lineage tracking between generation and training via `--data-generation-experiment-id`
- Works with both local SQLite and remote MLflow servers

```bash
# View MLflow UI after running experiments
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Dependencies

This package pulls `ssm-simulators` and `lanfactory` from their GitHub main branches:

```toml
[tool.uv.sources]
lanfactory = { git = "https://github.com/lnccbrown/lanfactory", branch = "main" }
ssm-simulators = { git = "https://github.com/lnccbrown/ssm-simulators", branch = "main" }
```

To update to the latest versions:

```bash
uv sync --refresh
```
