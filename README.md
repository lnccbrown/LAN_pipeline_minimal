# LAN_pipeline_minimal

Minimal version of the LAN pipeline for internal purposes.

## Installation

We recommend using this pipeline with `uv`. Find installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone and setup
git clone <repo-url>
cd LAN_pipeline_minimal
uv sync
```

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
│   └── legacy/             # Archived old workflow configs
├── sbatch_scripts/
│   ├── gen_sbatch.py       # Main orchestrator script
│   └── sample_*.sh         # Example generated SBATCH scripts
├── local_test_run.sh       # Local end-to-end test script
└── using_mlflow.md         # MLflow integration guide
```

## Quick Start

### Local Testing

Run a quick end-to-end test locally:

```bash
./local_test_run.sh
```

### Generate SBATCH Scripts

The `gen_sbatch.py` script creates SBATCH scripts for Slurm clusters:

```bash
# View available commands
uv run python sbatch_scripts/gen_sbatch.py --help
uv run python sbatch_scripts/gen_sbatch.py generate --help
uv run python sbatch_scripts/gen_sbatch.py jaxtrain --help
uv run python sbatch_scripts/gen_sbatch.py torchtrain --help
```

### Data Generation

```bash
# Generate SBATCH script for data generation
uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path /path/to/output \
    --n-jobs-in-array 10 \
    --partition gpu

# Or run directly (local)
uv run generate \
    --config-path configs/quick_test/data_generation.yaml \
    --output ./data \
    --n-files 5
```

### Network Training

```bash
# Generate SBATCH script for training
uv run python sbatch_scripts/gen_sbatch.py jaxtrain \
    --config-path configs/examples/network_training_lan.yaml \
    --output-path /path/to/networks \
    --training-data-folder /path/to/data \
    --partition gpu

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
```

### Network Training Config

```yaml
NETWORK_TYPE: "lan"
MODEL: "ddm"
N_EPOCHS: 20
LAYER_SIZES: [[100, 100, 100, 1]]
ACTIVATIONS: [['tanh', 'tanh', 'tanh']]
```

## MLflow Integration

This pipeline includes MLflow integration for experiment tracking. See `using_mlflow.md` for details.

Key features:
- Automatic experiment organization by model name
- Data lineage tracking between generation and training
- Works with both local SQLite and remote MLflow servers

```bash
# View MLflow UI after running experiments
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
