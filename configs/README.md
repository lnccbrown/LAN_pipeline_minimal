# Configuration Files

This folder contains YAML configuration files for the LAN pipeline.

## Folder Structure

```
configs/
├── examples/           # Production-ready example configs
├── quick_test/         # Fast testing configs (~1-2 min runtime)
└── legacy/             # Archived bash workflow configs (deprecated)
```

## Configuration Files

### Data Generation (`data_generation.yaml`)

Used by `ssm-simulators` to generate training data.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MODEL` | Model type to simulate | `'ddm'`, `'angle'`, `'weibull'` |
| `GENERATOR_APPROACH` | Network type | `'lan'`, `'cpn'` |
| `PIPELINE.N_PARAMETER_SETS` | Number of parameter combinations | `1000` |
| `PIPELINE.N_SUBRUNS` | Subruns per parameter set | `20` |
| `SIMULATOR.N_SAMPLES` | Samples per simulation | `20000` |
| `SIMULATOR.DELTA_T` | Time step | `0.001` |
| `TRAINING.N_SAMPLES_PER_PARAM` | Training samples per parameter set | `2000` |
| `ESTIMATOR.TYPE` | Likelihood estimator | `'kde'` |

### Network Training (`network_training_*.yaml`)

Used by `LANfactory` to train neural networks.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `NETWORK_TYPE` | Network architecture | `'lan'`, `'cpn'`, `'opn'` |
| `MODEL` | Model the network approximates | `'ddm'` |
| `N_EPOCHS` | Training epochs | `20` |
| `LAYER_SIZES` | List of layer configurations | `[[100, 100, 1]]` |
| `ACTIVATIONS` | Activation functions | `[['tanh', 'tanh']]` |
| `CPU_BATCH_SIZE` | Batch size for CPU | `1000` |
| `GPU_BATCH_SIZE` | Batch size for GPU | `50000` |
| `LEARNING_RATE` | Initial learning rate | `0.001` |
| `LR_SCHEDULER` | Learning rate scheduler | `'reduce_on_plateau'` |
| `TRAIN_VAL_SPLIT` | Train/validation split ratio | `0.98` |

## Quick Test Configs

The `quick_test/` folder contains minimal configs for fast local testing:

- **Data generation**: ~10-30 seconds
- **Network training**: ~30-60 seconds

Use these for validating the pipeline works before running production jobs.

## Usage Examples

```bash
# Data generation with quick test config
uv run generate \
    --config-path configs/quick_test/data_generation.yaml \
    --output ./test_data

# Network training with production config
uv run jaxtrain \
    --config-path configs/examples/network_training_lan.yaml \
    --training-data-folder ./data/ddm \
    --networks-path-base ./networks

# Generate SBATCH script
uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path /shared/data
```

## Legacy Configs

The `legacy/` folder contains bash-style configuration files from an older workflow.
These are kept for reference but are **not used** by the current `gen_sbatch.py` workflow.
