# Configuration Files

This folder contains YAML configuration files for the LAN pipeline.

## Folder Structure

```
configs/
├── examples/           # Production-ready example configs
├── quick_test/         # Fast testing configs (~1-2 min runtime)
├── cluster/            # Cluster resource inventories (oscar.yaml)
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

## Cluster Configs (`cluster/oscar.yaml`)

A record of *what we can schedule where*: the condos we hold, their partitions
and limits, which job kinds each suits, and the sbatch defaults to use for
each. Pass it with `--cluster-config` so submissions are made with awareness
of the actual cluster rather than hardcoded guesses:

```bash
uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path /shared/data \
    --cluster-config configs/cluster/oscar.yaml
```

| Section | Purpose |
|---------|---------|
| `condos` | Per-account record: `partitions`, `nodes`, `limits`, `suited_for`, `verified` |
| `job_defaults` | Job kind (`generate`/`jaxtrain`/`torchtrain`) → `account`, `partition`, `cores`, `mem`, `num_gpus`, `time`, optional `modules` |
| `modules` | Modules loaded at the top of every generated script |

**Precedence:** built-in fallbacks < `job_defaults.<job kind>` < explicit CLI
flags. For `modules`, a per-job-kind list overrides the top-level one, and an
explicit empty list (`modules: []`) means "load no modules" — distinct from
omitting the key, which keeps the defaults.

**Quote your wall times.** YAML 1.1 reads an unquoted `time: 12:00:00` as the
integer 43200, which SLURM would interpret as 43200 *minutes*. Write
`time: "12:00:00"`. `gen_sbatch` rejects the unquoted form with an explicit
error rather than submitting a 30-day request.

Entries carry `verified: false` until they are seeded from a live cluster
snapshot (`sacctmgr show associations`, `sinfo`, `scontrol show partition`) —
do not add condos or limits from memory.

## Legacy Configs

The `legacy/` folder contains bash-style configuration files from an older workflow.
These are kept for reference but are **not used** by the current `gen_sbatch.py` workflow.
