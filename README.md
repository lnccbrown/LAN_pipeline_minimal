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
├── validation/
│   ├── validate_network.py # Pre-publication gate (G1-G4)
│   ├── recovery_designs.py # Parameter-recovery recipe + the design ladder
│   ├── recover_parameters.py # One fit per invocation -> one shard
│   └── aggregate_recovery.py # Shards -> coverage/contraction report + verdict
├── publish/
│   └── publish_network.py  # Gate, then upload to HuggingFace
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

## Parameter Recovery

Does a trained network actually support inference? Density checks do not answer
that — users fit models, they do not evaluate densities. The recovery harness in
`validation/` is the acceptance test that does, and it is a **recipe meant to
run on any model**, not a script for one. `validation/recovery_designs.py` is
where the recipe is written down; the three ideas it rests on:

**1. Read coverage and contraction together.** Coverage — how often the 94% HDI
contains the truth — tests the *likelihood*. Contraction — posterior sd over
prior sd — tests the *design*.

|                    | narrow posterior           | wide posterior              |
| ------------------ | -------------------------- | --------------------------- |
| **covers truth**   | identifiable, correct      | unidentifiable, but honest  |
| **misses truth**   | **the likelihood is wrong**| wrong and vague             |

Only the bottom-left cell blocks a release. Coverage is gated; contraction is
only ever reported, because a wide-but-covering posterior is the model being
honest about what a hard dataset supports.

**2. Hold a reference that contains no network**, or a failure cannot be
attributed. First choice is the model's *analytical* likelihood on
byte-identical data with identical priors. Most models have none, so the
fallback is the ladder: a design limit relaxes when you add trials or
conditions, and a broken likelihood does not.

**3. Walk a ladder of increasing design complexity**, since weak
identifiability is usually a property of the design. Total trials are held
constant down each column so "not enough data" and "not enough design" stay
distinguishable:

|            | 1 condition | 4 conditions |
| ---------- | ----------- | ------------ |
| 500 total  | `L0_n500`   | `L1_n500`    |
| 2000 total | `L0_n2000`  | `L1_n2000`   |

**Which parameter varies across the conditions is a free choice, and it is the
interesting knob.** Drift is only the default, because it is what experiments
usually manipulate — the ladder is about design structure, not about drift.
Each choice asks a different question, in two ways at once:

- the varying parameter gets **direct experimental leverage** — is it
  recoverable when something actually moves it?
- every *other* parameter is **pooled across all four conditions**, so it is
  constrained by the whole dataset rather than one cell. This is how a
  multi-condition design rescues a parameter it never manipulates.

So if `sv` comes back badly, `L1@v` asks *"does pooling fix `sv`?"* and `L1@sv`
asks *"is `sv` recoverable when we manipulate it?"* — different questions with
different answers, and running both against the same L0 is the point of the
design rather than an abuse of it. Each variant is its own rung, identified as
`L1_n500@sv`, scored separately, and never pooled with another. A shortfall at
L0 is excused if **any** variant at a richer rung recovers the parameter, while
**within** one variant every condition must clear its own floor.

Applying it to a new model needs no code change — the parameter list, bounds and
available likelihood kinds are read from HSSM's own model config:

```bash
# One fit = one shard = one SLURM array task.
uv run --group validate python validation/recover_parameters.py \
  --model angle --design L1_n500 --dataset-index 7 \
  --likelihood approx_differentiable --onnx-path /path/to/angle.onnx \
  --out-dir results/

# Fan back in: coverage, bias, contraction and a verdict per parameter.
uv run --group validate python validation/aggregate_recovery.py --shard-dir results/
```

```bash
# The same rung, manipulating sv instead of drift. Writes L1_n500@sv shards,
# which are kept separate from L1_n500@v everywhere.
uv run --group validate python validation/recover_parameters.py \
  --model ddm_sdv --design L1_n500 --condition-param sv --dataset-index 7 ...
```

### Two settings that silently decide the answer

**`--p-outlier` defaults to `None` here, not to HSSM's `0.05`.** HSSM's default
fits `0.95·f(rt|θ) + 0.05·Uniform(0, 20)`, and the simulator generates no lapse
process at all. That mismatch is a *fixed* misspecification, so the bias in the
posterior mean stays put while the posterior sd shrinks as 1/√n — coverage gets
**worse as the dataset grows**, which reads exactly like a broken likelihood.
Measured on plain `ddm` with its exact analytical likelihood, 12 datasets:

| n | `p_outlier` | coverage | mean \|z\| |
| --- | --- | --- | --- |
| 500 | 0.05 | 0.88 | 0.96 |
| 500 | none | **0.96** | 0.62 |
| 2000 | 0.05 | 0.77 | 1.24 |
| 2000 | none | **0.90** | 0.91 |

Set it to a real value only when the data really were generated with lapses.
Either way it is recorded in every shard.

**`--bounds-from` decides the priors for *every* arm**, defaulting to the
network's box. Arms that differ in their priors as well as their likelihoods
are not paired, and the whole comparison rests on the pairing.

### What the gate will and will not say

A cell needs at least 10 converged fits to be judged at all; below that it is
**inconclusive**, which is not a pass. An arm that was attempted and produced
nothing usable fails — a sweep in which every task crashed must not come back
green. And because the run fails if any single cell fails, the coverage floor
carries a Šidák correction across the number of cells: uncorrected, a perfectly
calibrated network fails this gate 53% of the time.

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
