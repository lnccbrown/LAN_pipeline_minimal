# Configuration Files

This folder contains YAML configuration files for the LAN pipeline.

## Folder Structure

```
configs/
├── examples/           # Production-ready example configs
├── quick_test/         # Fast testing configs (~1-2 min runtime)
└── cluster/            # Cluster resource inventories (oscar.yaml)
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
| `condos` | Per-account record: `partitions`, `qos`, `nodes`, `limits`, `suited_for`, `verified_on` |
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

Every entry carries `verified_on`, the date it was last read off the cluster
(`sacctmgr show associations user=$USER`, `sacctmgr show qos`, `sinfo`,
`scontrol show partition`) — do not add condos or limits from memory. Note that
the real per-user caps live in the **QOS**, not the partition: `scontrol show
partition` reports `MaxTime=UNLIMITED` for batch, gpu and gpu-he alike.

## Personalising: two files, one committed, one not

Cluster config is split by *who it is true for*:

| file | committed? | holds |
|------|-----------|-------|
| `cluster/oscar.yaml` | yes | the cluster and the lab's condo: partitions, QOS caps, node inventory, modules, per-job-kind resources — identical for every member |
| `cluster/oscar.local.yaml` | **no** (gitignored) | *your* associations, lane caps and choices |

`gen_sbatch` merges the local file over the committed one automatically
whenever you pass `--cluster-config .../oscar.yaml`; there is no second flag.
Precedence end to end:

```
built-in fallbacks  <  oscar.yaml  <  oscar.local.yaml  <  explicit CLI flags
```

That is why no path, quota or account of yours needs to be committed to share
this repo with the rest of the lab.

### Step 1 — discover what you can actually schedule

```bash
# from a login node
uv run python scripts/discover_cluster.py

# or from a laptop, through your ssh config entry
uv run python scripts/discover_cluster.py --ssh-host oscar
```

It asks SLURM which associations you hold and what each one's QOS allows, then
writes `configs/cluster/oscar.local.yaml`. Output looks like:

```
Found 4 lanes:
  <your-condo>               batch      qos=<condo-qos>   priority=10000  208 cores
  <your-condo>               gpu-he     qos=<condo-gqos>  priority=10000  160 cores, 75 gpus
  default                    batch      qos=normal        priority=0      64 cores
  default                    gpu        qos=norm-gpu      priority=0      12 cores, 2 gpus
  -> 272 CPU cores usable for datagen across 2 lane(s)
```

Re-run it whenever your allocations change. Never hand-edit the generated file
— it is overwritten, and hand-edits are exactly the personal values that used
to leak into the repo.

**Why discovery rather than documentation:** the numbers that matter are
`MaxTRESPU` on the *QOS*, and nothing else surfaces them. `scontrol show
partition` reports `MaxTime=UNLIMITED` for every partition on this cluster, so
reading partition limits tells you nothing.

### Step 2 — use every lane you have

A **lane** is one (account, partition) pair you may submit to. SLURM applies
each QOS's cap *per QOS*, so two lanes are two independent budgets and their
capacity genuinely adds up. `--use-all-lanes` splits an array across them in
proportion to their core caps:

```bash
uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path "$LAN_PIPELINE_ROOT/data" \
    --cluster-config configs/cluster/oscar.yaml \
    --n-jobs-in-array 100 \
    --use-all-lanes
```

```json
{"lane": 0, "n_lanes": 2, "account": "<your-condo>", "array_size": 77, "job_id": 9876543, ...}
{"lane": 1, "n_lanes": 2, "account": "default",      "array_size": 23, "job_id": 9876544, ...}
```

One JSON line per lane. Without the flag you get exactly one line and one
submission, as before — fan-out is opt-in because it changes how many jobs
land on the cluster.

**Read the priorities before relying on this.** A condo lane typically has
priority 10000 and the general `default` lane has 0, so spillover tasks queue
behind everyone else's work. Treat the extra lanes as *opportunistic* capacity
— excellent for an overnight sweep, unreliable when you need results in an
hour. If any lane fails to submit, the command exits non-zero and the JSON
lines tell you which ones did land.

Training is deliberately **not** fanned out: a single job cannot use two lanes,
so discovery maps `jaxtrain`/`torchtrain` onto one GPU on your highest-priority
GPU lane instead.

## Where things go on Oscar

The cluster config deliberately holds **no paths** — paths are per-person, the
condo is not. Use this convention and pass it per invocation:

```bash
# The lab's data volume, one tree per user. Home is small (100G) and is the
# wrong place for run output or an MLflow store; the frankmj volume is sized
# for it.
export LAN_PIPELINE_ROOT="/oscar/data/frankmj/$USER/proj_hssm_pipeline"
export MLFLOW_TRACKING_URI="sqlite:///$LAN_PIPELINE_ROOT/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="$LAN_PIPELINE_ROOT/mlflow/artifacts"

uv run python sbatch_scripts/gen_sbatch.py generate \
    --config-path configs/examples/data_generation.yaml \
    --output-path "$LAN_PIPELINE_ROOT/data" \
    --cluster-config configs/cluster/oscar.yaml
```

`gen_sbatch` reads `MLFLOW_TRACKING_URI` from the environment and absolutizes a
relative sqlite path before embedding it in the job script, so every array
worker writes to the same database as the submitting process.

Check your own quotas with `checkquota` before a large run — the shared volume
is sized in TB but is shared across the lab.

## The old bash workflow

The pre-`gen_sbatch.py` bash configs and sbatch scripts used to live in
`configs/legacy/` and `sbatch_scripts/legacy/`. They were removed: nothing
referenced them, everything they did is handled by `gen_sbatch.py` and the
cluster config, and they hardcoded one person's home directory and conda
environment — which reads as an endorsed pattern rather than a dead one.

Git still has them if you need to look:

```bash
git show 7764979:sbatch_scripts/legacy/sbatch_network_training.sh
git log --diff-filter=D -- 'configs/legacy/*' 'sbatch_scripts/legacy/*'
```
