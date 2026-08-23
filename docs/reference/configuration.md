# Configuration reference

The repository carries three configuration classes. Generation and training
YAML are consumed by the upstream scientific packages. Cluster YAML is consumed
by `lan-sbatch` to render Slurm resources.

## Layout

```text
configs/
  examples/
    data_generation.yaml
    network_training_lan.yaml
    network_training_cpn.yaml
  quick_test/
    data_generation.yaml
    network_training.yaml
  cluster/
    oscar.yaml
    oscar.local.yaml       # generated, gitignored, personal
```

`quick_test/` is the executable smoke-test profile used by
`local_test_run.sh` and CI. `examples/` starts at a much larger scale and must be
reviewed for the intended model, network, storage budget, and allocation before
submission.

## Data-generation YAML

The included files expose this shape:

| Key | Example | Meaning |
| --- | --- | --- |
| `MODEL` | `ddm` | ssm-simulators model name; also determines experiment/job naming |
| `GENERATOR_APPROACH` | `lan` | Training-data generator approach |
| `PIPELINE.N_PARAMETER_SETS` | `1000` | Parameter combinations generated |
| `PIPELINE.N_SUBRUNS` | `20` | Subruns per parameter set |
| `SIMULATOR.N_SAMPLES` | `20000` | Simulation samples per parameter set |
| `SIMULATOR.DELTA_T` | `0.001` | Simulator time step |
| `TRAINING.N_SAMPLES_PER_PARAM` | `2000` | Samples retained per parameter set for training |
| `ESTIMATOR.TYPE` | `kde` | Likelihood-target estimator |

`lan-sbatch` reads `MODEL` itself and forwards the full file to
ssm-simulators. Consult the ssm-simulators documentation for the scientific
configuration supported by the locked revision.

Scale is multiplicative. Review parameter sets, subruns, simulator samples,
files per worker, and array size together rather than changing one in isolation.

## Network-training YAML

| Key | Example | Meaning |
| --- | --- | --- |
| `NETWORK_TYPE` | `lan` | Network family (`lan`, `cpn`, or another trainer-supported type) |
| `MODEL` | `ddm` | Model whose target the network learns |
| `GENERATOR_APPROACH` | `lan` | Layout/approach of the training data |
| `N_EPOCHS` | `20` | Maximum training epochs |
| `LAYER_SIZES` | `[[100, 100, 100, 1]]` | Candidate network architectures; `--network-id` selects one |
| `ACTIVATIONS` | `[['tanh', 'tanh', 'tanh']]` | Hidden activations corresponding to each architecture |
| `CPU_BATCH_SIZE` | `1000` | CPU batch size |
| `GPU_BATCH_SIZE` | `50000` | GPU batch size |
| `N_TRAINING_FILES` | `10000` | Files selected for training; LANfactory also accepts supported list forms |
| `TRAIN_VAL_SPLIT` | `0.98` | Training share of the data |
| `SHUFFLE` | `true` | Shuffle training data |
| `OPTIMIZER_` | `adam` | Optimizer name expected by LANfactory |
| `LEARNING_RATE` | `0.001` | Initial learning rate |
| `LR_SCHEDULER` | `reduce_on_plateau` | Scheduler name |
| `LR_SCHEDULER_PARAMS` | mapping | Scheduler parameters such as factor, patience, threshold, and minimum rate |
| `WEIGHT_DECAY` | `0.0` | Optimizer weight decay |
| `LABELS_LOWER_BOUND` | `np.log(1e-7)` | Lower clipping expression/value understood by the trainer config parser |
| `TRAINING_DATA_FOLDER` | path or empty | Data path; the orchestrator overrides it with `--training-data-folder` |

`lan-sbatch` reads `MODEL`, uses the resource flags, and forwards the training
file to LANfactory. LANfactory owns the complete trainer schema and how each
network type interprets it.

## Cluster YAML

The committed file separates human-readable inventory from executable defaults.

| Section | Consumer | Purpose |
| --- | --- | --- |
| `condos` | humans | Accounts, partitions, QOS, node classes, known limits, suitability, and verification date |
| `job_defaults.generate` | `lan-sbatch` | CPU generation account, partition, cores, memory, GPU count, and wall time |
| `job_defaults.jaxtrain` | `lan-sbatch` | JAX training resources |
| `job_defaults.torchtrain` | `lan-sbatch` | Torch training resources |
| `modules` | generated script | Top-level module load list |

Accepted executable resource keys are `account`, `partition`, `cores`, `mem`,
`num_gpus`, and `time`. A job-specific `modules` list overrides the top-level
list. An explicit empty list means load no modules; an omitted key inherits the
less-specific default.

For generation, `job_defaults.generate.lanes` may list:

```yaml
lanes:
  - account: example-condo
    partition: batch
    max_cores: 128
    priority: 10000
```

`--use-all-lanes` sorts usable lanes by priority/capacity and divides the array
approximately in proportion to `max_cores`.

## Personal overlay and precedence

`scripts/discover_cluster.py` writes `<name>.local.yaml` beside the committed
cluster file. Loading `oscar.yaml` merges that local file automatically. Within
`job_defaults`, per-kind mappings merge rather than replacing the entire
section.

```text
built-in fallback < oscar.yaml < oscar.local.yaml < explicit CLI flag
```

Quote every wall time:

```yaml
time: "12:00:00"
```

An unquoted colon-separated value can become a YAML integer and would be
ambiguous in Slurm minutes. Resource validation rejects it rather than guessing.
