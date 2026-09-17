# Configuration reference

The repository carries four configuration classes. Generation and training
YAML are consumed by the upstream scientific packages. Cluster YAML is consumed
by `lan-sbatch` to render Slurm resources. An experiment directory ties one
generation config, one training config and one lineage id together.

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
experiments/               # created by `lan-sbatch init`, one directory per experiment
  <name>/
    experiment.yaml
    data_generation.yaml
    network_training.yaml
    state.json             # written by the stages, gitignored
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

## Experiment YAML

`lan-sbatch init` writes `experiment.yaml` (schema_version 1). Every key has a
default; the file it writes is the full form.

```yaml
schema_version: 1
name: ddm-pilot
model: ddm                      # must equal MODEL in both stage configs
lineage_id: 96d9bb5f...         # uuid4 hex minted by init; shared by every stage
network:
  type: lan                     # lan | cpn | opn | gonogo; must equal NETWORK_TYPE
  trainer: jaxtrain             # jaxtrain | torchtrain
  network_id: 0
mlflow:
  experiments:                  # "{model}" is interpolated
    data_generation: "{model}-data-generation"
    training: "{model}-training"
    inference: "{model}-inference"
  tracking_uri: null            # null -> $MLFLOW_TRACKING_URI
  artifact_location: null       # null -> $MLFLOW_ARTIFACT_LOCATION; ignored for http servers
  tags: {}                      # extra MLflow tags; reserved keys are rejected
stages:
  generate: {config: data_generation.yaml, output: data, n_jobs_in_array: 1, n_files: null}
  train:    {config: network_training.yaml, output: networks, dl_workers: 1}
  validate: {skip_density: false, skip_hssm: false}
  recover:
    designs: [L0_n250]          # names from validation/recovery_designs.py
    likelihoods: [approx_differentiable]
    add_reference_arm: true     # also fit `analytical` when the model has it
    n_datasets: 1               # array size per design x likelihood cell
    draws: 200
    tune: 200
    chains: 2
    target_accept: 0.9
    p_outlier: null
    condition_param: null
    out_dir: recovery
```

Paths are relative to the experiment directory. Loading validates the schema
version, the `MODEL`/`NETWORK_TYPE` agreement with the stage configs, the
network type and trainer, the design names, and that `mlflow.tags` sets none of
`schema_version`, `phase`, `lineage_id`. `state.json` is the stages' scratch
record (`data_generation_experiment_id`, `training_data_folder`, `onnx_path`,
`mlflow_run_id_train`, ...); delete it to start the chain over.

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
