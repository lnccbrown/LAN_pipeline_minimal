# Track runs and preserve lineage with MLflow

The pipeline uses MLflow to connect distributed generation workers, one
training run, and a later publication record. Set the tracking and artifact
locations before invoking the orchestrator; the generated job inherits them.
For the rationale behind its experiment- and run-level identifiers, read
[Data lineage and run identity](../explanations/data-lineage.md).

## Choose one authoritative store

For local work, the defaults are sufficient:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="./mlflow_artifacts"
```

For cluster work, both locations must be visible from the submit host and every
compute node:

```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/path/mlflow/tracking.db"
export MLFLOW_ARTIFACT_LOCATION="/shared/path/mlflow/artifacts"
```

Use four slashes for an absolute SQLite URI: the fourth begins the filesystem
path. Ensure the shared filesystem supports the locking and concurrency pattern
of your chosen MLflow backend.

!!! warning "A mirror is not authoritative"

    `lan-publish` writes a publication run and back-references to the tracking
    URI active on the operator machine. If that URI points at a disposable copy
    downloaded from the cluster, those records disappear on the next refresh.
    Point publication at the store whose history must survive.

## Use a tracking server

With a shared `mlflow server` (the lab's central store), set only the tracking
URI; the server owns artifact storage and a client-side
`MLFLOW_ARTIFACT_LOCATION` is ignored with a warning:

```bash
export MLFLOW_TRACKING_URI="http://<mlflow-host>:5000"
```

## Carry one lineage id through every stage

Schema v2 of the ecosystem's MLflow run schema adds a `lineage_id` tag that the
data, the network and every fit share. An experiment directory (`lan-sbatch
init`) mints it once and every `--experiment` submission renders it; a plain
`lan-sbatch generate` mints one on submission and reports it in the JSON line.
Pass `--lineage-id` explicitly to add workers to an existing dataset. Training
inherits the id from the training pickles even when the flag is omitted.

`data_generation_experiment_id` (below) is still recorded: it names the
*collection* of worker runs, while `lineage_id` names the dataset across
stages. Query one lineage with `tags.lineage_id = '<id>'`.

## Understand generation identity

For `generate`, the orchestrator creates or reuses `{model}-data-generation` but
does not create a parent run. Each Slurm array worker starts its own MLflow run
with a name derived from the task ID. The submission JSON returns the shared
`mlflow_experiment_id`.

With multi-lane generation, run names include a lane index because every array
numbers tasks from one. All lanes still write into the same experiment.

Preserve the experiment ID from every successful submission. The training
handoff is experiment-level because a dataset can be the union of files from
many worker runs.

## Link training to generation

Pass the experiment ID when rendering or submitting a trainer:

```bash
uv run lan-sbatch jaxtrain \
  --config-path configs/examples/network_training_lan.yaml \
  --output-path /shared/networks \
  --training-data-folder /shared/data/training_data/lan/.../ddm \
  --data-generation-experiment-id "$EXPERIMENT_ID" \
  --cluster-config configs/cluster/oscar.yaml
```

The orchestrator creates a run under `{model}-training` and injects
`--mlflow-run-id` into the generated LANfactory command. The compute job resumes
that run. `data_generation_experiment_id` is also recorded on it, letting
LANfactory compare the training folder with generation inventories.

Do not infer training completion from the MLflow run status alone. The submitter
ends its local handle after `sbatch` returns, so the record can read `FINISHED`
before the compute job starts. Publication treats the LANfactory `run_uuid` tag
as evidence that training reached artifact creation.

## Inspect the store

```bash
uv run mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

For a candidate network, verify:

- the generation experiment contains the expected worker runs and inventories;
- the training run has the intended model, network type, config, and output path;
- `data_generation_experiment_id` points at the expected generation experiment;
- `run_uuid` exists before selecting artifacts for promotion.
