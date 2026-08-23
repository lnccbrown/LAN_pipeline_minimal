# Data lineage and run identity

A network is meaningful only together with the simulation, configuration, code,
and training run that produced it. The pipeline carries several identifiers
because each one answers a different question.

For the operator procedure, see
[Track runs and preserve lineage with MLflow](../how-to/track-with-mlflow.md).

## The lineage chain

```text
uv.lock + generation config
            |
            v
data-generation experiment
  |-- worker run 1 -> generated file inventory
  |-- worker run 2 -> generated file inventory
  `-- worker run N -> generated file inventory
            |
            | data_generation_experiment_id
            v
training run -> run_uuid -> run-named artifact set
            |
            v
validation_report.json
            |
            v
publication run -> Hugging Face URL + verified commit (when available)
```

## Why training links to an experiment

One data-generation submission can create many worker runs, and multi-lane
fan-out can create several Slurm arrays. The training dataset is their union.
A single worker run ID would therefore name only a fragment of the input.

`data_generation_experiment_id` names the collection. LANfactory can aggregate
worker inventories from that experiment and compare them with the files in the
training folder. Preserve the ID printed by `lan-sbatch generate` and pass it to
the training command.

## Why artifacts link to `run_uuid`

LANfactory stores multiple runs for a model in one flat directory. File names
carry a `run_uuid`, but JAX and Torch exporters place it in different positions.
The publisher matches it anywhere in the filename and copies that set into an
isolated directory.

The MLflow status is not a safe completion signal. The submission process ends
its handle after `sbatch` returns, before the compute job may have started. A
training run without `run_uuid` has not reached the artifact-producing stage and
is ineligible for publication.

## Which record is authoritative

| Question | Record to trust |
| --- | --- |
| What dependency revisions ran? | The checkout's tracked `uv.lock` |
| Which Slurm submission landed? | Each submission JSON object's `job_id`, account, partition, and script path |
| Which workers formed the data source? | Runs and file inventories in the generation experiment |
| Which generation collection trained the network? | `data_generation_experiment_id` on the training run |
| Which files belong to the training run? | The run's `run_uuid` and matching artifact names |
| Which checks ran on the candidate? | `validation_report.json`, including each gate's skipped state |
| What was uploaded? | The publication run and Hugging Face URL |
| Which remote revision is confirmed? | `hf_commit` only when `hf_commit_verified` is true |

An `hf_commit_candidate` is deliberately weaker than `hf_commit`: another
writer may have moved repository head between upload and read-back.

## Preserve the chain

- Keep every JSON line from a multi-lane submission; partial success is still
  work running on the cluster.
- Point all stages at one authoritative MLflow store rather than a disposable
  mirror.
- Treat copied artifact folders as inputs: preserve names containing the
  `run_uuid`.
- Keep the validation report beside the staged artifacts that produced it.
- Do not infer lineage from timestamps or "latest" when a stable identifier is
  available.
