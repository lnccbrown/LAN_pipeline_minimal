# Generate and submit Slurm jobs

`lan-sbatch` renders a job script, stores it under the output tree, and normally
submits it with `sbatch`. Rehearse every new config and resource plan with
`--script-only` first.

## Put output and tracking on shared storage

On Oscar, choose one personal tree on the lab data volume:

```bash
export LAN_PIPELINE_ROOT="/oscar/data/frankmj/$USER/proj_hssm_pipeline"
export MLFLOW_TRACKING_URI="sqlite:///$LAN_PIPELINE_ROOT/mlflow/mlflow.db"
export MLFLOW_ARTIFACT_LOCATION="$LAN_PIPELINE_ROOT/mlflow/artifacts"
```

Use an absolute tracking URI for cluster work. The orchestrator converts a
relative SQLite URI to an absolute path before embedding it, but an explicit
shared path is easier to audit and avoids accidental per-directory stores.

## Rehearse data generation

```bash
uv run lan-sbatch generate \
  --config-path configs/examples/data_generation.yaml \
  --output-path "$LAN_PIPELINE_ROOT/data" \
  --cluster-config configs/cluster/oscar.yaml \
  --n-jobs-in-array 10 \
  --n-files 20 \
  --script-only
```

`--script-only` writes the script and returns its JSON plan without calling
`sbatch` or creating MLflow state. Inspect the file named by `sbatch_script`.
In particular, confirm:

- checkout path and `uv run` command;
- account, partition, array size, cores, memory, GPU request, and wall time;
- module loads;
- output and error paths under `<output-path>/runs/`.

Generated scripts and Slurm logs are timestamped, so a later invocation does
not overwrite the earlier plan.

## Submit and capture the handoff

Remove `--script-only` only after review:

```bash
set -o pipefail

uv run lan-sbatch generate \
  --config-path configs/examples/data_generation.yaml \
  --output-path "$LAN_PIPELINE_ROOT/data" \
  --cluster-config configs/cluster/oscar.yaml \
  --n-jobs-in-array 10 \
  --n-files 20 \
  | tee submission.jsonl

jq -r '.job_id' submission.jsonl
jq -r '.mlflow_experiment_id' submission.jsonl
```

Logging goes to standard error. Standard output is reserved for one JSON line
per submission, so a shell driver can capture it without suppressing logs.
Check `job_id` before chaining work; a failed `sbatch` call returns non-zero and
reports `null` for that field.

## Use multiple CPU lanes when throughput matters

```bash
uv run lan-sbatch generate \
  --config-path configs/examples/data_generation.yaml \
  --output-path "$LAN_PIPELINE_ROOT/data" \
  --cluster-config configs/cluster/oscar.yaml \
  --n-jobs-in-array 100 \
  --use-all-lanes
```

Fan-out is opt-in and emits one JSON line per lane. Each line includes `lane`,
`n_lanes`, `array_size`, `account`, and `partition`. If any lane fails to
submit, the command exits non-zero even though jobs on other lanes may already
be running. Preserve every line and reconcile the partial submission before
retrying.

Lower-priority lanes are opportunistic capacity: they may queue much longer
than a condo lane. Do not use `--use-all-lanes` when predictable start time is
more important than aggregate throughput.

## Submit training with lineage

After all generation tasks finish, submit one trainer:

```bash
export EXPERIMENT_ID="<generation-experiment-id>"

uv run lan-sbatch jaxtrain \
  --config-path configs/examples/network_training_lan.yaml \
  --output-path "$LAN_PIPELINE_ROOT/networks" \
  --training-data-folder "$LAN_PIPELINE_ROOT/data/training_data/lan/.../ddm" \
  --data-generation-experiment-id "$EXPERIMENT_ID" \
  --cluster-config configs/cluster/oscar.yaml
```

Use `torchtrain` in place of `jaxtrain` for a Torch training config. The
orchestrator creates one training MLflow run before submission and passes its
run ID into the compute job so LANfactory continues the same record.

## Avoid two Slurm parsing traps

- Quote wall times in YAML: write `time: "12:00:00"`. YAML 1.1 can parse an
  unquoted colon-separated value as an integer, which Slurm then interprets as
  minutes. The CLI rejects that ambiguity.
- Keep output paths free of whitespace. `#SBATCH --output` and `--error` values
  are literal; quoting cannot make a space-bearing path safe.
