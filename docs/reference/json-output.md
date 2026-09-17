# JSON output contracts

Pipeline driver commands reserve standard output for compact JSON. Logging goes
to standard error, so automation should parse stdout and preserve stderr for
operators. Each object occupies exactly one line.

## Slurm submission

`lan-sbatch generate`, `lan-sbatch jaxtrain`, `lan-sbatch torchtrain` and
`lan-sbatch recover` emit one object per generated script (one per selected
resource lane; for `recover`, one per design × likelihood cell). When
`--script-only` is used, `job_id` is null because no Slurm submission occurs:

```json
{
  "command": "generate --config-path /repo/config.yaml --output /shared/data ...",
  "job_id": 9876543,
  "mlflow_experiment_id": "42",
  "mlflow_run_id": null,
  "sbatch_script": "/shared/data/runs/20260822T120000000000_ddm_generate_sbatch.sh",
  "output_path": "/shared/data",
  "account": "example-condo",
  "partition": "batch",
  "array_size": 10,
  "lane": 0,
  "n_lanes": 1,
  "lineage_id": "96d9bb5f055b4881bf166336eecd4b6d"
}
```

| Field | Type | Contract |
| --- | --- | --- |
| `command` | string | Fully assembled downstream command embedded in the script |
| `job_id` | integer or null | Parsed Slurm job ID; null for `--script-only` or submission failure |
| `mlflow_experiment_id` | string or null | Generation/training experiment when initialization succeeded |
| `mlflow_run_id` | string or null | Parent training run; null for generation and `--script-only` |
| `sbatch_script` | string | Timestamped generated script path |
| `output_path` | string | Absolute data/network output root |
| `account`, `partition` | string | Resolved lane destination |
| `array_size` | integer | Tasks in this lane's array |
| `lane` | integer | Zero-based lane index |
| `n_lanes` | integer | Number of submissions produced by this invocation |
| `lineage_id` | string or null | Lineage id rendered into the job (MLflow tag `lineage_id`); null only for `--script-only` without an experiment or flag |

## Experiment scaffold

`lan-sbatch init` prints one object:

```json
{"experiment_dir": "/repo/experiments/ddm-pilot",
 "lineage_id": "96d9bb5f055b4881bf166336eecd4b6d",
 "experiments": {"data_generation": "ddm-data-generation",
                 "training": "ddm-training",
                 "inference": "ddm-inference"}}
```

`--use-all-lanes` emits one line per lane. Consumers must read the stream rather
than assume one object. If any `sbatch` call fails, already-submitted lanes are
not rolled back; the process exits non-zero after emitting every result.

`--script-only` is side-effect-free with respect to Slurm and MLflow. It still
writes and reports one object for every generated script, with
job/run/experiment IDs null. An MLflow initialization error in a real invocation
is logged and submission can continue with null MLflow fields, so automation
that requires lineage must validate them.

## Local run

`lan-sbatch run --local` prints one object after the last stage:

```json
{"ok": true, "lineage_id": "6b29...", "experiment_dir": "/repo/experiments/quick",
 "tracking_uri": "sqlite:////repo/experiments/quick/mlflow.db",
 "state": "/repo/experiments/quick/state.json",
 "phases": ["datagen", "infer", "train"], "n_runs": 4,
 "validation_passed": true, "recovery_verdict": false}
```

On failure: `{"ok": false, "failed_stage": "<stage>", "lineage_id": "..."}` and
a non-zero exit; the stage's own JSON line is in the log. `recovery_verdict`
is the aggregate recovery pass/fail and is `null` when recovery did not run.

## Validation result

The validator writes the detailed report to disk and prints a compact result:

```json
{
  "passed": true,
  "report": "/staged/validation_report.json",
  "gates": {
    "structure": "passed",
    "parity": "skipped",
    "hssm_load": "passed",
    "density": "passed"
  }
}
```

Gate states are `passed`, `failed`, or `skipped`. The process exits non-zero
when aggregate `passed` is false. For promotion, do not rely on that aggregate:
the publisher additionally requires structure, HSSM load, and density to be
present and not skipped.

The detailed report has `schema_version: 1`, artifact/model/network identity,
aggregate `passed`, and a `gates` list whose entries include thresholds, scores,
errors, or skip reasons as applicable.

## Publication result

A successful dry run returns a publication plan without uploading to Hugging
Face or recording a publication run in MLflow. It can still copy artifacts into
the staging directory and write `validation_report.json` there:

```json
{
  "published": false,
  "dry_run": true,
  "model": "ddm",
  "network_type": "lan",
  "hf_repo": "example/HSSM_staging",
  "root_filename": "ddm.onnx",
  "training_run_id": "0123456789abcdef",
  "run_uuid": "run-uuid",
  "staged": ["run-uuid_lan_ddm__network.onnx", "validation_report.json"],
  "gate": "all required gates ran and passed"
}
```

A successful upload adds:

```json
{
  "published": true,
  "hf_url": "<Hugging Face commit URL>",
  "hf_commit": "abc123",
  "hf_commit_verified": true,
  "publish_run_id": "fedcba9876543210"
}
```

The plan fields are present in the real result as well. If head cannot be
verified, `hf_commit_candidate` replaces `hf_commit` and may be null;
`hf_commit_verified` is false.

A handled refusal or gate failure returns `published: false` plus `error` and
any plan fields established before the refusal. A real publication attempt
exits non-zero in that case. A dry run can exit successfully with
`published: false`, because not publishing is its intended outcome.
