# JSON output contracts

Pipeline driver commands reserve standard output for compact JSON. Logging goes
to standard error, so automation should parse stdout and preserve stderr for
operators. Each object occupies exactly one line.

## Slurm submission

`lan-sbatch generate`, `lan-sbatch jaxtrain`, and `lan-sbatch torchtrain` emit
one object per generated script (one per selected resource lane). When
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
  "n_lanes": 1
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

`--use-all-lanes` emits one line per lane. Consumers must read the stream rather
than assume one object. If any `sbatch` call fails, already-submitted lanes are
not rolled back; the process exits non-zero after emitting every result.

`--script-only` is side-effect-free with respect to Slurm and MLflow. It still
writes and reports one object for every generated script, with
job/run/experiment IDs null. An MLflow initialization error in a real invocation
is logged and submission can continue with null MLflow fields, so automation
that requires lineage must validate them.

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
    "density": "passed",
    "mass_survey": "skipped"
  }
}
```

Gate states are `passed`, `failed`, or `skipped`. The process exits non-zero
when aggregate `passed` is false. For promotion, do not rely on that aggregate:
the publisher additionally requires structure, HSSM load, and density to be
present and not skipped. `mass_survey` is advisory: it is `skipped` with the
reason `lanfactory.derive not available; this environment's LANfactory predates
it` when the installed LANfactory has no `derive` package, and a skip there
never blocks a publish. Once it runs, a `failed` mass survey refuses a publish
like any other failed gate, and a warn-level result is `passed` with a
`warning` detail in the report.

An auxiliary network (`cpn`, `opn`) reports a different gate set
after `structure` and `parity`:

```json
{
  "passed": true,
  "report": "/staged/validation_report.json",
  "gates": {
    "structure": "passed",
    "parity": "skipped",
    "hssm_missing_load": "passed",
    "accuracy": "passed"
  }
}
```

A cpn reports `hssm_missing_load` as `skipped` under HSSM < 0.6.0, whose
missing-data path ignores `response`; that skip records the `hssm_version` it
saw.

Auxiliary reports are validator-only for now. The publisher still requires the
LAN gate set (`structure`, `hssm_load`, `density`) and does not pass
`--aux-category`, so it refuses every cpn/opn report and cannot yet
validate a cpn at all. Teaching it the auxiliary gate set is a separate change.

The detailed report has `schema_version: 1`, artifact/model/network identity
(`onnx`, `model`, `network_type`, and `aux_category` — what the output is the
probability of: `null` for a LAN, `choice` for a cpn, `omission` for an
opn), aggregate `passed`, and a
`gates` list whose entries include thresholds, scores, errors, or skip reasons
as applicable. Adding a nullable top-level key does not bump `schema_version`;
the gate set a report carries is keyed on `network_type`, not on the version,
so a consumer that reads gates by name must look at `network_type` first.

In the LAN gates, `mass_survey` records its `verdict` (`pass`, `warn`, or
`fail`), the three numbers it judged — `p99_abs_dev`, `frac_gt_0.10`, and
`frac_gt_0.05` — the lines it judged them against (`p99_max`,
`frac_gt_0_10_max`, `warn_p99`, `warn_frac_gt_0_05`), `n_theta`, the
`seconds` the survey took, and the whole `survey` dict as
`lanfactory.derive.survey` returned it: `n_theta`, `grid`, `seconds`, `total`
and `shrunk_box` (each with `mean`, `p50_abs_dev`, `p90_abs_dev`,
`p99_abs_dev`, `min`, `max`, `frac_gt_0.02`, `frac_gt_0.05`, `frac_gt_0.10`;
`shrunk_box` adds `frac_of_theta`), `leak_below_onset` (`mean`, `p99`, `max`,
or `null` for a model without an onset parameter), `by_param` (ten bins per
parameter with `lo`, `hi`, `mean_dev`, `max_abs_dev`, `n`), and `worst_cell`.
A warn-level result adds `warning`; a failure adds `error` and keeps the
survey. A skip carries only `skipped` and `reason`.

In the auxiliary gates, `hssm_missing_load` records `initial_logp_by_p_outlier`
(keys `"0.0"` and `"0.05"`), `n_trials`, `n_missing`, and the `lan` it was
assembled with; `accuracy` records `mean_abs_error`, `max_abs_error`, the two
thresholds it was judged against, `n_param_draws`, `n_core_draws`,
`n_edge_draws`, the `shrink` that separates the strata, `n_sim`, and one
`draws` entry per parameter draw with `theta`, its `stratum` (`core`: from
the shrunk box; `edge`: from the full box outside it), the base LAN's
`total_mass` at that θ (`null` unless `--lan-onnx` was given and
`lanfactory.derive` is importable), the `choice` or `deadline` fed to the
network, `network_logp`, `network_value`, `truth`, `truth_mc_se`, `abs_error`,
and for a cpn `truth_rt_lt_max_t`. When an output is not a log-probability the
gate fails at once with `error`, the index `draw`, and that draw's `theta`,
`stratum`, `total_mass`, and `choice`/`deadline` in place of the `draws` list.

## Parameter-recovery shard

After its arguments are accepted, the recovery worker writes one shard and
prints one object:

```json
{
  "shard": "results/recovery_ddm_sdv_approx_differentiable@candidate_L1_n500@v_0007.json",
  "error": null
}
```

The on-disk shard uses `schema_version: 2` and records model, design,
dataset/seed, likelihood and arm identity, prior and outlier choices, ONNX path,
data sanity checks, sampler settings and diagnostics, per-parameter recovery
summaries, posterior correlations, and environment versions. A failed fit still
writes a shard with its identity plus an `error` string, prints that error in
the compact object, and exits non-zero.

## Parameter-recovery report

The aggregator writes a schema-version-2 report and prints a compact verdict:

```json
{
  "passed": true,
  "report": "results/recovery_report.json",
  "n_shards": 24,
  "n_usable_fits": 24,
  "n_errored_shards": 0,
  "failures": []
}
```

The report records all thresholds, cell summaries, errored shards, coverage
failures, and aggregate `passed`. `n_shards` includes errored shards, whereas
`n_usable_fits` is the sum of `n_converged` across the reported parameter cells,
not a count of distinct shard files. The process exits non-zero when `passed`
is false and also rejects an empty shard directory; too few eligible fits and
failed calibration gates therefore cannot be mistaken for a clean sweep.

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
