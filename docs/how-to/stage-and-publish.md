# Stage and publish a validated network

`lan-publish` resolves one completed training run, copies only that run's
artifacts into an isolated directory, runs the validation gate, uploads to a
non-production Hugging Face repository, and records the result in MLflow.

The [Promotion safety](../explanations/promotion-safety.md) explanation describes
why the staging, validation, and destination guardrails fail closed.

The command is intended to run on an operator machine, not a compute node. Keep
the Hugging Face credential there and point MLflow at the authoritative store.

## Select the training run explicitly

Prefer an MLflow run ID when promoting a known candidate:

```bash
export MLFLOW_TRACKING_URI="sqlite:////shared/path/mlflow/tracking.db"

uv run lan-publish \
  --hf-repo your-org/HSSM_staging \
  --run-id "$TRAINING_RUN_ID" \
  --artifact-dir /local/path/to/training/artifacts \
  --staging-dir /local/path/to/staged-candidate \
  --dry-run
```

You may instead pass `--model` and `--network-type`; the command takes the most
recent matching run with a `run_uuid`. That convenience is useful for
exploration, but an explicit run ID makes an audited promotion unambiguous.

If the training output path recorded in MLflow exists locally,
`--artifact-dir` can be omitted. Cluster paths commonly do not, so fetch the
artifacts first and name the local source explicitly.

## Why staging is mandatory

LANfactory writes multiple training runs into a flat model directory. The
publisher uses the selected run's `run_uuid` to copy its files into an isolated
staging directory, then requires exactly one ONNX there. This prevents three
silent errors:

- uploading files from a different run;
- matching the wrong trainer state during parity validation;
- writing publication-generated files back into the training output.

The entire staging directory is uploaded. If a persistent `--staging-dir`
contains unrelated files, the command refuses to proceed. It preserves files
from its own earlier dry run or failed gate so you can inspect the report.

## Read the dry-run plan

A successful dry run prints one JSON object with `dry_run: true` and
`published: false`. Review:

- `training_run_id` and `run_uuid`;
- `model`, `network_type`, and target `hf_repo`;
- the complete `staged` filename list;
- the canonical `root_filename` HSSM will request;
- `gate`, which must say all required gates ran and passed.

`--skip-density` can make a dry rehearsal faster, but its gate result cannot
authorize a real upload.

## Publish to staging

After reviewing the dry run, repeat the command without `--dry-run`:

```bash
uv run lan-publish \
  --hf-repo your-org/HSSM_staging \
  --run-id "$TRAINING_RUN_ID" \
  --artifact-dir /local/path/to/training/artifacts \
  --staging-dir /local/path/to/staged-candidate
```

If the target already has the canonical root artifact, replacement requires an
explicit `--overwrite-root`. That flag is consequential: HSSM consumers of the
target resolve the root name.

The CLI refuses `franklab/HSSM` by default, including capitalization and
trailing slash variants. That repository is the production source used by
released HSSM versions. Promoting to it requires `--allow-production` and
retyping the repo id when prompted -- a separate, deliberate, non-scriptable
action.

## Verify the records

On success, the JSON result contains the Hugging Face URL, the publication
MLflow run ID, and a commit SHA only when the uploader can verify that the
repository head matches this upload. An unverified read-back is labeled
`hf_commit_candidate`, never `hf_commit`.

The publication experiment records the source training run, source `run_uuid`,
validation report and scores, target repository, and upload result. The training
run receives back-reference tags when the tracking store allows writes.
