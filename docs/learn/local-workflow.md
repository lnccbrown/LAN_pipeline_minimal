# Run the CI-backed local workflow

The quickest meaningful check is `local_test_run.sh`. GitHub Actions runs this
same script after the unit tests, so the tutorial stays tied to an executable
integration path rather than a separate set of hand-maintained commands.

## What you will run

The quick workflow:

1. installs and invokes ssm-simulators to generate a small DDM dataset;
2. retrieves the data-generation experiment ID from MLflow;
3. invokes LANfactory's JAX trainer with that experiment ID;
4. writes local MLflow metadata and artifacts for inspection.

It takes a few minutes on a typical development machine. The first dependency
sync takes longer because ssm-simulators includes compiled extensions.

## Install the tracked environment

From the repository root:

```bash
uv sync --locked
```

On macOS, install the native prerequisites first:

```bash
brew install gsl libomp
```

On Debian or Ubuntu, install GSL before syncing:

```bash
sudo apt-get update
sudo apt-get install libgsl-dev
```

## Run the workflow

```bash
bash local_test_run.sh
```

!!! warning "The tutorial output is disposable"

    The script removes and recreates `local_test_data/` in the current working
    directory. Run it from the repository root and do not place your own files
    in that directory. It also creates `mlflow.db` and `mlflow_artifacts/`.

A successful run ends with `=== Test Complete ===`. Inspect these outputs:

| Output | What it demonstrates |
| --- | --- |
| `local_test_data/` | Generated training data and the trained network artifacts |
| `mlflow.db` | Generation and training metadata share one tracking store |
| `mlflow_artifacts/` | Run artifacts are separate from tracking metadata |

## Inspect the lineage in MLflow

Start the UI against the same tracking database:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the MLflow UI at localhost port 5000. Find the
`{model}-data-generation` experiment and the `{model}-training` experiment. The
training run should carry the
`data_generation_experiment_id` that points back to the generation experiment.

Stop the UI with `Ctrl+C` when you are done.

## Exercise the production entry point without Slurm

CI also renders both job types without submitting them. Do the same for data
generation:

```bash
uv run python sbatch_scripts/gen_sbatch.py generate \
  --config-path configs/quick_test/data_generation.yaml \
  --output-path /tmp/lan-pipeline-generate \
  --n-jobs-in-array 2 \
  --script-only
```

The command prints one JSON object. Its `sbatch_script` field names the generated
file under `/tmp/lan-pipeline-generate/runs/`. Read that file before moving to a
real cluster; `--script-only` does not submit a job or create an MLflow
experiment.

## Check the repository gates

```bash
uv run pytest tests/ -q
uv run ruff check .
```

The unit tests cover resource resolution, script generation, the validation
gate, and publish staging. Together with `local_test_run.sh`, these are the same
core checks used by the repository's CI workflow.

## Next

Continue with the operator guides to configure personal cluster lanes and shared
MLflow storage. Keep the quick-test configs for rehearsal; copy an example config
and review its scale before scheduling production work.
