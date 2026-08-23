# Environment variables

The pipeline keeps personal paths and credentials out of committed YAML. Set
runtime locations in the shell that invokes the orchestrator; generated jobs
receive the MLflow variables they need.

## Operator-set variables

| Variable | Used by | Default/requirement |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `lan-sbatch`, MLflow clients, `lan-publish` | `lan-sbatch` defaults to `sqlite:///mlflow.db`; set explicitly for publication and cluster work |
| `MLFLOW_ARTIFACT_LOCATION` | Experiment creation and publication records | Optional MLflow-managed location; use an absolute shared path on a cluster |
| `INSPECT_ONNX` | marimo network inspector | Required; path to the candidate ONNX |
| `INSPECT_MODEL` | marimo network inspector | `ddm` |

Hugging Face authentication is handled by `huggingface_hub`. Authenticate on
the operator machine using its supported credential mechanism; never put a
token in pipeline YAML, a generated script, or documentation.

## Documentation convention

The guides use `LAN_PIPELINE_ROOT` as a shell convenience:

```bash
export LAN_PIPELINE_ROOT="/oscar/data/frankmj/$USER/proj_hssm_pipeline"
```

No Python code reads this name. The shell expands it in `--output-path`,
`--training-data-folder`, and MLflow variables before the CLI runs. You may use
another name or pass literal paths.

## Variables injected into jobs

When MLflow initialization succeeds and the command is not `--script-only`, the
generated script exports:

| Variable | Value |
| --- | --- |
| `MLFLOW_TRACKING_URI` | The configured URI; a relative SQLite URI is converted to an absolute path |
| `MLFLOW_ARTIFACT_LOCATION` | The configured artifact root, when set |
| `MLFLOW_EXPERIMENT_NAME` | `{model}-data-generation` or `{model}-training` |

Generation also receives a run-name argument containing
`$SLURM_ARRAY_TASK_ID`; a fanned-out submission adds the lane index so names do
not collide between arrays. The generated shell permits expansion only for its
allowlisted Slurm identifiers and quotes all other config-derived text.

## Path rules

- The submit host and compute nodes must resolve the same MLflow tracking and
  artifact locations.
- Prefer an absolute SQLite URI for shared work. `sqlite:////path/to/file.db`
  has four slashes because the final slash begins an absolute POSIX path.
- Keep Slurm output roots free of whitespace; scheduler directives cannot quote
  those paths safely.
- Keep large data, artifacts, and tracking stores off small home volumes.
- The generated script changes into the checkout that produced it, then adds
  `$HOME/.local/bin` to `PATH` to find a standard uv installation.

## Precedence

MLflow-aware downstream commands generally resolve an explicit CLI argument
before an environment variable and then a package default. The orchestrator's
scheduling resources follow a separate YAML/CLI precedence documented in the
[configuration reference](configuration.md#personal-overlay-and-precedence).
