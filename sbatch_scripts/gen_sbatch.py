#!/usr/bin/env python3
"""
Generates and submits individual sbatch job for generating simulated data for a given model,
and/or for training neural network on simulated data.

MLflow Integration:
- For data generation: Each SBATCH array task creates its own MLflow run using --mlflow-run-name
- For training: The orchestrator creates a parent run and passes --mlflow-run-id to continue logging
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import re
import shlex
import subprocess
import typer
import yaml
import os

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Importable both as the installed package (console script) and by path (the
# test suite adds sbatch_scripts/ to sys.path).
try:
    from sbatch_scripts import experiment as xp
except ImportError:  # pragma: no cover - path import in tests
    import experiment as xp

app = typer.Typer(add_completion=False)

# SBATCH template
SBATCH_TEMPLATE = """#!/bin/bash

#SBATCH --account={account}
#SBATCH -p {partition} --gres=gpu:{num_gpus}
#SBATCH -c {cores}
#SBATCH --mem={mem}
#SBATCH -J {job_name}
#SBATCH --time={time}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --array=1-{n_jobs_in_array}

{module_loads}

# The uv project is resolved from the working directory, and SLURM starts the
# job in whatever directory sbatch was invoked from — $HOME for a driver that
# submits over ssh. Pin it to the checkout that generated this script.
cd {project_root} || exit 1

# MLflow environment variables
{env_vars}

# uv's own installer puts it here, and a batch shell does not read the rc file
# that would normally add it.
export PATH="$HOME/.local/bin:$PATH"

resolve_uv() {{
  if command -v uv >/dev/null 2>&1; then UV="uv"; return 0; fi
  if python -m uv --version >/dev/null 2>&1; then UV="python -m uv"; return 0; fi
  return 1
}}

if ! resolve_uv; then
  if python -m pip --version >/dev/null 2>&1; then
    if [ -n "${{VIRTUAL_ENV:-}}" ]; then
      python -m pip install uv
    else
      python -m pip install --user uv
    fi
    hash -r   # a newly installed binary is not in bash's command hash yet
  fi
fi

# Re-resolve rather than assuming the install worked. pip can fail for reasons
# that have nothing to do with this script — compute nodes frequently have no
# outbound network — and assuming success here would land us back at the
# ModuleNotFoundError this whole block exists to prevent.
if ! resolve_uv; then
  echo "uv is not on PATH, not importable, and could not be installed." >&2
  echo "The loaded python is: $(command -v python)" >&2
  echo "Install uv once on the cluster, then resubmit:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

$UV run {uv_run_flags}{command}
"""
# Notes on the uv resolution block above:
# - The binary comes first. Oscar's `module load python` provides a spack
#   Python 3.13 with neither pip nor uv, so `python -m uv` is not a dependable
#   entry point *and* the pip bootstrap has nothing to bootstrap with — the
#   job died on "No module named pip" while a working uv sat in ~/.local/bin.
# - $HOME/.local/bin is prepended because a non-interactive batch shell skips
#   the profile that usually adds it.
# - `python -m pip`, not bare `pip`: after `module load python` the two can
#   resolve to different interpreters, and uv must land in the one that runs
#   the job on the next line.
# - `--user` only outside a virtualenv. pip *hard-errors* on `--user` inside
#   one ("User site-packages are not visible in this virtualenv"), and sbatch
#   propagates the submitting environment by default — so an unconditional
#   `--user` fails for anyone submitting from their project venv.
# - The last branch fails loudly with the install command rather than letting
#   the job die two lines later on a confusing ModuleNotFoundError.
# Braces are doubled because this string is consumed by str.format().


# Values that must stay shell-expandable inside the job, because SLURM
# substitutes per-task. Only the variables in ALLOWED_SHELL_VARS expand; every
# other character is quoted literally (see quote_param_value).
SHELL_EXPANDED_PARAMS = frozenset({"mlflow-run-name", "dataset-index"})

# The only shell expansions allowed inside SHELL_EXPANDED_PARAMS values.
ALLOWED_SHELL_VARS = ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID")
_ALLOWED_VAR_RE = re.compile(r"\$(?:" + "|".join(ALLOWED_SHELL_VARS) + r")\b")

# Historical CLI defaults per job kind, used when no --cluster-config is given.
# A cluster config's job_defaults section overrides these; explicit CLI flags
# override both.
JOB_KIND_FALLBACKS = {
    "generate": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
    "jaxtrain": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
    "torchtrain": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 1,
        "mem": "16G",
        "time": "00:30:00",
    },
    # Parameter recovery: one HSSM fit per array task, CPU-bound (numpyro).
    "recover": {
        "account": "default",
        "partition": "batch",
        "num_gpus": 0,
        "cores": 4,
        "mem": "16G",
        "time": "02:00:00",
    },
}

# Extra `uv run` flags per job kind. Recovery needs the `validate` dependency
# group (hssm), which the default environment deliberately does not carry.
UV_RUN_FLAGS = {"recover": "--group validate "}

DEFAULT_MODULES = ["python", "gcc"]

# The checkout this script lives in — baked into generated scripts so the job
# runs against the right uv project regardless of the submission directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SLURM wall-time syntax accepted for the `time` resource.
_SLURM_TIME_RE = re.compile(r"^(\d+|\d+:\d{2}|\d+:\d{2}:\d{2}|\d+-\d+(:\d{2}){0,2})$")


def safe_name(value) -> str:
    """Reduce a config-supplied name to something safe in a path or job name.

    The model name comes from a user-supplied pipeline YAML and ends up in the
    generated script's filename, the SLURM job name, and the log filenames.
    Path separators and ``..`` would let it escape the ``runs/`` directory, and
    SLURM job names should not contain whitespace or shell metacharacters.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(value)).lstrip(".")
    return cleaned or "unnamed"


def absolutize_tracking_uri(uri: str) -> str:
    """Make a relative sqlite MLflow URI absolute.

    ``sqlite:///mlflow.db`` resolves against the *current* working directory,
    which differs between the submitting process and each compute node's job
    (SLURM starts jobs in the submission directory). Left relative, workers
    silently create their own database and the experiment id reported in the
    JSON contract points at a file nothing ever wrote to.
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri  # server URIs and already-absolute sqlite:////... are fine
    path_part = uri[len(prefix) :]
    if path_part.startswith("/"):
        return uri
    return f"{prefix}{Path(path_part).resolve()}"


def quote_param_value(key: str, value) -> str:
    """Quote a CLI parameter value for embedding in the sbatch script.

    Default: shlex.quote, so the value is one inert literal argument.

    For SHELL_EXPANDED_PARAMS the value must keep expanding SLURM's per-task
    variables, so it is split into segments: occurrences of ALLOWED_SHELL_VARS
    are emitted bare (inside double quotes) and *everything else* is
    shlex.quoted. Adjacent quoted segments concatenate into one shell word, so
    e.g. `ddm-worker-$SLURM_ARRAY_TASK_ID` becomes

        'ddm-worker-'"$SLURM_ARRAY_TASK_ID"

    Blanket double-quoting would be wrong here: the model name comes from a
    user-supplied pipeline YAML, and inside double quotes bash still executes
    backticks and `$(...)` and expands every other `$VAR` — so a model named
    ``ddm`hostname` `` would run a command on the compute node.
    """
    text = str(value)
    if key not in SHELL_EXPANDED_PARAMS:
        return shlex.quote(text)

    segments = []
    position = 0
    for match in _ALLOWED_VAR_RE.finditer(text):
        if match.start() > position:
            segments.append(shlex.quote(text[position : match.start()]))
        segments.append(f'"{match.group(0)}"')
        position = match.end()
    if position < len(text):
        segments.append(shlex.quote(text[position:]))
    return "".join(segments) if segments else "''"


def create_command(command_name: str, **params) -> str:
    """Create CLI command string from parameters, shell-quoted.

    A list value renders the flag once per element (``--mlflow-tag a=1
    --mlflow-tag b=2``); a bare ``True`` renders the flag alone.
    """
    parts = [command_name]
    for key, value in params.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is True:
                parts.append(f"--{key}")
            else:
                parts.append(f"--{key} {quote_param_value(key, item)}")
    return " ".join(parts)


def _artifact_location_for(tracking_uri: str, raw: str | None, logger=None):
    """Artifact root to pass to ``create_experiment``, or None.

    Against an http(s) tracking server the *server* owns artifact storage
    (``--serve-artifacts``); a client-side path would be recorded in the
    experiment and then fail on every compute node. A scheme-bearing value
    (``s3://``, ``gs://``, ``file://``) passes through untouched; a plain path
    is made absolute so submitter and workers agree on it.
    """
    if not raw:
        return None
    if tracking_uri.startswith(("http://", "https://")):
        if "://" not in raw:
            if logger is not None:
                logger.warning(
                    f"Ignoring MLFLOW_ARTIFACT_LOCATION={raw!r}: the tracking server "
                    f"{tracking_uri} owns artifact storage."
                )
            return None
        return raw
    if "://" in raw:
        return raw
    return str(Path(raw).absolute())


def _ensure_experiment(name: str, artifact_location, logger) -> str:
    """Get-or-create an MLflow experiment; return its id."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        if artifact_location:
            mlflow.create_experiment(name, artifact_location=artifact_location)
            logger.info(f"Created experiment: {name} (artifacts: {artifact_location})")
        else:
            mlflow.create_experiment(name)
            logger.info(f"Created experiment: {name} (default artifact location)")
        experiment = mlflow.get_experiment_by_name(name)
    mlflow.set_experiment(name)
    return experiment.experiment_id


def parse_mlflow_tags(raw_tags: list[str] | None) -> dict[str, str]:
    """Parse repeatable ``--mlflow-tag KEY=VALUE`` into a dict.

    Reserved schema tags are refused here, matching ssm-simulators' own check,
    so a bad value fails before any script is written.
    """
    tags: dict[str, str] = {}
    for raw in raw_tags or []:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            raise typer.BadParameter(
                f"--mlflow-tag expects KEY=VALUE, got {raw!r}",
                param_hint="--mlflow-tag",
            )
        if key in xp.RESERVED_MLFLOW_TAGS:
            raise typer.BadParameter(
                f"--mlflow-tag may not set the reserved schema tag {key!r}",
                param_hint="--mlflow-tag",
            )
        tags[key] = value
    return tags


def _load_experiment_or_die(path: Path | None):
    if path is None:
        return None
    try:
        return xp.load_experiment(path)
    except xp.ExperimentError as e:
        raise typer.BadParameter(str(e), param_hint="--experiment") from e


def load_cluster_config(cluster_config_path: Path) -> dict:
    """The committed cluster config, with the per-user file merged over it.

    `<name>.yaml` holds facts that are the same for the whole lab; the
    gitignored `<name>.local.yaml` beside it holds one person's associations,
    caps and lanes, written by scripts/discover_cluster.py. Merging here means
    nobody has to remember to pass a second flag, and no personal value ever
    needs to be committed.
    """
    with open(cluster_config_path, "rb") as f:
        config = yaml.safe_load(f) or {}

    local_path = cluster_config_path.with_suffix(".local.yaml")
    if not local_path.exists():
        return config

    with open(local_path, "rb") as f:
        local = yaml.safe_load(f) or {}
    for key, value in local.items():
        if key == "job_defaults" and isinstance(value, dict):
            merged = dict(config.get("job_defaults") or {})
            for kind, overrides in value.items():
                merged[kind] = {**(merged.get(kind) or {}), **(overrides or {})}
            config["job_defaults"] = merged
        else:
            config[key] = value
    return config


def split_across_lanes(lanes: list[dict], n_jobs: int) -> list[tuple[dict, int]]:
    """Divide an array job across lanes in proportion to their core caps.

    Each lane's cap is a separate SLURM budget, so the capacity really adds up
    — but lanes differ in priority, and a low-priority lane may sit in the
    queue. Bigger, higher-priority lanes therefore get proportionally more of
    the array, and the remainder goes to the best lane.
    """
    usable = [x for x in lanes if x.get("max_cores")]
    if len(usable) < 2 or n_jobs < 2:
        return [(lanes[0] if lanes else {}, n_jobs)]

    ordered = sorted(usable, key=lambda x: (-x.get("priority", 0), -x["max_cores"]))
    total = sum(x["max_cores"] for x in ordered)
    shares = [max(1, n_jobs * x["max_cores"] // total) for x in ordered]

    # Trim or pad to land exactly on n_jobs, always adjusting the best lane.
    while sum(shares) > n_jobs:
        biggest = shares.index(max(shares))
        shares[biggest] -= 1
    shares[0] += n_jobs - sum(shares)
    return [(lane, size) for lane, size in zip(ordered, shares) if size > 0]


def resolve_resources(
    command_name: str,
    cluster_config_path: Path | None,
    cli_values: dict,
    logger=None,
) -> dict:
    """Resolve job resources: builtin fallbacks < cluster config < CLI flags.

    Within the cluster config, a per-job-kind ``modules`` list beats the
    top-level one (specific beats generic), and an explicitly empty list means
    "load no modules" — distinct from the key being absent, which means "use
    the defaults".

    cli_values holds only flags the user actually passed (None = not passed).
    """
    resolved = dict(JOB_KIND_FALLBACKS[command_name])
    modules = list(DEFAULT_MODULES)

    if cluster_config_path is not None:
        cluster_config = load_cluster_config(cluster_config_path)
        job_defaults = (cluster_config.get("job_defaults") or {}).get(command_name, {})

        known = set(resolved) | {"modules", "lanes"}
        unknown = set(job_defaults) - known
        if unknown and logger is not None:
            logger.warning(
                f"Ignoring unknown job_defaults keys for {command_name!r} in "
                f"{cluster_config_path}: {sorted(unknown)}"
            )
        resolved.update({k: v for k, v in job_defaults.items() if k in resolved})
        if job_defaults.get("lanes"):
            resolved["lanes"] = job_defaults["lanes"]

        # Presence, not truthiness: `modules: []` disables module loading.
        if "modules" in cluster_config:
            modules = list(cluster_config["modules"] or [])
        if "modules" in job_defaults:
            modules = list(job_defaults["modules"] or [])

    resolved.update({k: v for k, v in cli_values.items() if v is not None})
    resolved["modules"] = modules
    validate_resources(resolved, cluster_config_path)
    return resolved


def validate_resources(resolved: dict, source) -> None:
    """Reject resource values SLURM would silently misread.

    Wall times must be **quoted strings** in YAML. Any bare number is
    ambiguous once PyYAML has parsed it, because YAML 1.1 reads colon-separated
    digits as sexagesimal: ``12:00:00`` and ``0:30`` load as the integers 43200
    and 30, indistinguishable from someone writing 43200 or 30 minutes
    directly. SLURM reads a bare integer as minutes, so an unquoted
    ``time: 12:00:00`` silently becomes a 30-day request (43200 minutes) and
    ``time: 4:00``, meant as 4 minutes, becomes 4 hours.

    Rather than guess which reading was intended, require the quoted form.
    """
    time_value = resolved.get("time")
    if isinstance(time_value, int):
        hours, remainder = divmod(time_value, 3600)
        minutes, seconds = divmod(remainder, 60)
        raise typer.BadParameter(
            f"Wall time in {source} must be a quoted string; YAML parsed it as "
            f"the integer {time_value}. If you meant {time_value} minutes write "
            f'time: "{time_value}"; if you meant '
            f"{hours:02d}:{minutes:02d}:{seconds:02d} write "
            f'time: "{hours:02d}:{minutes:02d}:{seconds:02d}". '
            "(YAML 1.1 reads 12:00:00 as the integer 43200, which SLURM would "
            "take as 43200 minutes.)"
        )
    if not _SLURM_TIME_RE.match(str(time_value)):
        raise typer.BadParameter(
            f"Invalid SLURM wall time {time_value!r}. Expected forms: "
            "MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS."
        )


def validate_log_path(path) -> None:
    """Reject SLURM log paths SLURM cannot express.

    The path is interpolated into ``#SBATCH --output=`` / ``--error=``, whose
    value SLURM takes literally to the end of the line without dequoting — so
    there is no quoting that makes whitespace work. Fail at generation time
    with the reason rather than at submission with a parse error.
    """
    text = str(path)
    if any(character.isspace() for character in text):
        raise typer.BadParameter(
            f"Output path {text!r} contains whitespace. SLURM writes job logs "
            "via #SBATCH --output=/--error=, which it reads literally (no "
            "quoting or escaping is honored), so a path with spaces cannot be "
            "expressed. Use a path without whitespace."
        )


def create_sbatch_script(
    account="default",
    partition="batch",
    num_gpus=0,
    cores=1,
    mem="4G",
    job_name="job",
    output="output.txt",
    error="error.txt",
    time="01:00:00",
    command="",
    n_jobs_in_array=1,
    env_vars="",
    modules=None,
    project_root=None,
    uv_run_flags="",
):
    # `is None` (absent), not falsy: an empty list means "load no modules".
    module_list = DEFAULT_MODULES if modules is None else modules
    module_loads = "\n".join(f"module load {module}" for module in module_list)
    sbatch_script = SBATCH_TEMPLATE.format(
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        job_name=job_name,
        time=time,
        # Deliberately NOT quoted: SLURM reads a #SBATCH directive's value
        # literally and does not dequote it, so quotes would end up in the
        # filename. Paths containing whitespace are rejected upfront instead
        # (validate_log_path).
        output=str(output),
        error=str(error),
        command=command,
        n_jobs_in_array=n_jobs_in_array,
        env_vars=env_vars,
        module_loads=module_loads,
        project_root=shlex.quote(str(project_root or PROJECT_ROOT)),
        uv_run_flags=uv_run_flags,
    )
    return sbatch_script


def write_sbatch(script, sbatch_script):
    with open(script, "w") as f:
        f.write(sbatch_script)


def submit_sbatch(script, logger) -> int | None:
    """Submit the script via sbatch; return the SLURM job id, or None on failure.

    The job id is parsed from sbatch's "Submitted batch job <id>" stdout line.
    Failures (nonzero exit, missing sbatch binary, unparseable output) are
    logged and yield None — the caller decides the process exit code.
    """
    try:
        result = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, check=False
        )
    except OSError as e:  # sbatch binary missing / not executable
        logger.error(f"Failed to submit job: {e}")
        return None
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.error(result.stderr.strip())
    if result.returncode != 0:
        logger.error(f"sbatch exited with code {result.returncode}")
        return None
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match is None:
        logger.error(
            "sbatch succeeded but its output did not contain "
            f"'Submitted batch job <id>': {result.stdout!r}"
        )
        return None
    return int(match.group(1))


def get_basic_config_from_yaml(yaml_config_path: str | Path):
    with open(yaml_config_path, "rb") as f:
        basic_config = yaml.safe_load(f)
    return basic_config


def get_parameters_setup(
    command: str,
    config_path: Path,
    output_path: Path,
    log_level: str,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
    # MLflow parameters - different for generate vs training
    mlflow_run_name: str = None,  # For data generation (ssm-simulators)
    mlflow_experiment_name: str = None,  # For data generation (ssm-simulators)
    mlflow_run_id: str = None,  # For training (LANfactory) - to resume a run
    data_generation_experiment_id: str = None,  # For training lineage
    lineage_id: str = None,  # schema v2: shared by every stage of an experiment
    mlflow_tags: dict | None = None,  # extra --mlflow-tag KEY=VALUE
    recover: dict | None = None,  # recover-only settings (see handle_job)
):
    """
    Prepare CLI arguments for the command based on the command type.

    For data generation (ssm-simulators):
        - Uses --mlflow-run-name and --mlflow-experiment-name
        - Each worker creates its own run

    For training (LANfactory):
        - Uses --mlflow-run-id to continue logging to parent run
        - Uses --data-generation-experiment-id for lineage tracking
    """
    params = {"config-path": config_path.resolve(), "log-level": log_level}

    if command == "generate":
        params["output"] = output_path.resolve()
        # ssm-simulators uses --mlflow-run-name and --mlflow-experiment-name
        if mlflow_run_name:
            params["mlflow-run-name"] = mlflow_run_name
        if mlflow_experiment_name:
            params["mlflow-experiment-name"] = mlflow_experiment_name

    elif command in ["jaxtrain", "torchtrain"]:
        params.update(
            {
                "networks-path-base": output_path.resolve(),
                "network-id": network_id,
                "dl-workers": dl_workers,
            }
        )
        # Optional since LANfactory's MLflow-first mode: with only
        # --data-generation-experiment-id the trainer derives the folder from
        # the datagen runs' `data_output_folder` param.
        if training_data_folder is not None:
            params["training-data-folder"] = training_data_folder.resolve()
        if mlflow_experiment_name:
            params["mlflow-experiment-name"] = mlflow_experiment_name
        # LANfactory uses --mlflow-run-id to resume the parent run
        if mlflow_run_id:
            params["mlflow-run-id"] = mlflow_run_id
        # Add data generation experiment ID for training lineage
        if data_generation_experiment_id:
            params["data-generation-experiment-id"] = data_generation_experiment_id

    elif command == "recover":
        rec = recover or {}
        params = {"log-level": log_level}
        params.update(
            {
                "model": rec["model"],
                "design": rec["design"],
                "likelihood": rec["likelihood"],
                # Expanded per array task; see SHELL_EXPANDED_PARAMS.
                "dataset-index": "$SLURM_ARRAY_TASK_ID",
                "out-dir": output_path.resolve(),
                "draws": rec["draws"],
                "tune": rec["tune"],
                "chains": rec["chains"],
                "target-accept": rec["target_accept"],
            }
        )
        if rec.get("onnx_path"):
            params["onnx-path"] = Path(rec["onnx_path"]).resolve()
        if rec.get("p_outlier") is not None:
            params["p-outlier"] = rec["p_outlier"]
        if rec.get("condition_param"):
            params["condition-param"] = rec["condition_param"]
        if mlflow_experiment_name:
            params["mlflow-experiment-name"] = mlflow_experiment_name
        if mlflow_run_name:
            params["mlflow-run-name"] = mlflow_run_name
        if rec.get("mlflow_run_id_train"):
            params["mlflow-run-id-train"] = rec["mlflow_run_id_train"]

    if lineage_id:
        params["lineage-id"] = lineage_id
    if mlflow_tags:
        params["mlflow-tag"] = [f"{k}={v}" for k, v in mlflow_tags.items()]

    return params


def handle_job(
    command_name: str,
    config_path: Path,
    output_path: Path,
    log_level: str,
    script_only: bool,
    account: str | None = None,
    partition: str | None = None,
    num_gpus: int | None = None,
    cores: int | None = None,
    mem: str | None = None,
    time: str | None = None,
    cluster_config: Path = None,
    n_jobs_in_array: int = 1,
    n_files: int = None,
    use_all_lanes: bool = False,
    training_data_folder: Path = None,
    network_id: int = 0,
    dl_workers: int = 1,
    data_generation_experiment_id: str = None,
    lineage_id: str = None,
    experiment_names: dict | None = None,
    mlflow_tags: dict | None = None,
    recover: dict | None = None,
):
    """Render (and unless script_only, submit) one job kind.

    ``experiment_names`` maps phase -> MLflow experiment name; defaults derive
    from the config's MODEL as they always have. ``lineage_id`` is rendered
    into the command when known; for ``generate`` without one, a fresh id is
    minted here (real submissions only) so every array task shares it.
    ``recover`` carries the recovery-specific settings (model, design,
    likelihood, onnx_path, sampler settings, mlflow_run_id_train).
    """
    # Case-insensitive, and validated: getattr(logging, "info") returns the
    # *function*, which basicConfig rejects with a TypeError traceback — and
    # the JSON contract line would never be printed.
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(
            f"Unknown log level {log_level!r}. Expected one of: "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logger = logging.getLogger("gen_sbatch")
    target = output_path.resolve()

    resources = resolve_resources(
        command_name,
        cluster_config,
        {
            "account": account,
            "partition": partition,
            "num_gpus": num_gpus,
            "cores": cores,
            "mem": mem,
            "time": time,
        },
        logger=logger,
    )

    # Initialize MLflow parameters
    mlflow_run_id = None  # For training: parent run ID
    mlflow_run_name = None  # For data generation / recovery: run name template
    mlflow_experiment_name = None  # experiment the workers log into
    mlflow_experiment_id = None  # to report to user
    env_vars = ""

    basic_config = get_basic_config_from_yaml(config_path.resolve())
    model_name = basic_config.get("MODEL", "unknown")
    names = dict(xp.DEFAULT_EXPERIMENT_NAMES)
    names.update(experiment_names or {})
    names = {k: v.format(model=model_name) for k, v in names.items()}
    phase_of = {
        "generate": "data_generation",
        "jaxtrain": "training",
        "torchtrain": "training",
        "recover": "inference",
    }
    experiment_name = names[phase_of[command_name]]

    # The lineage id must be identical across the array, so it is decided here,
    # once, and rendered into the script. Only a real datagen submission mints;
    # training/recovery inherit it (from the experiment, or LANfactory reads it
    # out of the training pickles), and --script-only never invents state.
    if command_name == "generate" and lineage_id is None and not script_only:
        lineage_id = xp.mint_lineage_id()
        logger.info(f"Minted lineage id: {lineage_id}")

    # Worker run names. Expanded per task by SLURM (see SHELL_EXPANDED_PARAMS).
    if command_name == "generate":
        mlflow_run_name = f"{model_name}-worker-$SLURM_ARRAY_TASK_ID"
    elif command_name == "recover":
        rec = recover or {}
        mlflow_run_name = (
            f"{model_name}-recovery-{rec.get('design', 'design')}-"
            f"{rec.get('likelihood', 'arm')}-$SLURM_ARRAY_TASK_ID"
        )

    # --script-only is side-effect-free: no experiment is created, no run is
    # started, and no MLflow wiring is embedded in the generated script. A
    # previous version created an empty orphan run per --script-only call.
    if MLFLOW_AVAILABLE and not script_only:
        try:
            # Set tracking URI from environment or use default (SQLite)
            # Absolute, so the submitting process and every compute-node
            # worker agree on which database the experiment lives in.
            tracking_uri = absolutize_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            )
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

            artifact_location = _artifact_location_for(
                tracking_uri, os.getenv("MLFLOW_ARTIFACT_LOCATION"), logger
            )

            mlflow_experiment_id = _ensure_experiment(
                experiment_name, artifact_location, logger
            )
            mlflow_experiment_name = experiment_name
            logger.info(
                f"{phase_of[command_name].replace('_', ' ').title()} experiment: "
                f"{experiment_name} (ID: {mlflow_experiment_id})"
            )

            if command_name in ["jaxtrain", "torchtrain"]:
                # Create a parent run for the training job. The SBATCH worker
                # continues logging to this run via --mlflow-run-id. Schema
                # tags go on now so the run is self-describing even if the
                # worker never starts; LANfactory re-sets identical values.
                run = mlflow.start_run(run_name=f"{command_name}_network_{network_id}")
                mlflow_run_id = run.info.run_id
                mlflow.log_param("command", command_name)
                mlflow.log_param("config_path", str(config_path))
                mlflow.log_param("output_path", str(output_path))
                mlflow.log_param("network_id", network_id)
                tags = {"schema_version": "2", "phase": "train"}
                if lineage_id:
                    tags["lineage_id"] = lineage_id
                if data_generation_experiment_id:
                    tags["data_generation_experiment_id"] = (
                        data_generation_experiment_id
                    )
                    logger.info(
                        f"Linked to data generation experiment: "
                        f"{data_generation_experiment_id}"
                    )
                tags.update(mlflow_tags or {})
                mlflow.set_tags(tags)
                logger.info(f"MLflow training run initialized: {mlflow_run_id}")
            # generate and recover: no parent run, each array task creates its own.

            env_vars = (
                f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(experiment_name)}\n"
                f"export MLFLOW_TRACKING_URI={shlex.quote(tracking_uri)}"
            )
            if artifact_location:
                env_vars += (
                    "\nexport MLFLOW_ARTIFACT_LOCATION="
                    f"{shlex.quote(artifact_location)}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")

    job_name_base = f"{safe_name(model_name)}_{command_name}_sbatch"

    # Scripts and SLURM logs live under <output_path>/runs/, timestamped —
    # repeated invocations never overwrite each other (previously the script
    # landed in CWD under a fixed name).
    runs_dir = target / "runs"
    validate_log_path(runs_dir)
    runs_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    # One submission per lane. A lane is an (account, partition) pair whose
    # SLURM budget is independent of the others', so splitting an array across
    # lanes really does buy extra parallelism — see split_across_lanes.
    if use_all_lanes and resources.get("lanes"):
        plan = split_across_lanes(resources["lanes"], n_jobs_in_array)
    else:
        plan = [({}, n_jobs_in_array)]
    fanned_out = len(plan) > 1

    if fanned_out:
        logger.warning(
            f"Fanning {n_jobs_in_array} array tasks across {len(plan)} lanes: "
            + ", ".join(
                f"{lane.get('account')}/{lane.get('partition')}={size}"
                for lane, size in plan
            )
        )

    failures = 0
    for index, (lane, array_size) in enumerate(plan):
        account = lane.get("account", resources["account"])
        partition = lane.get("partition", resources["partition"])

        lane_run_name = mlflow_run_name
        if fanned_out and mlflow_run_name:
            # Two arrays both number their tasks from 1, so without the lane
            # index the MLflow run names would collide across lanes.
            lane_run_name = mlflow_run_name.replace("-worker-", f"-worker-l{index}-")

        params = get_parameters_setup(
            command=command_name,
            config_path=config_path,
            output_path=output_path,
            log_level=log_level,
            training_data_folder=training_data_folder,
            network_id=network_id,
            dl_workers=dl_workers,
            mlflow_run_name=lane_run_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_id=mlflow_run_id,
            data_generation_experiment_id=data_generation_experiment_id,
            lineage_id=lineage_id,
            mlflow_tags=mlflow_tags,
            recover=recover,
        )
        if command_name == "generate" and n_files is not None:
            params["n-files"] = n_files
        if command_name == "generate":
            # State the worker count instead of letting the generator infer it.
            # Its "all" default reads sched_getaffinity, which is correct under
            # SLURM but grabs the whole machine anywhere else — and either way
            # the number never appears in the submission record. Passing the
            # cores we actually requested keeps the two in step and puts the
            # value in the generated script and the JSON line.
            params["n-cpus"] = resources["cores"]
        command = create_command(command_name, **params)
        logger.info(f"Generated command: {command}")

        job_name = f"{job_name_base}_l{index}" if fanned_out else job_name_base
        script = runs_dir / f"{timestamp}_{job_name}.sh"

        sbatch_script = create_sbatch_script(
            account=account,
            partition=partition,
            num_gpus=resources["num_gpus"],
            cores=resources["cores"],
            mem=resources["mem"],
            job_name=job_name,
            # %A_%a = array job id + task index. The template always emits
            # --array, so every task would otherwise append to one shared pair
            # of files and interleave its output with the others'.
            output=str(runs_dir / f"{job_name}_%A_%a.out"),
            error=str(runs_dir / f"{job_name}_%A_%a.err"),
            time=resources["time"],
            command=command,
            n_jobs_in_array=array_size,
            env_vars=env_vars,
            modules=resources["modules"],
            uv_run_flags=UV_RUN_FLAGS.get(command_name, ""),
        )
        write_sbatch(script, sbatch_script)

        # One machine-readable line per submission, regardless of log level:
        # the laptop driver's API. Logging goes to stderr, so stdout carries
        # only these. A single-lane run emits exactly one line, as before.
        result_record = {
            "command": command,
            "job_id": None,
            "mlflow_experiment_id": mlflow_experiment_id,
            "mlflow_run_id": mlflow_run_id,
            "sbatch_script": str(script),
            "output_path": str(target),
            "account": account,
            "partition": partition,
            "array_size": array_size,
            "lane": index,
            "n_lanes": len(plan),
            # Additive (schema v2): the id every stage of this experiment shares.
            "lineage_id": lineage_id,
        }

        if script_only:
            logger.info(f"Generated sbatch script: {script}")
            print(json.dumps(result_record))
            continue

        job_id = submit_sbatch(script, logger)
        result_record["job_id"] = job_id
        print(json.dumps(result_record))
        if job_id is None:
            failures += 1

    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.end_run()

    if script_only:
        return

    if command_name == "generate":
        logger.info(f"Simulated data output folder: {target}")
    elif command_name == "recover":
        logger.info(f"Recovery shards output folder: {target}")
    else:
        logger.info(f"Trained networks output folder: {target}")

    if failures:
        # Partial success is still a failure for the caller: some lanes are
        # running and some are not, and only the JSON lines say which.
        logger.error(f"{failures} of {len(plan)} lane submissions failed")
        raise typer.Exit(code=1)
    logger.info("Job submitted successfully")


EXPERIMENT_OPTION = typer.Option(
    None,
    "--experiment",
    help="Experiment directory (from `lan-sbatch init`). Supplies the config, "
    "output path, model, lineage id, experiment names and tags; explicit "
    "flags override it.",
)
LINEAGE_OPTION = typer.Option(
    None,
    "--lineage-id",
    help="Lineage id shared by every stage of one experiment (MLflow tag "
    "`lineage_id`). Default: the experiment's; for a plain `generate` a fresh "
    "id is minted on submission.",
)
MLFLOW_TAG_OPTION = typer.Option(
    None,
    "--mlflow-tag",
    help="Extra MLflow tag as KEY=VALUE, repeatable. Reserved schema tags "
    "(schema_version, phase, lineage_id) are refused.",
)


def _resolve_job_inputs(
    command_name: str,
    experiment,
    *,
    config_path: Path | None,
    output_path: Path | None,
    lineage_id: str | None,
    mlflow_tags: dict,
    script_only: bool,
    training_data_folder: Path | None = None,
    network_id: int | None = None,
    dl_workers: int | None = None,
    n_jobs_in_array: int | None = None,
    n_files: int | None = None,
    data_generation_experiment_id: str | None = None,
) -> dict:
    """Merge explicit flags over the experiment's values.

    Returns the keyword arguments for handle_job. Anything still unresolved
    that a job kind requires is reported as a CLI error.
    """
    names = None
    tags = dict(mlflow_tags)
    if experiment is not None:
        stage = "generate" if command_name == "generate" else "train"
        if command_name == "recover":
            stage = None
        config_path = config_path or (
            experiment.stage_config(stage)
            if stage
            else experiment.stage_config("train")
        )
        if output_path is None:
            output_path = (
                experiment.recovery_dir()
                if command_name == "recover"
                else experiment.stage_output(stage)
            )
        if lineage_id is None:
            lineage_id = experiment.lineage_id
            if lineage_id is None and not script_only:
                lineage_id = xp.ensure_lineage_id(experiment)
        names = experiment.experiment_names()
        tags = {**(experiment.mlflow.tags or {}), **tags}
        if command_name in ("jaxtrain", "torchtrain"):
            state = xp.read_state(experiment)
            if network_id is None:
                network_id = experiment.network.network_id
            if dl_workers is None:
                dl_workers = experiment.stages.train.dl_workers
            if training_data_folder is None and state.get("training_data_folder"):
                training_data_folder = Path(state["training_data_folder"])
            if data_generation_experiment_id is None:
                data_generation_experiment_id = state.get(
                    "data_generation_experiment_id"
                )
        if command_name == "generate":
            if n_jobs_in_array is None:
                n_jobs_in_array = experiment.stages.generate.n_jobs_in_array
            if n_files is None:
                n_files = experiment.stages.generate.n_files

    if config_path is None:
        raise typer.BadParameter(
            "Missing --config-path (or pass --experiment DIR).",
            param_hint="--config-path",
        )
    if output_path is None:
        raise typer.BadParameter(
            "Missing --output-path (or pass --experiment DIR).",
            param_hint="--output-path",
        )
    if (
        command_name in ("jaxtrain", "torchtrain")
        and training_data_folder is None
        and data_generation_experiment_id is None
    ):
        raise typer.BadParameter(
            "Training needs --training-data-folder, or "
            "--data-generation-experiment-id so LANfactory can derive the folder "
            "from the data-generation runs (MLflow-first mode).",
            param_hint="--training-data-folder",
        )

    return {
        "config_path": config_path,
        "output_path": output_path,
        "lineage_id": lineage_id,
        "experiment_names": names,
        "mlflow_tags": tags or None,
        "training_data_folder": training_data_folder,
        "network_id": 0 if network_id is None else network_id,
        "dl_workers": 1 if dl_workers is None else dl_workers,
        "n_jobs_in_array": 1 if n_jobs_in_array is None else n_jobs_in_array,
        "n_files": n_files,
        "data_generation_experiment_id": data_generation_experiment_id,
    }


@app.command()
def init(
    name: str = typer.Argument(..., help="Experiment name; becomes the directory name"),
    directory: Path = typer.Option(
        Path("experiments"), "--dir", help="Parent directory for the experiment"
    ),
    model: str = typer.Option("ddm", help="ssm-simulators model to simulate/train/fit"),
    quick: bool = typer.Option(
        False, "--quick", help="Copy the small configs/quick_test templates"
    ),
    network_type: str = typer.Option("lan", help="lan | cpn | opn | gonogo"),
    trainer: str = typer.Option("jaxtrain", help="jaxtrain | torchtrain"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite a non-empty directory"
    ),
):
    """Scaffold an experiment directory: experiment.yaml plus stage configs."""
    try:
        exp = xp.scaffold_experiment(
            name,
            directory,
            model=model,
            quick=quick,
            network_type=network_type,
            trainer=trainer,
            force=force,
        )
    except xp.ExperimentError as e:
        raise typer.BadParameter(str(e)) from e
    print(
        json.dumps(
            {
                "experiment_dir": str(exp.dir),
                "lineage_id": exp.lineage_id,
                "experiments": exp.experiment_names(),
            }
        )
    )


@app.command()
def generate(
    config_path: Path = typer.Option(
        None, help="Path to configuration .yaml file for running commands"
    ),
    output_path: Path = typer.Option(
        None, help="Path to output folder for simulated data"
    ),
    experiment: Path = EXPERIMENT_OPTION,
    lineage_id: str = LINEAGE_OPTION,
    mlflow_tag: list[str] = MLFLOW_TAG_OPTION,
    n_jobs_in_array: int = typer.Option(
        None, help="Size of the job array [default: 1]"
    ),
    n_files: int = typer.Option(
        None, help="Number of files each worker generates (passed to `generate`)"
    ),
    cluster_config: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Cluster inventory YAML (e.g. configs/cluster/oscar.yaml); its "
        "job_defaults section supplies account/partition/resources for this "
        "job kind. A gitignored <name>.local.yaml beside it, written by "
        "scripts/discover_cluster.py, is merged over it. Explicit flags "
        "below override both.",
    ),
    use_all_lanes: bool = typer.Option(
        False,
        "--use-all-lanes",
        is_flag=True,
        help="Split the job array across every lane in the cluster config, "
        "proportional to each lane's core cap. Lane budgets are independent "
        "in SLURM, so this adds capacity — but lower-priority lanes queue "
        "longer, so treat the extra as spillover. Emits one JSON line per "
        "lane.",
    ),
    account: str = typer.Option(
        None, help="Condo to run the SBATCH job on [default: from cluster config]"
    ),
    partition: str = typer.Option(
        None,
        help="Partition to run the SBATCH script on [default: from cluster config]",
    ),
    num_gpus: int = typer.Option(
        None, help="Number of GPUs requested [default: from cluster config]"
    ),
    mem: str = typer.Option(
        None, help="Memory limit for each job [default: from cluster config]"
    ),
    time: str = typer.Option(
        None, help="Wall time limit for each job [default: from cluster config]"
    ),
    cores: int = typer.Option(
        None, help="Number of cores per job [default: from cluster config]"
    ),
    script_only: bool = typer.Option(
        False,
        help="Generate the sbatch script without submitting the job. "
        "Side-effect-free: no MLflow experiment/run is created and no MLflow "
        "wiring is embedded in the script.",
    ),
    log_level: str = typer.Option(
        "WARNING", help="Set the log level", show_default=True
    ),
):
    """Generate SBATCH script for data generation using ssm-simulators."""
    inputs = _resolve_job_inputs(
        "generate",
        _load_experiment_or_die(experiment),
        config_path=config_path,
        output_path=output_path,
        lineage_id=lineage_id,
        mlflow_tags=parse_mlflow_tags(mlflow_tag),
        script_only=script_only,
        n_jobs_in_array=n_jobs_in_array,
        n_files=n_files,
    )
    handle_job(
        command_name="generate",
        config_path=inputs["config_path"],
        output_path=inputs["output_path"],
        log_level=log_level,
        time=time,
        script_only=script_only,
        cluster_config=cluster_config,
        account=account,
        partition=partition,
        num_gpus=num_gpus,
        cores=cores,
        mem=mem,
        n_jobs_in_array=inputs["n_jobs_in_array"],
        n_files=inputs["n_files"],
        use_all_lanes=use_all_lanes,
        lineage_id=inputs["lineage_id"],
        experiment_names=inputs["experiment_names"],
        mlflow_tags=inputs["mlflow_tags"],
    )


def train_command(command_name: str):
    def train(
        config_path: Path = typer.Option(
            None, help="Path to configuration .yaml file for running commands"
        ),
        output_path: Path = typer.Option(
            None, help="Path to output folder for trained neural network"
        ),
        training_data_folder: Path = typer.Option(
            None,
            help="Path to folder with data to train the neural network on. "
            "Optional with --data-generation-experiment-id (LANfactory derives "
            "it from the data-generation runs).",
        ),
        experiment: Path = EXPERIMENT_OPTION,
        lineage_id: str = LINEAGE_OPTION,
        mlflow_tag: list[str] = MLFLOW_TAG_OPTION,
        network_id: int = typer.Option(
            None, help="Id for the neural network to train [default: 0]"
        ),
        data_generation_experiment_id: str = typer.Option(
            None,
            "--data-generation-experiment-id",
            help="MLflow Experiment ID of the data generation experiment. "
            "If provided, training data lineage will be logged.",
        ),
        cluster_config: Path = typer.Option(
            None,
            exists=True,
            dir_okay=False,
            readable=True,
            help="Cluster inventory YAML (e.g. configs/cluster/oscar.yaml); its "
            "job_defaults section supplies account/partition/resources for "
            "this job kind. Explicit flags below override it.",
        ),
        account: str = typer.Option(
            None, help="Condo to run the SBATCH job on [default: from cluster config]"
        ),
        partition: str = typer.Option(
            None,
            help="Partition to run the SBATCH script on [default: from cluster config]",
        ),
        num_gpus: int = typer.Option(
            None, help="Number of GPUs requested [default: from cluster config]"
        ),
        cores: int = typer.Option(
            None, help="Number of cores per job [default: from cluster config]"
        ),
        dl_workers: int = typer.Option(
            None, help="Number of cores to use with the dataloader class [default: 1]"
        ),
        time: str = typer.Option(
            None, help="Wall time limit for each job [default: from cluster config]"
        ),
        mem: str = typer.Option(
            None, help="Memory limit for each job [default: from cluster config]"
        ),
        script_only: bool = typer.Option(
            False,
            help="Generate the sbatch script without submitting the job. "
            "Side-effect-free: no MLflow experiment/run is created and no "
            "MLflow wiring is embedded in the script.",
        ),
        log_level: str = typer.Option(
            "WARNING", help="Set the log level", show_default=True
        ),
    ):
        """Generate SBATCH script for neural network training using LANfactory."""
        inputs = _resolve_job_inputs(
            command_name,
            _load_experiment_or_die(experiment),
            config_path=config_path,
            output_path=output_path,
            lineage_id=lineage_id,
            mlflow_tags=parse_mlflow_tags(mlflow_tag),
            script_only=script_only,
            training_data_folder=training_data_folder,
            network_id=network_id,
            dl_workers=dl_workers,
            data_generation_experiment_id=data_generation_experiment_id,
        )
        handle_job(
            command_name=command_name,
            config_path=inputs["config_path"],
            output_path=inputs["output_path"],
            log_level=log_level,
            time=time,
            script_only=script_only,
            cluster_config=cluster_config,
            account=account,
            partition=partition,
            num_gpus=num_gpus,
            cores=cores,
            mem=mem,
            training_data_folder=inputs["training_data_folder"],
            network_id=inputs["network_id"],
            dl_workers=inputs["dl_workers"],
            data_generation_experiment_id=inputs["data_generation_experiment_id"],
            lineage_id=inputs["lineage_id"],
            experiment_names=inputs["experiment_names"],
            mlflow_tags=inputs["mlflow_tags"],
        )

    return train


@app.command()
def recover(
    experiment: Path = typer.Option(
        ..., "--experiment", help="Experiment directory (from `lan-sbatch init`)"
    ),
    onnx_path: Path = typer.Option(
        None,
        help="Network to test [default: state.json, else the newest ONNX the "
        "training stage wrote]",
    ),
    design: list[str] = typer.Option(
        None, "--design", help="Recovery design(s) [default: experiment.yaml]"
    ),
    likelihood: list[str] = typer.Option(
        None,
        "--likelihood",
        help="Arm(s) to fit [default: experiment.yaml]. One array per "
        "design x likelihood cell.",
    ),
    n_datasets: int = typer.Option(
        None, help="Datasets (array tasks) per cell [default: experiment.yaml]"
    ),
    lineage_id: str = LINEAGE_OPTION,
    mlflow_tag: list[str] = MLFLOW_TAG_OPTION,
    cluster_config: Path = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Cluster inventory YAML; job_defaults.recover supplies resources.",
    ),
    account: str = typer.Option(None, help="[default: from cluster config]"),
    partition: str = typer.Option(None, help="[default: from cluster config]"),
    cores: int = typer.Option(None, help="[default: from cluster config]"),
    mem: str = typer.Option(None, help="[default: from cluster config]"),
    time: str = typer.Option(None, help="[default: from cluster config]"),
    script_only: bool = typer.Option(
        False, help="Generate the sbatch scripts without submitting or touching MLflow."
    ),
    log_level: str = typer.Option(
        "WARNING", help="Set the log level", show_default=True
    ),
):
    """Submit parameter-recovery fits (HSSM) for an experiment's network.

    One Slurm array per design x likelihood cell; each task fits one synthetic
    dataset with `lan-recover --track`, so every fit lands in the experiment's
    inference experiment carrying the shared lineage id.
    """
    exp = _load_experiment_or_die(experiment)
    state = xp.read_state(exp)
    rec = exp.stages.recover
    tags = {**(exp.mlflow.tags or {}), **parse_mlflow_tags(mlflow_tag)}

    resolved_lineage = lineage_id or exp.lineage_id
    if resolved_lineage is None and not script_only:
        resolved_lineage = xp.ensure_lineage_id(exp)

    onnx = onnx_path or (
        Path(state["onnx_path"]) if state.get("onnx_path") else xp.find_onnx(exp)
    )
    likelihoods = list(likelihood or rec.likelihoods)
    if any(lk != "analytical" for lk in likelihoods) and onnx is None:
        raise typer.BadParameter(
            "No ONNX network found for the network arm: train first, or pass "
            "--onnx-path.",
            param_hint="--onnx-path",
        )
    if rec.add_reference_arm and "analytical" not in likelihoods and not likelihood:
        likelihoods.append("analytical")

    for design_name in list(design or rec.designs):
        for lk in likelihoods:
            handle_job(
                command_name="recover",
                config_path=exp.stage_config("train"),
                output_path=exp.recovery_dir(),
                log_level=log_level,
                time=time,
                script_only=script_only,
                cluster_config=cluster_config,
                account=account,
                partition=partition,
                num_gpus=0,
                cores=cores,
                mem=mem,
                n_jobs_in_array=n_datasets or rec.n_datasets,
                lineage_id=resolved_lineage,
                experiment_names=exp.experiment_names(),
                mlflow_tags=tags or None,
                recover={
                    "model": exp.model,
                    "design": design_name,
                    "likelihood": lk,
                    "onnx_path": str(onnx) if (onnx and lk != "analytical") else None,
                    "draws": rec.draws,
                    "tune": rec.tune,
                    "chains": rec.chains,
                    "target_accept": rec.target_accept,
                    "p_outlier": rec.p_outlier,
                    "condition_param": rec.condition_param,
                    "mlflow_run_id_train": state.get("mlflow_run_id_train"),
                },
            )


app.command("jaxtrain")(train_command("jaxtrain"))
app.command("torchtrain")(train_command("torchtrain"))


@app.command()
def run(
    experiment: Path = typer.Option(
        ..., "--experiment", help="Experiment directory (from `lan-sbatch init`)"
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run the stages on this machine as subprocesses (no Slurm).",
    ),
    stages: str = typer.Option(
        "generate,train,validate,recover,summary",
        "--stages",
        help="Comma-separated subset of generate,train,validate,recover,summary.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log the commands each stage would run; run nothing."
    ),
    continue_on_gate_failure: bool = typer.Option(
        False,
        "--continue-on-gate-failure",
        help="Keep going when lan-validate reports a failed gate.",
    ),
    log_level: str = typer.Option("INFO", help="Set the log level", show_default=True),
):
    """Run an experiment end to end on this machine (--local).

    Drives generate -> train -> validate -> recover -> summary from the
    experiment directory, passing the shared lineage id to every stage, and
    finishes by listing every MLflow run of the lineage and checking that the
    stages link to each other. Slurm chaining is not implemented; submit the
    stages individually with `generate`, `jaxtrain`/`torchtrain` and `recover`.
    """
    try:
        from sbatch_scripts import run_local as rl
    except ImportError:  # pragma: no cover - path import in tests
        import run_local as rl

    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(f"Unknown log level {log_level!r}.")
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    if not local:
        raise typer.BadParameter(
            "Only --local is implemented. On a cluster, submit the stages with "
            "`lan-sbatch generate/jaxtrain/torchtrain/recover --experiment DIR`.",
            param_hint="--local",
        )
    exp = _load_experiment_or_die(experiment)
    wanted = [s.strip() for s in stages.split(",") if s.strip()]
    try:
        state = rl.run_local(
            exp,
            wanted,
            dry_run=dry_run,
            continue_on_gate_failure=continue_on_gate_failure,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--stages") from e
    except rl.StageFailed as e:
        logging.getLogger("run_local").error(str(e))
        print(
            json.dumps(
                {"ok": False, "failed_stage": e.stage, "lineage_id": exp.lineage_id}
            )
        )
        raise typer.Exit(code=1)
    print(
        json.dumps(
            {
                "ok": True,
                "lineage_id": exp.lineage_id,
                "experiment_dir": str(exp.dir),
                "tracking_uri": xp.mlflow_tracking_uri(exp),
                "state": str(exp.dir / xp.STATE_FILE),
                **{
                    k: state.get(k)
                    for k in (
                        "phases",
                        "n_runs",
                        "validation_passed",
                        "recovery_verdict",
                    )
                },
            }
        )
    )


if __name__ == "__main__":
    app()
